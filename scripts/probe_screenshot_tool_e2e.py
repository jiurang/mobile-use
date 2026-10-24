"""真机 + 真模型的端到端对照：截图工具到底有没有把「像素」送到模型眼前。

为什么要做 A/B 对照
--------------------
单看「HTTP 200 / 图在请求体里」只能证明**管道通**，不能证明**模型真看见了**。
所以这里跑同一台手机、同一个模型、同一道题，只差「历史里有没有那张图」：

    A 组：只有元素表 + OCR（现行能力）
    B 组：历史里多一条 ``take_screenshot`` 的工具结果（图挂在里面）
    C 组：不给图，看模型**自己**会不会喊 ``take_screenshot``

判据（三个都要看）：
1. **input_tokens 跳上去**（图 ≈500 token）⇒ 图真的进了请求体；
2. **B 组的答案里有 A 组没有的视觉细节**（颜色 / 形状 / 图案）⇒ 模型真看见了；
3. **C 组模型自己调了 take_screenshot** ⇒ 这条路在真实决策里走得通。

跑法：
    .venv\\Scripts\\python.exe scripts\\probe_screenshot_tool_e2e.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minitap"))

import adbutils  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from minitap.mobile_use.agents.cortex.dispatcher import dispatch_decisions_with_reason  # noqa: E402
from minitap.mobile_use.agents.contextor.contextor import ContextorNode  # noqa: E402
from minitap.mobile_use.agents.executor.prompt import build_executor_messages  # noqa: E402
from minitap.mobile_use.agents.executor.types import ExecutorOutput  # noqa: E402
from minitap.mobile_use.clients.ui_automator_client import UIAutomatorClient  # noqa: E402
from minitap.mobile_use.config import initialize_llm_config  # noqa: E402
from minitap.mobile_use.context import DeviceContext, DevicePlatform, MobileUseContext  # noqa: E402
from minitap.mobile_use.graph.state import State  # noqa: E402
from minitap.mobile_use.services.llm import get_structured_llm  # noqa: E402
from minitap.mobile_use.tools.index import EXECUTOR_WRAPPERS_TOOLS, get_tools_from_wrappers  # noqa: E402
from minitap.mobile_use.tools.mobile.take_screenshot import get_take_screenshot_tool  # noqa: E402
from minitap.mobile_use.utils.llm_usage import (  # noqa: E402
    close_capture,
    extract_usage,
    first_captured,
    open_capture,
)
from minitap.mobile_use.utils.screen_elements import render_screen_rows  # noqa: E402

DEVICE = "89R8ZXNBGQIRJRLF"
OUT_DIR = ROOT / "probe_out"

#: 这道题**只有看图才能答**：问的是角标**里面画的图案**（图形没有文字，
#: 元素表和 OCR 都给不出来）。
GOAL = (
    "Report what is drawn inside the round floating badge at the top of the "
    "'task list' card: what picture is on it? Do not guess - if you cannot "
    "see it, say so."
)


class _FakeState:
    """工具只需要 `asanitize_update` 和 `executor_messages`。"""

    def __init__(self, executor_messages=None):
        self.executor_messages = executor_messages or []

    async def asanitize_update(self, *, ctx, update, agent):
        return update


def _build_ctx() -> MobileUseContext:
    return MobileUseContext(
        trace_id="probe-screenshot-e2e",
        device=DeviceContext(
            host_platform="WINDOWS",
            mobile_platform=DevicePlatform.ANDROID,
            device_id=DEVICE,
            device_width=1220,
            device_height=2712,
        ),
        adb_client=adbutils.AdbClient(),
        ui_adb_client=UIAutomatorClient(device_id=DEVICE),
        llm_config=initialize_llm_config(),
    )


def _build_state() -> State:
    return State(
        messages=[],
        initial_goal=GOAL,
        subgoal_plan=[],
        agents_thoughts=[],
        latest_ui_hierarchy=None,
        latest_ocr_results=None,
        focused_app_info=None,
        device_date=None,
        structured_decisions=None,
        complete_subgoals_by_ids=[],
        cortex_last_thought=None,
        executor_messages=[],
    )


def _usage(response) -> dict:
    meta = getattr(response, "usage_metadata", None) or {}
    return {
        "input_tokens": meta.get("input_tokens"),
        "output_tokens": meta.get("output_tokens"),
    }


async def _ensure_awake(ctx) -> str:
    """屏幕灭着就先点亮 —— 否则元素表只剩 systemui 残留，测的就不是真东西了。

    真机实测（2026-10-24）：手机 30 分钟息屏，探针跑到一半屏已经黑了，
    元素表里全是 keyguard 残留 —— 这正是 ``wake_screen`` 工具存在的理由。
    """
    from minitap.mobile_use.controllers.unified_controller import UnifiedMobileController

    controller = UnifiedMobileController(ctx)
    power = await controller.get_screen_power_state()
    if power != "on":
        print(f"[probe] 屏幕状态 {power} → 先唤醒")
        await controller.wake_up()
        await asyncio.sleep(1.5)
        power = await controller.get_screen_power_state()
    print(f"[probe] 屏幕状态: {power}")
    return power


async def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = _build_ctx()
    await _ensure_awake(ctx)

    # ---- 真 contextor：抓屏 + 建元素表（和线上同一条路径）----
    state = _build_state()
    update = await ContextorNode(ctx)(state)
    elements = update.get("latest_screen_elements")
    ocr_results = update.get("latest_ocr_results")
    rows = render_screen_rows(elements or [])
    lines = rows.splitlines() if isinstance(rows, str) else list(rows)
    print(f"\n=== 当屏元素表（{len(lines)} 行，模型看到的就是这个）===")
    print("\n".join(lines[:22]))
    print("...")

    runner = get_structured_llm(ctx=ctx, name="executor", schema=ExecutorOutput)
    tools = get_tools_from_wrappers(ctx, list(EXECUTOR_WRAPPERS_TOOLS))

    async def _ask(messages) -> tuple[object, dict]:
        """问一次模型，并**同时**把 token 用量抓出来。

        结构化输出这条路返回的是 pydantic 对象，usage 落不到它身上 ——
        usage 在原始 AIMessage 上（见 ``utils/llm_usage.py`` 的捕获槽说明）。
        """
        sink = open_capture()
        try:
            response = await runner.ainvoke(messages)
        finally:
            close_capture()
        raw = first_captured(sink)
        usage = extract_usage(raw)
        return response, {
            "input_tokens": usage[0] if usage else None,
            "output_tokens": usage[1] if usage else None,
        }

    def _messages(local_messages):
        return build_executor_messages(
            ctx=ctx,
            goal=GOAL,
            screen_elements=elements,
            failures=[],
            local_messages=local_messages,
            device_state_lines=[],
            ocr_results=ocr_results,
        )

    results: dict[str, object] = {}

    # ================= A 组：没有图 =================
    resp_a, usage_a = await _ask(_messages([]))
    out_a = resp_a if isinstance(resp_a, ExecutorOutput) else None
    print("\n=== A 组（无图）===")
    print(f"thought: {getattr(out_a, 'agent_thought', '')}")
    print(f"actions: {getattr(out_a, 'actions', None)}")
    print(f"usage: {usage_a}")
    results["A_no_picture"] = {
        "thought": getattr(out_a, "agent_thought", ""),
        "actions": getattr(out_a, "actions", None),
        "done": getattr(out_a, "done", None),
        "done_reason": getattr(out_a, "done_reason", None),
        "usage": usage_a,
    }

    # ================= B 组：历史里带图 =================
    # 走真 dispatcher + 真工具，不手工拼消息
    tool_calls, reason = dispatch_decisions_with_reason(
        decisions=json.dumps([{"action": "take_screenshot"}]),
        default_thought="probe: take a picture so I can answer",
        tools=tools,
    )
    assert tool_calls, f"派发失败: {reason}"
    tool = get_take_screenshot_tool(ctx)
    cmd = await tool.coroutine(  # type: ignore[misc]
        agent_thought="probe: take a picture so I can answer",
        tool_call_id=tool_calls[0]["id"],
        state=_FakeState(),
    )
    tool_message = cmd.update["executor_messages"][-1]
    if isinstance(tool_message.content, list):
        print(f"\n工具结果块类型: {[b.get('type') for b in tool_message.content]}")
        # 把模型真正看到的那张图存下来（人眼可看）
        block = next(b for b in tool_message.content if b.get("type") == "image_url")
        url = block["image_url"]["url"]
        (OUT_DIR / "screenshot_tool_e2e.jpg").write_bytes(base64.b64decode(url.split(",", 1)[1]))
    else:
        print(f"\n工具结果（文字）: {tool_message.content}")

    history = [AIMessage(content="", tool_calls=tool_calls), tool_message]
    resp_b, usage_b = await _ask(_messages(history))
    out_b = resp_b if isinstance(resp_b, ExecutorOutput) else None
    print("\n=== B 组（历史里有图）===")
    print(f"thought: {getattr(out_b, 'agent_thought', '')}")
    print(f"actions: {getattr(out_b, 'actions', None)}")
    print(f"usage: {usage_b}")
    results["B_with_picture"] = {
        "thought": getattr(out_b, "agent_thought", ""),
        "actions": getattr(out_b, "actions", None),
        "done": getattr(out_b, "done", None),
        "done_reason": getattr(out_b, "done_reason", None),
        "usage": usage_b,
    }

    # ================= C 组：模型自己要不要图 =================
    resp_c, usage_c = await _ask(_messages([]))
    out_c = resp_c if isinstance(resp_c, ExecutorOutput) else None
    actions_c = getattr(out_c, "actions", None) or []
    called = [a.get("action") for a in actions_c]
    print("\n=== C 组（模型自己决定）===")
    print(f"thought: {getattr(out_c, 'agent_thought', '')}")
    print(f"actions: {actions_c}")
    print(f"usage: {usage_c}")
    results["C_model_decides"] = {
        "thought": getattr(out_c, "agent_thought", ""),
        "actions": actions_c,
        "called_take_screenshot": "take_screenshot" in called,
        "usage": usage_c,
    }

    # ---- 结论 ----
    delta = (usage_b.get("input_tokens") or 0) - (usage_a.get("input_tokens") or 0)
    print("\n=== 结论 ===")
    print(f"input_tokens A→B 增量: {delta:+d}  （一张图 ≈500 token ⇒ 增量里含图）")
    called_c = results["C_model_decides"]["called_take_screenshot"]
    print(f"C 组模型自己调了 take_screenshot: {'是' if called_c else '否（它选了别的动作）'}")

    (OUT_DIR / "screenshot_tool_e2e.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  => {OUT_DIR / 'screenshot_tool_e2e.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
