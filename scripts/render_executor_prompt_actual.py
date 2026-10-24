"""渲染**已落地**的 Executor 提示词（2026-10-24 之后的形态）。

和 `render_executor_prompt_v3_mockup.py` 的区别：那份是**提案**（评审用的样张），
这份是**真代码跑出来的** —— 走生产的 `build_executor_messages` +
生产的 `contextor.render_action_result`，把「工具层递过来的失败事实」
一路渲染到模型看到的那一行。

跑法::

    uv run python scripts/render_executor_prompt_actual.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minitap"))
sys.stdout.reconfigure(encoding="utf-8")

from minitap.mobile_use.agents.contextor.contextor import render_action_result  # noqa: E402
from minitap.mobile_use.agents.executor.prompt import build_executor_messages  # noqa: E402
from minitap.mobile_use.config import LLM, LLMConfig, LLMConfigUtils, LLMWithFallback  # noqa: E402
from minitap.mobile_use.context import (  # noqa: E402
    DeviceContext,
    DevicePlatform,
    MobileUseContext,
)
from minitap.mobile_use.utils.screen_elements import ScreenElement  # noqa: E402
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds  # noqa: E402

GOAL = "进入 WLAN 页面并读出当前连接的 Wi-Fi 名称"


def _ctx(language: str = "zh") -> MobileUseContext:
    def llm() -> LLMWithFallback:
        return LLMWithFallback(
            provider="openai", model="gpt-5", fallback=LLM(provider="openai", model="o4-mini")
        )

    ctx = MobileUseContext(
        trace_id="probe",
        device=DeviceContext(
            host_platform="WINDOWS",
            mobile_platform=DevicePlatform.ANDROID,
            device_id="probe",
            device_width=1220,
            device_height=2712,
        ),
        llm_config=LLMConfig(
            planner=llm(), orchestrator=llm(), contextor=llm(), cortex=llm(), executor=llm(),
            utils=LLMConfigUtils(outputter=llm(), hopper=llm()),
        ),
    )
    ctx.prompt_language = language  # type: ignore[assignment]
    return ctx


def _elements(n: int = 8) -> list[ScreenElement]:
    labels = ["WLAN", "移动网络", "蓝牙", "更多设置", "WLAN", "已保存的网络", "开关", "信号强度"]
    out = []
    for i in range(n):
        out.append(
            ScreenElement(
                index=i + 1,
                kind="ui",
                label=labels[i % len(labels)],
                text="",
                resource_id=f"com.android.settings:id/row_{i}",
                display_id="",
                class_name="TextView",
                bounds=ElementBounds(x=60, y=480 + 160 * i, width=320, height=120),
                flags="C",
            )
        )
    return out


def render(language: str = "zh") -> str:
    ctx = _ctx(language)

    # ---- 工具层递过来的失败事实（形状见 tools/utils.last_action_failure_fact）----
    failure = {"label": "点赞", "resource_id": "com.xingin.app:id/like_button"}

    # ---- Contextor 并进「上一步的结果」那一行（真代码）----
    result_line = render_action_result(
        previous_elements=_elements(),
        previous_app="com.android.settings",
        current_elements=_elements(),
        current_app="com.android.settings",
        failure=failure,
    )

    # ---- 局部历史：动作 → 工具回执 → 上一步的结果 ----
    history = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "ExecutorOutput",
                "args": {
                    "actions": [{"action": "tap", "target": {"index": 12}}],
                    "agent_thought": "第 1 步：先点顶部的「WLAN」进去。",
                },
                "id": "call_00",
            }],
        ),
        ToolMessage(content="Tapped element (610, 300) successfully.", tool_call_id="call_00"),
        HumanMessage(
            content="[上一步的结果] 前台应用变成了 com.android.settings；"
            "屏幕上新出现了：WLAN、移动网络、蓝牙"
        ),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "ExecutorOutput",
                "args": {
                    "actions": [{"action": "tap", "target": {"index": 48}}],
                    "agent_thought": "第 2 步：点「点赞」按钮。",
                },
                "id": "call_01",
            }],
        ),
        ToolMessage(
            content=(
                "Tap failed: no element with resource_id="
                "'com.xingin.app:id/like_button' on the current screen"
            ),
            tool_call_id="call_01",
            status="error",
        ),
        HumanMessage(content=result_line or ""),
    ]

    # ---- 跨子目标失败事实（真代码渲染；注意：**没有 index**）----
    failures = [
        {
            "key": "index=48|id=com.xingin.app:id/like_button|text=点赞",
            "index": 48,
            "resource_id": "com.xingin.app:id/like_button",
            "label": "点赞",
            "reason": "no element with resource_id on the current screen",
        }
    ]

    msgs = build_executor_messages(
        ctx=ctx,
        goal=GOAL,
        screen_elements=_elements(),
        failures=failures,
        local_messages=history,
        device_state_lines=["设备日期：2026-10-24 10:05:33", "前台应用：com.android.settings"],
    )

    out: list[str] = []
    for m in msgs:
        role = type(m).__name__
        content = str(m.content)
        if role == "AIMessage":
            calls = getattr(m, "tool_calls", None)
            if calls:
                content = f"ExecutorOutput({calls[0]['args']})"
        out.append(f"[{role}] {content}")
    return "\n\n".join(out)


def main() -> int:
    for language in ("zh", "en"):
        print("=" * 78)
        print(f"已落地的 Executor 提示词（{language}）")
        print("=" * 78)
        print(render(language))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
