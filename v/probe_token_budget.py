"""实测 mobile-use 每轮 LLM 调用的上下文构成（token）。

不猜测，直接把各 agent 真正发送的 message list 渲染出来，用 tiktoken 计数。
工具 schema 也照 bind_tools 的实际 JSON 形态计入。

用法：python v/probe_token_budget.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

import tiktoken
from jinja2 import Template
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))

from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402

from minitap.mobile_use.agents.cortex.types import CortexOutput  # noqa: E402
from minitap.mobile_use.agents.planner.types import (  # noqa: E402
    PlannerOutput,
    Subgoal,
    SubgoalStatus,
)
from minitap.mobile_use.agents.planner.utils import generate_id  # noqa: E402
from minitap.mobile_use.tools.index import EXECUTOR_WRAPPERS_TOOLS  # noqa: E402
from minitap.mobile_use.utils.ui_hierarchy_view import render_ui_hierarchy  # noqa: E402

ENC = tiktoken.get_encoding("cl100k_base")


def ntok(text: str) -> int:
    return len(ENC.encode(text))


def hr(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------
# 构造一个「跑了几轮之后」的真实状态：5 个 subgoal、12 条 agent thoughts
# --------------------------------------------------------------------------
def make_plan(n: int = 5) -> list[Subgoal]:
    plan = []
    for i in range(n):
        plan.append(
            Subgoal(
                id=generate_id(),
                description=f"Subgoal {i + 1}: open the app and navigate to the settings page",
                status=SubgoalStatus.SUCCESS if i < 2 else SubgoalStatus.PENDING,
                completion_reason="Executor confirmed the expected screen was displayed"
                if i < 2
                else None,
            )
        )
    return plan


GOAL = "Open WhatsApp, find the chat with Alice and send her the message 'Hello!'"
THOUGHTS = [f"[executor] Tap on element with resource_id='com.x:id/btn_{i}' was successful." for i in range(12)]
FEEDBACK_MSG = ToolMessage(
    name="tap",
    tool_call_id="call_001",
    content="Tap on element with resource_id='com.x:id/btn_3' was successful.",
)

HIERARCHY = [
    {
        "text": f"Item {i}",
        "content-desc": "",
        "resource-id": f"com.example.app:id/item_{i}",
        "class": "android.widget.TextView",
        "bounds": f"[{i * 10},{i * 20}][{i * 10 + 300},{i * 20 + 90}]",
        "clickable": "true",
        "enabled": "true",
        "visible-to-user": "true",
    }
    for i in range(90)
]

DEVICE_INFO = (
    "Host platform: WINDOWS\nMobile platform: android\nDevice ID: emulator-5554\n"
    "Device width: 1080\nDevice height: 2400\n"
)

TOOLS_LIST = ", ".join(w.tool_fn_getter.__name__ for w in EXECUTOR_WRAPPERS_TOOLS)


def _real_tools():
    """按 index.get_tools_from_wrappers 的方式拿到真实 BaseTool（不碰设备）。"""
    from minitap.mobile_use.context import DeviceContext, DevicePlatform, MobileUseContext
    from minitap.mobile_use.config import LLMConfig
    from minitap.mobile_use.tools.index import get_tools_from_wrappers

    # 工具 schema 生成期完全不看 llm_config（只有真正调用工具时才用 ctx.device），
    # 所以直接 model_construct 跳过校验，避免为了计数去造一整套 LLM 配置。
    ctx = MobileUseContext.model_construct(
        trace_id="probe",
        device=DeviceContext(
            host_platform="WINDOWS",
            mobile_platform=DevicePlatform.ANDROID,
            device_id="probe-device",
            device_width=1080,
            device_height=2400,
        ),
        llm_config=LLMConfig.model_construct(),
        video_recording_enabled=False,
    )
    return get_tools_from_wrappers(ctx, list(EXECUTOR_WRAPPERS_TOOLS))


# --------------------------------------------------------------------------
# 1. Cortex —— 每轮循环都跑一次
# --------------------------------------------------------------------------
def probe_cortex():
    hr("1. CORTEX  (每轮 contextor->cortex 循环跑一次，带截图)")
    tpl = Template((ROOT / "minitap/mobile_use/agents/cortex/cortex.md").read_text(encoding="utf-8"))
    system = tpl.render(
        platform="android",
        initial_goal=GOAL,
        subgoal_plan="\n".join(str(s) for s in make_plan()),
        current_subgoal="Subgoal 3",
        executor_feedback=(
            f"Latest UI decisions:\n{state_decisions}\n\nExecutor feedback:\n{[FEEDBACK_MSG]}"
        ),
        executor_tools_list=TOOLS_LIST,
        locked_app_package=None,
    )
    ui = render_ui_hierarchy(HIERARCHY)

    parts = [
        ("system prompt (cortex.md)", ntok(system)),
        ("device info (HumanMessage)", ntok(DEVICE_INFO)),
        (f"agent thoughts x{len(THOUGHTS)} (每条一个 AIMessage)", ntok("\n".join(THOUGHTS)) + len(THOUGHTS) * 4),
        (f"UI hierarchy 裁剪后 ({len(HIERARCHY)} nodes)", ntok(ui)),
        ("structured output schema (CortexOutput)", ntok(json.dumps(convert_to_openai_tool(CortexOutput)))),
    ]
    for label, v in parts:
        print(f"  {v:>7,}  {label}")
    total = sum(v for _, v in parts)
    print(f"  {'-' * 70}")
    print(f"  {total:>7,}  TOTAL cortex (不含截图)")

    # 截图按 Gemini/Claude 的常见计价折算
    for label, px in [("1080x2400 原图", (1080, 2400)), ("压缩后 540x1200", (540, 1200))]:
        tiles = ((px[0] // 768) + 1) * ((px[1] // 768) + 1) if px[0] > 768 or px[1] > 768 else 1
        print(f"  ~{tiles * 258:>6,}  截图 {label} (Gemini tile 计价估算)")


state_decisions = '[{"action": "tap", "target": {"resource_id": "com.x:id/btn_3", "resource_id_index": 0, "bounds": {"x": 30, "y": 60, "width": 300, "height": 90}}}]'


# --------------------------------------------------------------------------
# 2. Executor —— 每个 tool-call 轮跑一次，且 bind_tools 每次都重发
# --------------------------------------------------------------------------
def probe_executor():
    hr("2. EXECUTOR  (每个 tool-call 轮跑一次，bind_tools 每次重发全部 schema)")
    tpl = Template((ROOT / "minitap/mobile_use/agents/executor/executor.md").read_text(encoding="utf-8"))
    system = tpl.render(platform="android")

    schemas = [convert_to_openai_tool(t) for t in _real_tools()]
    schema_json = json.dumps(schemas)

    # executor_messages 在两次 cortex 之间累积：AI(tool_calls) + Tool 交替
    for k in (1, 4, 8):
        history = []
        for i in range(k):
            history.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "tap",
                            "args": {
                                "agent_thought": "Tapping the target button to proceed",
                                "target": state_decisions and json.loads(state_decisions)[0]["target"],
                            },
                            "id": f"call_{i}",
                            "type": "tool_call",
                        }
                    ],
                )
            )
            history.append(FEEDBACK_MSG)
        hist_txt = json.dumps(
            [
                m.model_dump(exclude={"additional_kwargs"})
                for m in history
            ],
            default=str,
        )
        parts = [
            ("system prompt (executor.md)", ntok(system)),
            ("cortex_last_thought", ntok("Decisions reason: ... Goals completion reason: None")),
            ("structured_decisions", ntok(state_decisions)),
            (f"executor_messages x{len(history)} (累积)", ntok(hist_txt)),
        ]
        for label, v in parts:
            print(f"  [{k} 轮] {v:>7,}  {label}")
        print(f"  [{k} 轮] {ntok(schema_json):>7,}  bind_tools schema (15 个工具)")
        print(f"  [{k} 轮] {'-' * 62}")
        print(f"  [{k} 轮] {sum(v for _, v in parts) + ntok(schema_json):>7,}  TOTAL executor\n")

    print(f"  tool schema 明细：")
    for s in schemas:
        n = ntok(json.dumps(s))
        print(f"      {n:>6,}  {s['function']['name']}")
    print(f"      {ntok(schema_json):>6,}  {'SUM + envelope'}")


# --------------------------------------------------------------------------
# 3. Planner —— 只在 replan 时跑
# --------------------------------------------------------------------------
def probe_planner():
    hr("3. PLANNER  (只在 replan 时跑，但带全量 agent thoughts)")
    tpl = Template((ROOT / "minitap/mobile_use/agents/planner/planner.md").read_text(encoding="utf-8"))
    system = tpl.render(
        platform="android",
        executor_tools_list=TOOLS_LIST,
        locked_app_package=None,
        current_foreground_app="com.whatsapp",
        video_recording_enabled=False,
    )
    human = Template((ROOT / "minitap/mobile_use/agents/planner/human.md").read_text(encoding="utf-8")).render(
        action="replan",
        initial_goal=GOAL,
        previous_plan="\n".join(str(s) for s in make_plan()),
        agent_thoughts="\n".join(THOUGHTS),
    )
    parts = [
        ("system prompt (planner.md)", ntok(system)),
        ("human.md (goal + plan + 全量 thoughts)", ntok(human)),
        ("structured output schema (PlannerOutput)", ntok(json.dumps(convert_to_openai_tool(PlannerOutput)))),
    ]
    for label, v in parts:
        print(f"  {v:>7,}  {label}")
    print(f"  {'-' * 70}")
    print(f"  {sum(v for _, v in parts):>7,}  TOTAL planner")


if __name__ == "__main__":
    probe_cortex()
    probe_executor()
    probe_planner()
    print()
