"""补测 orchestrator / outputter 两个节点，并算出一轮完整循环的上下文总量。

用法：python v/probe_minor_nodes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

import tiktoken
from jinja2 import Template
from langchain_core.utils.function_calling import convert_to_openai_tool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))

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


def ntok(t: str) -> int:
    return len(ENC.encode(t))


def make_plan(n=5):
    return [
        Subgoal(
            id=generate_id(),
            description=f"Subgoal {i + 1}: open the app and navigate to the settings page",
            status=SubgoalStatus.SUCCESS if i < 2 else SubgoalStatus.PENDING,
            completion_reason="Executor confirmed the expected screen was displayed" if i < 2 else None,
        )
        for i in range(n)
    ]


GOAL = "Open WhatsApp, find the chat with Alice and send her the message 'Hello!'"
TOOLS_LIST = ", ".join(w.tool_fn_getter.__name__ for w in EXECUTOR_WRAPPERS_TOOLS)
N_THOUGHTS = 60
THOUGHTS = [
    f"[executor] Tap on element with resource_id='com.x:id/btn_{i}' was successful."
    for i in range(N_THOUGHTS)
]
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

print("=" * 78)
print("ORCHESTRATOR  (每次 review_subgoals 跑一次，带全量 agent thoughts)")
print("=" * 78)
orch_system = Template(
    (ROOT / "minitap/mobile_use/agents/orchestrator/orchestrator.md").read_text(encoding="utf-8")
).render(platform="android")
orch_human = Template(
    (ROOT / "minitap/mobile_use/agents/orchestrator/human.md").read_text(encoding="utf-8")
).render(
    initial_goal=GOAL,
    subgoal_plan="\n".join(str(s) for s in make_plan()),
    subgoals_to_examine="\n".join(str(s) for s in make_plan()[2:]),
    agent_thoughts="\n".join(THOUGHTS),
)
for label, v in [
    ("system prompt (orchestrator.md)", ntok(orch_system)),
    (f"human.md (goal + plan + 全量 thoughts x{N_THOUGHTS})", ntok(orch_human)),
]:
    print(f"  {v:>7,}  {label}")
print(f"  {ntok(orch_human) + ntok(orch_system):>7,}  TOTAL orchestrator\n")

print("=" * 78)
print("OUTPUTTER  (任务结束跑一次，带全量 agent thoughts)")
print("=" * 78)
out_system = (
    "You are a helpful assistant tasked with generating "
    "the final structured output of a multi-agent reasoning process."
)
out_human = Template(
    (ROOT / "minitap/mobile_use/agents/outputter/human.md").read_text(encoding="utf-8")
).render(
    initial_goal=GOAL,
    # 注意：outputter.py:32 传的是 list，模板用 {% for %} 遍历。
    # 这里必须传 list —— 传 join 后的字符串会退化成「按字符迭代」。
    agents_thoughts=THOUGHTS,
    structured_output=None,
    output_description=None,
    last_ai_message=None,
)
for label, v in [
    ("system prompt (硬编码)", ntok(out_system)),
    (f"human.md (goal + 全量 thoughts x{N_THOUGHTS})", ntok(out_human)),
]:
    print(f"  {v:>7,}  {label}")
print(f"  {ntok(out_human) + ntok(out_system):>7,}  TOTAL outputter\n")

print("=" * 78)
print("一轮完整循环的上下文总量（以 5 subgoal 任务的中段为例）")
print("=" * 78)
DECISIONS = '[{"action": "tap", "target": {"resource_id": "com.x:id/btn_3", "resource_id_index": 0, "bounds": {"x": 30, "y": 60, "width": 300, "height": 90}}}]'

rows = [
    ("cortex (system + device + thoughts + UI + 截图)", 4_802 + 2_064),
    ("executor x2 轮 (含 bind_tools 3,560 x2)", (336 + 59 + 199 + 3_560) * 2),
    ("orchestrator (cortex 标记完成时)", ntok(orch_system) + ntok(orch_human)),
]
for label, v in rows:
    print(f"  {v:>7,}  {label}")
total = sum(v for _, v in rows)
print(f"  {'-' * 70}")
print(f"  {total:>7,}  单轮循环合计（不含 replan / outputter）")
