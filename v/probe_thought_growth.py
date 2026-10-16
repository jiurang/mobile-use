"""验证 agents_thoughts 是否随任务轮次无界增长 —— 也就是 cortex 每轮都要重发的那个块。

读 graph/state.py::_add_agent_thoughts + agents/cortex/cortex.py 的消息拼装，
按真实调用语义模拟一次多 subgoal 任务，数出第 N 轮 cortex 的上下文。

用法：python v/probe_thought_growth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

import tiktoken

ROOT = Path(__file__).resolve().parent.parent
ENC = tiktoken.get_encoding("cl100k_base")


def ntok(t: str) -> int:
    return len(ENC.encode(t))


# cortex.md 里 executor_tools_list / plan / goal 这些是常量，只算随轮次增长的部分。
# agents_thoughts 每轮新增：executor 的 agent_thought + 每个 tool 的 outcome。
AGENT_THOUGHT = (
    "Tapping the search box at coordinates (540, 320) to focus it before typing the query."
)
TOOL_OUTCOME = "Tap on element with coordinates (540.0, 320.0) was successful."

BASE_CORTEX = 1_747  # system prompt + device info + schema，上一轮实测
UI_TREE = 2_527  # 90 节点裁剪后
SCREENSHOT = 2_064  # 1080x2400 原图

print("=" * 78)
print("agents_thoughts 累积 → cortex 上下文增长曲线")
print("=" * 78)
print(f"固定开销: system {BASE_CORTEX:,} + UI {UI_TREE:,} + 截图 {SCREENSHOT:,}")
print()

# 每个 subgoal 平均 3 轮 cortex 循环，每轮 1~2 个 tool call
for n_subgoals in (1, 5, 10, 20, 30):
    turns = n_subgoals * 3
    # 每轮：cortex 产生 1 条 thought，executor 产生 1 条 agent_thought + N 条 outcome
    thoughts: list[str] = []
    per_turn_total = 0
    for t in range(turns):
        n_calls = 2
        round_added = [
            f"[cortex] Decisions reason: ... Goals completion reason: None",
            f"[executor] {AGENT_THOUGHT}",
        ] + [f"[executor] {TOOL_OUTCOME}" for _ in range(n_calls)]
        thoughts.extend(round_added)
        per_turn_total += ntok("\n".join(round_added))

    # 最后一轮的 cortex 要重发全部历史 thoughts（cortex.py:77-78 逐条 append AIMessage）
    history = "\n".join(thoughts)
    # 每条 AIMessage 的 role/分隔开销，按 4 token 估
    overhead = len(thoughts) * 4
    cortex_total = BASE_CORTEX + UI_TREE + SCREENSHOT + ntok(history) + overhead

    print(f"  subgoal {n_subgoals:>2} / 第 {turns:>3} 轮")
    print(f"     thoughts 条目数        {len(thoughts):>6,}")
    print(f"     全量 thoughts 重发     {ntok(history) + overhead:>7,} tokens")
    print(f"     cortex 本轮总上下文   {cortex_total:>7,} tokens")
    print(f"     其中 thoughts 占比     {(ntok(history) + overhead) / cortex_total * 100:>5.1f}%")
    print()
