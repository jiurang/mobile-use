"""实测 v3：system prompt 换中文是否省 token，以及最终推荐组合。

用法：python v/probe_prompt_zh.py
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


# executor.md 的规则段落，原文 vs 中文译法
PAIRS = [
    (
        "Don't reason about strategy - just execute what Cortex decided",
        "不要推理策略——只执行 Cortex 的决定",
    ),
    (
        "`agent_thought` must be specific - not generic/vague",
        "`agent_thought` 必须具体，不要笼统含糊",
    ),
    (
        "Order matters - tools execute in the order you return them",
        "顺序很重要——工具按你返回的顺序执行",
    ),
    (
        "Interpret Cortex decisions and execute tools on android mobile device. "
        "You are the hands, Cortex is the brain.",
        "解释 Cortex 的决策并在 android 设备上执行工具。你是手，Cortex 是脑。",
    ),
    (
        "Parses decisions from Cortex, calls tools in the specified order, "
        "always includes an agent_thought for each tool",
        "解析 Cortex 决策，按指定顺序调用工具，每次工具调用都附带 agent_thought",
    ),
    (
        "Clear text: Clears entire field. If fails, try: long press → select all → erase_one_char",
        "清空文本：清空整个输入框。失败时改用：长按 → 全选 → erase_one_char",
    ),
]

# cortex.md 里的 CRITICAL RULES 段落
CORTEX_RULES = [
    (
        "Before ANY decision, review agent thoughts history to detect repeated failures, "
        "spot contradictions between plan and reality, and learn from what worked or failed.",
        "做任何决策前，先回顾历史思路：发现重复失败、找出计划与现实的矛盾、总结成败经验。",
    ),
    (
        "If something failed, understand WHY before trying again. Ask: "
        "How would a human solve this differently?",
        "失败后先弄清原因再重试。问自己：人类会怎么用别的方法解决？",
    ),
    (
        "These actions change the screen unpredictably: back, launch_app, stop_app, "
        "open_link, navigation taps. If your decision includes one of these, it MUST be "
        "the ONLY action in that turn.",
        "以下动作会不可预测地改变屏幕：back、launch_app、stop_app、open_link、导航点击。"
        "若决策包含其一，本轮必须只做这一个动作。",
    ),
    (
        "Never mark a goal complete in advance. Only complete based on executor feedback "
        "confirming success.",
        "绝不提前标记目标完成。只有执行器反馈确认成功时才可标记。",
    ),
]

print("=" * 78)
print("中文 vs 英文 token 密度（同义句逐条对照）")
print("=" * 78)
tot_en = tot_zh = 0
for en, zh in PAIRS + CORTEX_RULES:
    e, z = ntok(en), ntok(zh)
    tot_en += e
    tot_zh += z
    ratio = z / e if e else 0
    verdict = "中文更贵" if z > e else "中文更省"
    print(f"  {verdict}")
    print(f"    EN {e:>3}  {en[:66]}")
    print(f"    ZH {z:>3}  {zh[:66]}")
    print()
print(f"  合计  EN {tot_en}  →  ZH {tot_zh}   ({tot_zh - tot_en:+d}, {(tot_zh / tot_en - 1) * 100:+.1f}%)")
print()

print("=" * 78)
print("按段落体量放大到 system prompt")
print("=" * 78)
ratio = tot_zh / tot_en
for name, cur in [("cortex.md", 1747), ("executor.md", 336), ("planner.md", 748)]:
    print(f"  {name:<14} 现 {cur:>5,}  →  全量中文化约 {int(cur * ratio):>5,}  ({int(cur * ratio) - cur:+,})")
