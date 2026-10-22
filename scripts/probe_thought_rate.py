"""量改造后 agents_thoughts 的增长速率 —— 决定 M13（steps_count 口径）和 L1（子目标粒度）。

对比：
* 改造前：cortex 每轮写 1~2 条（decisions_reason 等）
* 改造后：executor 每个动作步写 1 条（工具结果）+ cortex 每次写 1 条 + orchestrator 写

数据源：`probe_out/dump_run.log`（UTF-16LE）。
"""

from __future__ import annotations

import pathlib
import re
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG = ROOT / "probe_out" / "dump_run.log"


def main() -> int:
    if not LOG.exists():
        print(f"缺 {LOG}")
        return 1
    raw = LOG.read_bytes()
    text = None
    for enc in ("utf-16", "utf-8", "gb18030"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)

    print("=" * 72)
    print("1. 各类 agent 的执行次数")
    print("=" * 72)
    counts = {
        "planner": len(re.findall(r"Starting Planner Agent", text)),
        "orchestrator": len(re.findall(r"Starting Orchestrator Agent", text)),
        "contextor": len(re.findall(r"Starting Contextor Agent", text)),
        "cortex": len(re.findall(r"Starting Cortex Agent", text)),
        "executor": len(re.findall(r"Starting Executor Agent", text)),
        "reporter": len(re.findall(r"Starting Reporter Agent", text)),
        "summarizer": len(re.findall(r"Starting Summarizer", text)),
    }
    for k, v in counts.items():
        print(f"  {k:<14} {v}")

    print()
    print("=" * 72)
    print("2. 工具调用次数（= 动作步数）")
    print("=" * 72)
    tool_ok = len(re.findall(r"Tool call succeeded", text))
    tool_fail = len(re.findall(r"Tool call failed", text))
    print(f"  成功 {tool_ok}  失败 {tool_fail}  合计 {tool_ok + tool_fail}")

    print()
    print("=" * 72)
    print("3. agents_thoughts 的写入方（看 thought 前缀）")
    print("=" * 72)
    prefixes = Counter(
        re.findall(r"\[(cortex|executor|orchestrator|planner|contextor|reporter)\]", text)
    )
    for k, v in prefixes.most_common():
        print(f"  [{k}] {v}")

    print()
    print("=" * 72)
    print("4. 关键推算")
    print("=" * 72)
    steps = tool_ok + tool_fail
    if steps:
        print(f"  动作步数 = {steps}")
        per_decision = steps / max(counts["executor"], 1)
        print(f"  executor 决策次数 = {counts['executor']}  → 约 {per_decision:.1f} 步/次决策")
        print(f"  每动作步产生的 thought 条数 ≈ {sum(prefixes.values()) / steps:.1f}")
        print()
        print("  ⇒ M13：steps_count = len(agents_thoughts) 会**高估**动作步数约 "
              f"{sum(prefixes.values()) / steps:.1f} 倍")
        print("  ⇒ L1：每子目标步数 =", f"{steps / max(counts['reporter'], 1):.1f}"
              f"（reporter 次数 = 子目标数）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
