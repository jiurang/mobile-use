"""汇总真机 A/B 结果。

⚠️ **对比的诚实性**：before / after 必须是**同任务、同设备、同模型**跑的。
本脚本会检查两边的 goal 是否一致，不一致就明确标出来（而不是假装可比）。

数据来源：
    probe_out/executor_ab_before.json   ← 改造前（scripts/probe_baseline_before.py）
    probe_out/executor_e2e.json         ← 改造后（scripts/probe_executor_e2e.py）
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load(p: pathlib.Path):
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    before = load(ROOT / "probe_out/executor_ab_before.json")
    after = load(ROOT / "probe_out/executor_e2e.json")

    if not before or not after:
        print("缺 before 或 after 的数据，跑对应脚本后再来")
        return 1

    same_goal = before["goal"] == after["goal"]
    print("=" * 74)
    print(f"真机 A/B   目标一致: {'✓' if same_goal else '✗'}")
    print(f"  before: {before['goal']}")
    print(f"  after : {after['goal']}")
    print("=" * 74)

    def pct(a: float, b: float) -> str:
        if not a:
            return "n/a"
        return f"{(b - a) / a * 100:+.1f}%"

    print()
    print(f"{'指标':<26}{'改造前':>12}{'改造后':>12}{'变化':>12}")
    print("-" * 74)
    rows = [
        ("任务结果", "success" if not before["error"] else "failed",
         "success" if not after["error"] else "failed", None),
        ("总耗时 (s)", before["wall_s"], after["wall_s"], None),
        ("LLM 调用次数", before["llm_calls"], after["llm_calls"], True),
        ("input token", before["input"], after["input"], True),
        ("cached token", before["cached"], after["cached"], True),
        ("output token", before["output"], after["output"], True),
        ("缓存命中率 (%)",
         round(before["cached"] / before["input"] * 100, 1) if before["input"] else 0,
         round(after["cached"] / after["input"] * 100, 1) if after["input"] else 0, None),
    ]
    for label, b, a, show_pct in rows:
        if show_pct and isinstance(b, (int, float)) and isinstance(a, (int, float)):
            chg = pct(b, a)
        elif isinstance(b, (int, float)) and isinstance(a, (int, float)):
            chg = f"{a - b:+.1f}" if a != b else "="
        else:
            chg = ""
        print(f"{label:<26}{str(b):>12}{str(a):>12}{chg:>12}")

    print()
    print("=" * 74)
    print("单次决策的推理量（方案 §2.4 的核心指标）")
    print("=" * 74)
    b_cortex = before["by_agent"].get("cortex", {})
    a_exec = after["by_agent"].get("executor", {})
    if b_cortex.get("n") and a_exec.get("n"):
        b_per = b_cortex["out"] / b_cortex["n"]
        a_per = a_exec["out"] / a_exec["n"]
        print(f"  改造前 cortex  output/次 = {b_per:>8.0f} tok")
        print(f"  改造后 executor output/次 = {a_per:>8.0f} tok")
        print(f"  → {pct(b_per, a_per)}  （方案预期 2,725 → ~150，18x）")
    b_in = before["by_agent"].get("cortex", {})
    a_in = after["by_agent"].get("executor", {})
    if b_in.get("n") and a_in.get("n"):
        print()
        print(f"  改造前 cortex  input/次  = {b_in['in'] / b_in['n']:>8.0f} tok")
        print(f"  改造后 executor input/次  = {a_in['in'] / a_in['n']:>8.0f} tok")
        print(f"  → {pct(b_in['in'] / b_in['n'], a_in['in'] / a_in['n'])}")

    print()
    print("=" * 74)
    print("逐 agent 明细")
    print("=" * 74)
    for tag, data in (("改造前", before), ("改造后", after)):
        print(f"\n  【{tag}】")
        print(f"    {'agent':<14}{'次数':>5}{'input':>9}{'cached':>9}{'output':>9}")
        for k, v in sorted(data["by_agent"].items(), key=lambda x: -x[1]["in"]):
            cached = v.get("ca", v.get("cached", 0))
            print(f"    {k:<14}{v['n']:>5}{v['in']:>9}{cached:>9}{v['out']:>9}")

    print()
    print("=" * 74)
    print("⚠️ 读这张表之前必须知道的三件事")
    print("=" * 74)
    print("  1. **调用次数必然增加** —— Executor 每轮要自己决策，就得多调一次。")
    print("     这是方案的设计（§2.3），不是缺陷。收益在 output 和 prompt 体积。")
    print("  2. **两次跑的步数可能不同** —— 任务难度不完全可比，")
    print("     所以「总耗时」的差值有噪声，单次运行不能当结论。")
    print("  3. **样本量 = 1** —— 成功率、缓存率都得多跑几次才有统计意义。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
