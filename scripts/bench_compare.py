"""对比两次 bench_run 的结果，输出优化前后的真实收益。

A/B 的坑：**agent 是随机的**，两次跑的迭代数可能不一样，所以不能直接比总耗时。
这里比的是「每轮」指标，以及 LLM 调用构成 —— 那才是优化的作用域。

用法：
    uv run python scripts/bench_compare.py baseline optimized
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(tag: str) -> dict:
    path = ROOT / "probe_out" / f"bench_{tag}.json"
    if not path.exists():
        raise SystemExit(f"缺少 {path}，先跑：uv run python scripts/bench_run.py --goal ... --tag {tag}")
    return json.loads(path.read_text(encoding="utf-8"))


def bar(frac: float, width: int = 28) -> str:
    return "#" * max(0, round(frac * width))


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__ or "用法: bench_compare.py <baseline> <optimized>")
    base, opt = load(sys.argv[1]), load(sys.argv[2])

    print("=" * 74)
    print(f"目标: {opt['goal']}")
    for tag, r in (("BASELINE", base), ("OPTIMIZED", opt)):
        status = "ERROR" if r.get("error") else "ok"
        print(f"  {tag:<10} {r['total_ms'] / 1000:6.1f}s  cortex轮次={r['nodes'].get('cortex', {}).get('calls', 0)}  {status}")
        if r.get("error"):
            print(f"             error: {r['error']}")
    print("=" * 74)

    bc, oc = base["nodes"].get("cortex"), opt["nodes"].get("cortex")
    if not bc or not oc:
        print("两组至少有一方没有 cortex 数据，无法对比")
        return 1

    print("\n【1】LLM 调用构成 —— 这是优化的作用域")
    print(f"  {'':<14}{'BASELINE':>12}{'OPTIMIZED':>12}{'变化':>14}")
    b_iter = base["llm_calls_per_iteration"] or 0
    o_iter = opt["llm_calls_per_iteration"] or 0
    delta = f"{(o_iter - b_iter) / b_iter * 100:+.0f}%" if b_iter else "-"
    print(f"  {'每轮 LLM 调用':<14}{b_iter:>12}{o_iter:>12}{delta:>14}")
    print(f"  {'LLM 调用总数':<14}{base['llm_calls_total']:>12}{opt['llm_calls_total']:>12}")
    hit, miss = opt["dispatch"]["hit"], opt["dispatch"]["miss"]
    if hit + miss:
        print(f"  {'直连命中':<14}{'n/a':>12}{f'{hit}/{hit + miss}':>12}{f'({hit / (hit + miss) * 100:.0f}%)':>14}")

    print("\n【2】cortex 节点（每轮均值）—— 你最初怀疑的热点")
    b_mean, o_mean = bc["mean_ms"], oc["mean_ms"]
    print(f"  均值 ms         {b_mean:>12}{o_mean:>12}{(o_mean - b_mean) / b_mean * 100:>13.0f}%")
    print(f"  中位 ms         {bc['median_ms']:>12}{oc['median_ms']:>12}")
    print(f"  轮次数          {bc['calls']:>12}{oc['calls']:>12}")

    print("\n【3】各节点单次均值 (ms)")
    print(f"  {'节点':<14}{'BASELINE':>12}{'OPTIMIZED':>12}{'变化':>12}")
    for label in sorted(set(base["nodes"]) | set(opt["nodes"])):
        b = base["nodes"].get(label, {}).get("mean_ms")
        o = opt["nodes"].get(label, {}).get("mean_ms")
        if b is None or o is None:
            print(f"  {label:<14}{str(b):>12}{str(o):>12}{'—':>12}")
            continue
        chg = f"{(o - b) / b * 100:+.0f}%" if b else "-"
        print(f"  {label:<14}{b:>12}{o:>12}{chg:>12}")

    # 每轮总成本：把一轮里所有节点的时间加起来
    b_round = sum(v["mean_ms"] for v in base["nodes"].values())
    o_round = sum(v["mean_ms"] for v in opt["nodes"].values())
    print("\n【4】一轮的合计（各节点均值之和）")
    print(f"  BASELINE  {b_round:8.0f} ms  {bar(b_round / max(b_round, o_round))}")
    print(f"  OPTIMIZED {o_round:8.0f} ms  {bar(o_round / max(b_round, o_round))}")
    if b_round:
        print(f"  变化: {(o_round - b_round) / b_round * 100:+.1f}%  "
              f"(省 {b_round - o_round:.0f} ms/轮)")

    print("\n" + "=" * 74)
    print("注意：agent 有随机性，轮次数不同会放大/缩小整体差距。")
    print("      '每轮 LLM 调用' 和 '单次均值' 才是干净指标，总耗时仅供参考。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
