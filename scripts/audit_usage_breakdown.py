"""复算真实归档的 token 分布（方案 §1.1 / §1.3 的数字出处）。

数据源：`probe_out/device_usage.jsonl`（52 次真实调用）。

跑法：
    uv run python scripts/audit_usage_breakdown.py
"""

from __future__ import annotations

import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
USAGE = ROOT / "probe_out" / "device_usage.jsonl"


def main() -> int:
    rows = [
        json.loads(line)
        for line in USAGE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"total calls: {len(rows)}")

    agg: dict[str, dict] = collections.defaultdict(
        lambda: {"n": 0, "inp": 0, "cached": 0, "out": 0, "dur": 0.0}
    )
    for r in rows:
        a = agg[r["agent"]]
        a["n"] += 1
        a["inp"] += r["input"]
        a["cached"] += r["cached"]
        a["out"] += r["output"]
        a["dur"] += r["dur"]

    tot_in = sum(a["inp"] for a in agg.values())
    tot_c = sum(a["cached"] for a in agg.values())
    tot_o = sum(a["out"] for a in agg.values())
    tot_n = sum(a["n"] for a in agg.values())

    for k, a in sorted(agg.items()):
        print(
            f'{k:12s} n={a["n"]:3d} input={a["inp"]:7d} cached={a["cached"]:7d} '
            f'output={a["out"]:7d} dur={a["dur"]:8.1f}s | '
            f'share n={a["n"] / tot_n:.1%} in={a["inp"] / tot_in:.1%} '
            f'out={a["out"] / tot_o:.1%}'
        )

    c = agg["cortex"]
    print()
    print(f'cortex mean input  = {c["inp"] / c["n"]:.0f}   (方案说 8,454)')
    print(f'cortex mean output = {c["out"] / c["n"]:.0f}  (方案说 2,725)')
    print(
        f'cortex cache-hit (cached/input) = {c["cached"] / c["inp"]:.1%}'
        "   (审计文档说 cortex 76.4%)"
    )
    print(f'overall cache-hit = {tot_c / tot_in:.1%}   (审计文档说整体 74.6%)')
    print(
        f'cortex 占比：calls {c["n"] / tot_n:.1%}, input {c["inp"] / tot_in:.1%}, '
        f'output {c["out"] / tot_o:.1%}  (方案说 90%/96%/99%)'
    )
    print(f'cortex dur total = {c["dur"]:.1f}s')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
