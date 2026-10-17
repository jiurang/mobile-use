"""把多个探针结果并排成一张表（分位数 + 长尾计数 + prompt 对账）。

为什么要有这个
--------------
探针各自只打印一行汇总，看不出**分布**。而这个项目的模型差异几乎全在长尾上
（中位只差 2 倍多，最坏差 10 倍），只看中位会低估问题。

严格性上注意三点：
  1. 两组必须同 prompt、同时间窗跑，否则时段网络波动会混进来。
     脚本会打印各组的 meta（system 字数 / 树字数 / 行数 / 截图 KB）对账。
  2. meta 不一致时脚本会**明确警告不可直接比**，而不是照样给你一张漂亮的表。
  3. 成功率只看这一批，不跨批次合并 —— 不同批次是不同样本，
     合并会把「偶发」和「固有」混为一谈。

跑法：
    # 默认对比 M3 vs M3.1-Flash
    uv run python scripts/rollup_model_ab.py

    # 任意两组（label=文件名去掉前缀）
    uv run python scripts/rollup_model_ab.py \
        "cortex_real_prompt_MiniMax-M3_1-Flash-Preview_effort_max.json" \
        "cortex_real_prompt_MiniMax-M3_1-Flash-Preview.json"
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "probe_out"

DEFAULT = [
    "cortex_real_prompt_MiniMax-M3.json",
    "cortex_real_prompt_MiniMax-M3_1-Flash-Preview.json",
]


def label_of(name: str) -> str:
    stem = name.removesuffix(".json").removeprefix("cortex_real_prompt_")
    return stem.replace("MiniMax-", "").replace("MiniMax", "").replace("_", " ") or name


def pct(xs: list[float], p: float) -> float:
    """线性插值分位数（跟 numpy 默认一致，样本少时比 index 法稳）。"""
    s = sorted(xs)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return round(s[f] + (s[c] - s[f]) * (k - f))


def main() -> int:
    names = sys.argv[1:] or DEFAULT
    if len(names) < 2:
        print("至少要两组")
        return 1

    data = {}
    for name in names:
        path = OUT / name
        if not path.exists():
            print(f"缺少 {path}")
            return 1
        data[label_of(name)] = json.loads(path.read_text(encoding="utf-8"))

    labels = list(data)
    stats: dict[str, dict] = {}
    for label, d in data.items():
        ms = [r["ms"] for r in d["results"]]
        stats[label] = {
            "n": d["n"],
            "ok": sum(1 for r in d["results"] if r["ok"]),
            "min": min(ms),
            "median": statistics.median(ms),
            "p75": pct(ms, 0.75),
            "p90": pct(ms, 0.90),
            "max": max(ms),
            "over10s": sum(1 for x in ms if x > 10000),
            "over16s": sum(1 for x in ms if x > 16000),
            "sum": sum(ms),
            "ms": ms,
        }

    cols = labels
    base = stats[cols[0]]
    w = max(12, max(len(c) for c in cols) + 1)

    print("=" * (w * len(cols) + 26))
    print(f"{'指标':<14}" + "".join(f"{c:>{w}}" for c in cols) + f"{'倍数(最左/其余)':>22}")
    print("-" * (w * len(cols) + 36))

    def row(stat_key: str, display: str, ratio=None) -> None:
        cells = "".join(f"{stats[c][stat_key]:>{w}.0f}" for c in cols)
        r = ""
        if ratio:
            vals = [stats[c][ratio] for c in cols]
            if all(vals):
                r = "  " + " / ".join(f"{vals[0] / v:.2f}x" for v in vals[1:])
        print(f"{display:<14}{cells}{r:>22}")

    print(f"{'样本数':<14}" + "".join(f"{stats[c]['n']:>{w}}" for c in cols))
    print(f"{'成功':<14}" + "".join(f"{f'{stats[c]['ok']}/{stats[c]['n']}':>{w}}" for c in cols))
    for key, name in (("min", "最小 ms"), ("median", "中位 ms"), ("p75", "p75 ms"),
                      ("p90", "p90 ms"), ("max", "最大 ms")):
        row(key, name, ratio=key)
    print(f"{'>10s 次数':<14}" + "".join(f"{f'{stats[c]['over10s']}/{stats[c]['n']}':>{w}}" for c in cols))
    print(f"{'>16s 次数':<14}" + "".join(f"{f'{stats[c]['over16s']}/{stats[c]['n']}':>{w}}" for c in cols))
    print(f"{'合计 s':<14}" + "".join(f"{stats[c]['sum'] / 1000:>{w}.1f}" for c in cols)
          + f"{base['sum'] / stats[cols[-1]]['sum']:>19.2f}x")

    print("\n逐次耗时 ms（看分布形状，别只看中位）")
    for c in cols:
        print(f"  {c:<28}{' '.join(f'{x:>6}' for x in stats[c]['ms'])}")

    print("\nprompt 对账（各组必须是同一批输入才可直接比）")
    for c in cols:
        m = data[c]["meta"]
        print(f"  {c:<28}{m}")
    metas = {json.dumps(data[c]["meta"], sort_keys=True) for c in cols}
    print()
    if len(metas) == 1:
        print("  ✅ 全部一致，可以直接比")
    else:
        print("  ⚠️  meta 不一致 —— **这两组不能直接比**，差异可能来自 prompt 而非模型")
    print("=" * (w * len(cols) + 26))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
