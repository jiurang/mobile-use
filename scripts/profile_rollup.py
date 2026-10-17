"""把 cortex_profile_*.json 换算到「全程」口径，回答优化该往哪使劲。

单看 cortex 的内部百分比会误导：cortex 只占全程的一部分，而它自己的 99%
又是 LLM。把三层口径摊开才看得清钱花在哪。

用法：
    uv run python scripts/profile_rollup.py ocr_on ocr_off
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(tag: str) -> dict:
    path = ROOT / "probe_out" / f"cortex_profile_{tag}.json"
    if not path.exists():
        raise SystemExit(f"缺少 {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    tags = sys.argv[1:] or ["ocr_on"]
    rows = [load(t) for t in tags]

    print("=" * 86)
    print(f"{'组':<12}{'全程(s)':>9}{'cortex轮':>9}{'LLM(s)':>10}{'contextor':>11}{'executor':>10}{'其它':>9}{'LLM占比':>10}")
    print("-" * 86)
    for d in rows:
        llm = d["buckets"].get("llm_main", {}).get("total_s", 0.0)
        fb = d["buckets"].get("llm_fallback", {}).get("total_s", 0.0)
        ctx = d["contextor_total"].get("total_s", 0.0)
        ex = d["executor_total"].get("total_s", 0.0)
        wall = d["wall_s"]
        other = wall - llm - fb - ctx - ex
        print(
            f"{d['tag']:<12}{wall:>9.1f}{d['cortex_total'].get('n', 0):>9}"
            f"{llm + fb:>10.1f}{ctx:>11.1f}{ex:>10.1f}{other:>9.1f}"
            f"{(llm + fb) / wall * 100:>9.0f}%"
        )
        if d.get("error"):
            print(f"{'':<12}⚠ {d['error'][:60]}")

    print("\n" + "=" * 86)
    print("cortex 内部构成（占 cortex 总耗时）")
    print(f"{'组':<12}{'LLM主':>10}{'LLM兜底':>10}{'截图压缩':>11}{'树/OCR渲染':>12}{'其它':>8}")
    print("-" * 86)
    for d in rows:
        ct = d["cortex_total"].get("total_s", 0.0) or 1.0
        def pct(k):
            return d["buckets"].get(k, {}).get("total_s", 0.0) / ct * 100
        print(
            f"{d['tag']:<12}{pct('llm_main'):>9.1f}%{pct('llm_fallback'):>9.1f}%"
            f"{pct('screenshot'):>10.1f}%{pct('render'):>11.1f}%"
            f"{d['other_ms_total'] / 1000 / ct * 100:>7.1f}%"
        )

    # OCR 开关的直接对照
    on = next((d for d in rows if d["ocr_enabled"]), None)
    off = next((d for d in rows if not d["ocr_enabled"]), None)
    if on and off:
        print("\n" + "=" * 86)
        print("OCR 开关对照")
        print("-" * 86)
        oc = on["buckets"].get("llm_main", {}).get("total_s", 0.0)
        oc_rounds = on["buckets"].get("llm_main", {}).get("n", 0)
        fcm = off["buckets"].get("llm_main", {}).get("total_s", 0.0)
        fcr = off["buckets"].get("llm_main", {}).get("n", 0)
        print(f"  contextor  均值  开 {on['contextor_total'].get('mean_ms', 0)}ms  "
              f"关 {off['contextor_total'].get('mean_ms', 0)}ms")
        if oc_rounds and fcr:
            per = (oc / oc_rounds - fcm / fcr) * 1000
            print(f"  cortex LLM 单次均值  开 {oc / oc_rounds * 1000:.0f}ms  "
                  f"关 {fcm / fcr * 1000:.0f}ms   差 {per:+.0f}ms/轮")
        print(f"  任务完成  开 {'成功' if not on.get('error') else '失败'}  "
              f"关 {'成功' if not off.get('error') else '失败'}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
