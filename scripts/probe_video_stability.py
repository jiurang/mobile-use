"""视频页稳定性探针：在**视频页**上连抓 N 次树，算不同判据的命中率。

为什么需要它
------------
现行 ``tree_fingerprint`` 哈希的是 ``resource-id + text + class + bounds`` **全量元素**。
视频页每秒都在变（点赞数、倒计时、进度条、推荐流内容）⇒ 指纹**永远不相等**
⇒ ``_confirm_tree_stable`` 每次都跑满 3 次 dump ⇒ 感知每步 13.3s。

而 ``tools/live_target.py`` 里已经写下了正确答案：

    OCR 的**内容会变**（点赞数在跳、时钟在走）
    **位置比内容稳**

⇒ 稳定性判据应该比「位置 + 身份」，不该比「内容」。

本探针
------
在当前屏（或指定 app）上，间隔 D 毫秒连抓 N 次**原始 XML** 存盘，
然后离线用不同判据算命中率：

* ``old``      —— 现行：id + text + class + bounds 全量集合
* ``struct``   —— 新：只取可交互元素的 (id, class, bounds)，**不带 text**
* ``struct_txt``—— 新 + 保留「两次 text 相同」的那部分 text（更严）

用法::

    python scripts/probe_video_stability.py --n 6 --gap-ms 500
    python scripts/probe_video_stability.py --app com.kuaishou.nebula --n 8 --gap-ms 500

产物：``probe_out/video_stability/`` 下的 XML + ``report.json``。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "probe_out" / "video_stability"

#: 算「结构摘要」时只认这些交互属性。
#: 视频画面、装饰、进度条都不在其中 ⇒ 内容在变也不影响判定。
INTERACTIVE_FLAGS = ("clickable", "long-clickable", "editable", "scrollable", "focusable")


def _flag(el: dict, name: str) -> bool:
    v = el.get(name)
    return v is True or str(v).lower() in ("true", "1")


def _is_interactive(el: dict) -> bool:
    return any(_flag(el, f) for f in INTERACTIVE_FLAGS)


def _rid(el: dict) -> str:
    for k in ("resource-id", "resourceId"):
        v = el.get(k)
        if isinstance(v, str) and v:
            return v
    return ""


def _box(el: dict) -> str:
    b = el.get("bounds")
    if isinstance(b, dict):
        return "%s,%s,%s,%s" % (b.get("x"), b.get("y"), b.get("width"), b.get("height"))
    return str(b or "")


def digest_old(elements):
    """现行判据：id + text + class + bounds 全量（排除 scrollbar）。"""
    parts = set()
    for e in elements:
        rid = _rid(e)
        if "scrollbar" in rid.lower():
            continue
        parts.add((rid, str(e.get("text") or ""), str(e.get("class") or ""), _box(e)))
    return frozenset(parts)


def digest_struct(elements):
    """新判据：只取可交互元素的 (id, class, bounds)。**不含 text。**"""
    parts = set()
    for e in elements:
        if not _is_interactive(e):
            continue
        rid = _rid(e)
        if "scrollbar" in rid.lower():
            continue
        parts.add((rid, str(e.get("class") or ""), _box(e)))
    return frozenset(parts)


def digest_struct_txt(elements, prev=None):
    """新判据 + 保留 text：text 只在「上一份里同位置的 text 相同」时计入。"""
    prev_map = {}
    if prev is not None:
        for e in prev:
            prev_map.setdefault((_rid(e), _box(e)), str(e.get("text") or ""))
    parts = set()
    for e in elements:
        if not _is_interactive(e):
            continue
        rid = _rid(e)
        if "scrollbar" in rid.lower():
            continue
        t = str(e.get("text") or "")
        key = (rid, _box(e))
        same = prev_map.get(key) == t
        parts.add((rid, str(e.get("class") or ""), _box(e), t if same else ""))
    return frozenset(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="连抓几次")
    ap.add_argument("--gap-ms", type=int, default=500, help="两次之间的间隔")
    ap.add_argument("--app", help="先拉起这个 app（包名）")
    ap.add_argument("--serial", default=None)
    args = ap.parse_args()

    import uiautomator2 as u2

    from minitap.mobile_use.clients.ui_automator_client import _parse_hierarchy_xml_to_elements

    OUT.mkdir(parents=True, exist_ok=True)
    dev = u2.connect(args.serial) if args.serial else u2.connect()

    if args.app:
        print("launch", args.app)
        try:
            dev.app_start(args.app, stop=True)
        except Exception as e:  # noqa: BLE001
            print("  拉起失败（继续，用当前屏）:", e)
        time.sleep(3)

    samples = []
    for i in range(args.n):
        t0 = time.perf_counter()
        xml = dev.dump_hierarchy(compressed=True)
        dt_ms = (time.perf_counter() - t0) * 1000
        f = OUT / f"dump_{i:02d}.xml"
        f.write_text(xml, encoding="utf-8")
        elements = _parse_hierarchy_xml_to_elements(xml)
        samples.append(elements)
        print(
            "  dump#%d  %6.0fms  元素 %3d  可交互 %3d"
            % (i, dt_ms, len(elements), sum(1 for e in elements if _is_interactive(e)))
        )
        if i < args.n - 1:
            time.sleep(args.gap_ms / 1000.0)

    # 判据对比（直接用线上实现，避免探针和代码各写一套）
    from minitap.mobile_use.utils.screen_stability import (
        structural_change_count,
        tree_fingerprint,
    )

    report = {"n": len(samples), "gap_ms": args.gap_ms, "pairs": []}
    for i in range(1, len(samples)):
        a, b = samples[i - 1], samples[i]
        row = {
            "pair": [i - 1, i],
            "n_elements": [len(a), len(b)],
            "old_equal": tree_fingerprint(a) == tree_fingerprint(b),
            "struct_diff": structural_change_count(a, b),
            "struct_diff_exact": structural_change_count(a, b, tolerance_px=0),
            "text_changed": len(
                {str(e.get("text") or "") for e in a} ^ {str(e.get("text") or "") for e in b}
            ),
        }
        report["pairs"].append(row)
        print(
            "  对 %d-%d: 旧指纹=%s  新判据差异=%d（精确=%d）  文本差异=%d"
            % (
                i - 1,
                i,
                "同" if row["old_equal"] else "异",
                row["struct_diff"],
                row["struct_diff_exact"],
                row["text_changed"],
            )
        )

    old_stable = sum(1 for r in report["pairs"] if r["old_equal"])
    new_stable = sum(1 for r in report["pairs"] if r["struct_diff"] == 0)
    print()
    print("判稳率: 旧指纹 %d/%d   新判据 %d/%d" % (old_stable, len(report["pairs"]), new_stable, len(report["pairs"])))

    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n产物:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
