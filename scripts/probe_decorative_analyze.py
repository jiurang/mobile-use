"""把实测结果（`decorative_rows.jsonl`）和账本（`decorative_impact.json`）对起来，
算出**每个候选判据**的：裁掉多少行 / 裁掉��里有多少真能点 / 省多少 token。

判据候选
--------
* `A` 原始判据：flags 里没有 C/L/F/K/S，且无文字
* `B` A + R 也算可交互（executor 提示词说 R 行的 bounds 用来做 swipe）
* `C` B + 类名本身可操作的控件也算（EditText / ImageButton …）
* `D` C + 「有 resource_id 的行不裁」（仓库自己的既有结论：id 名字本身就是线索）
* `E<T>` C + 只裁面积极小的行（width*height < T）
* `F` C + 只裁「几何容器」行（它的框里还套着别的行 ⇒ 它只是个包装）

「真能点」的两种口径
-------------------
* `hit`：屏幕有任何可见变化（像素差 > 0.3%）
* `hit_strong`：屏幕明显换了个画面（像素差 > 3%）—— 排除焦点抖动/微动画
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "probe_out"
ROWS = HERE / "decorative_rows.jsonl"
IMPACT = HERE / "decorative_impact.json"

STRONG_THRESHOLD = 0.03


def _load_rows() -> list[dict]:
    if not ROWS.exists():
        return []
    lines = ROWS.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _contains(outer: dict, inner: dict) -> bool:
    """inner 的框是不是完全落在 outer 的框里。"""
    return (
        outer["x"] <= inner["x"]
        and outer["y"] <= inner["y"]
        and outer["x"] + outer["w"] >= inner["x"] + inner["w"]
        and outer["y"] + outer["h"] >= inner["y"] + inner["h"]
    )


def main() -> int:
    rows = _load_rows()
    if not ROWS.exists():
        print("没有 decorative_rows.jsonl —— 先跑 probe_decorative_clickable.py")
        return 1
    if not IMPACT.exists():
        print("没有 decorative_impact.json —— 先跑 probe_decorative_crop_impact.py")
        return 1

    impact = json.loads(IMPACT.read_text(encoding="utf-8"))
    table_rows = impact["rows"]

    # ---- 几何容器判定：这一行的框里还套着别的（会被保留的）行吗 ----
    by_page: dict[str, list[dict]] = {}
    for row in table_rows:
        by_page.setdefault(row["package"], []).append(row)

    def is_container(row: dict) -> bool:
        page = by_page.get(row["package"], [])
        for other in page:
            if other is row:
                continue
            if other["kind"] == "ocr":
                continue
            # 只看**会保留**的行（不是装饰候选）
            if _is_decor(other):
                continue
            if _contains(row, other):
                return True
        return False

    def _is_decor(row: dict) -> bool:
        flags = set(row["flags"] or "")
        has_text = bool((row["label"] or "").strip() or (row["text"] or "").strip())
        if row["kind"] == "ocr":
            return False
        if flags & set("CLFKSR"):
            return False
        if row["class_name"] in {
            "EditText",
            "CheckBox",
            "RadioButton",
            "Switch",
            "SeekBar",
            "Spinner",
            "ToggleButton",
            "ImageButton",
            "AutoCompleteTextView",
            "Button",
        }:
            return False
        return not has_text

    # ---- 把实测结果按 (包名, id, 中心点) 挂到表行上 ----
    outcome_by_key: dict[tuple, dict] = {}
    for record in rows:
        key = (record["package"], record["id"], record["cx"], record["cy"])
        outcome_by_key[key] = record

    matched = 0
    unmatched = 0
    for row in table_rows:
        if row["kind"] == "ocr":
            continue
        cx = row["x"] + row["w"] // 2
        cy = row["y"] + row["h"] // 2
        key = (row["package"], row["resource_id"], cx, cy)
        record = outcome_by_key.get(key)
        if record is None:
            row["outcome"] = "unmeasured"
            unmatched += 1
        else:
            row["outcome"] = record["outcome"]
            row["ratio"] = record["ratio"]
            matched += 1

    print(f"实测行 {len(rows)}，挂上 {matched}，没挂上 {unmatched}（页面漂移导致）")
    print()

    # ---- 逐判据算账 ----
    names = ["A", "B", "C", "D"] + [f"E{t}" for t in (400, 1600, 6400, 25000)] + ["F"]

    def crops(row: dict, name: str) -> bool:
        if name == "F":
            return row["criteria"]["C"] and is_container(row)
        return row["criteria"][name]

    print(
        f"{'判据':<7}{'裁掉行':>8}{'实测到':>8}{'hit':>7}{'hit%':>8}"
        f"{'strong':>8}{'strong%':>9}{'省 tok':>9}{'占比':>8}"
    )
    for name in names:
        cut = [
            r
            for r in table_rows
            if r["kind"] != "ocr" and crops(r, name)
        ]
        measured = [r for r in cut if r["outcome"] in ("responded", "no_change")]
        hits = [r for r in measured if r["outcome"] == "responded"]
        strong = [r for r in hits if r.get("ratio", 0) > STRONG_THRESHOLD]
        after = impact["tok_after"].get(name)
        saved = (impact["tok_before"] - after) if after is not None else 0
        pct = saved / impact["tok_before"] if impact["tok_before"] else 0
        hit_rate = f"{len(hits) / len(measured):.1%}" if measured else "-"
        strong_rate = f"{len(strong) / len(measured):.1%}" if measured else "-"
        print(
            f"{name:<7}{len(cut):>8}{len(measured):>8}{len(hits):>7}{hit_rate:>8}"
            f"{len(strong):>8}{strong_rate:>9}{saved:>9}{pct:>8.0%}"
        )

    print()
    print("总行数", impact["total_rows"], "，总 tok", impact["tok_before"])

    # ---- F 判据的命中明细 ----
    print()
    print("F 判据（只裁几何容器）里被裁掉、但实测点了有反应的行：")
    for row in table_rows:
        if row["kind"] == "ocr" or not crops(row, "F"):
            continue
        if row["outcome"] == "responded":
            print(
                f"  [{row['package']}] id={row['resource_id']} "
                f"box=({row['x']},{row['y']},{row['w']}x{row['h']}) "
                f"像素差={row.get('ratio', 0):.2%}"
            )
    print()
    print("所有实测命中明细（不管判据）：")
    for row in table_rows:
        if row.get("outcome") == "responded":
            print(
                f"  [{row['package']}] id={row['resource_id']} cls={row['class_name']!r} "
                f"box=({row['x']},{row['y']},{row['w']}x{row['h']}) "
                f"像素差={row.get('ratio', 0):.2%} "
                f"flags={row['flags']!r} 容器={is_container(row)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
