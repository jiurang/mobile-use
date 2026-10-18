"""探针：OCR 渲染出来的 `[i]`，能不能原样喂给 tap。

纯本地、无设备。回答三个问题：
1. `Target` 能不能直接吃那个索引（`ocr_index`）。
2. 渲染出来的编号和还原用的编号是不是同一套（不改去重、不改重排）。
3. 越界 / 越界的 0 有没有被如实拒绝，而不是悄悄点别处。
"""

from __future__ import annotations

from minitap.mobile_use.services.ocr.types import OcrText
from minitap.mobile_use.tools.bounds_validation import (
    coordinates_look_like_hierarchy_derived_from_elements,
)
from minitap.mobile_use.tools.types import Target
from minitap.mobile_use.utils.ocr_view import render_ocr_results, resolve_ocr_index
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds


def ocr(text: str, x: int, y: int, w: int = 200, h: int = 60) -> OcrText:
    return OcrText(text=text, score=0.95, bounds=ElementBounds(x=x, y=y, width=w, height=h))


print("=" * 72)
print("Q1  Target 能不能直接吃 [i]？")
print("=" * 72)
print(f"  Target fields = {list(Target.model_fields.keys())}")
print(f"  Target(ocr_index=3) = {Target(ocr_index=3).ocr_index}")

print()
print("=" * 72)
print("Q2  渲染的编号 == 还原用的编号？")
print("=" * 72)
raw = [
    ocr("搜索", 100, 200),
    ocr("热搜", 100, 400),       # 这条在层级树里也有，但不去重、也不跳号
    ocr("全屏观看", 700, 2000),
    ocr("6.8万", 100, 800),
]
block = render_ocr_results(raw)
print(block)
print()
ok = True
for i, item in enumerate(raw, 1):
    hit = resolve_ocr_index(raw, i)
    same = hit is item
    ok = ok and same
    print(f"  [{i}] -> {'命中同一条' if same else '对不上!!'}  {item.text!r} @ {item.bounds}")
print(f"  => {'全部对齐' if ok else '存在错位'}")

print()
print("=" * 72)
print("Q3  OCR 框能否通过 tap 的坐标门禁？")
print("=" * 72)
tree_has_it = [{"text": "搜索", "bounds": {"x": 100, "y": 200, "width": 200, "height": 60}}]
target_hit = Target(ocr_index=1).model_copy(update={"bounds": raw[0].bounds})
ok_a, why_a = coordinates_look_like_hierarchy_derived_from_elements(
    target_hit, tree_has_it, raw
)
print(f"  树里有对应节点   -> accepted={ok_a} reason={why_a}")

target_miss = Target(ocr_index=3).model_copy(update={"bounds": raw[2].bounds})
ok_b, why_b = coordinates_look_like_hierarchy_derived_from_elements(target_miss, [], None)
ok_b2, why_b2 = coordinates_look_like_hierarchy_derived_from_elements(
    target_miss, tree_has_it, raw
)
print(f"  树里没有 + OCR 关 -> accepted={ok_b}")
print(f"  树里没有 + OCR 开 -> accepted={ok_b2} reason={why_b2}")

guessed = Target(bounds=ElementBounds(x=610, y=1400, width=200, height=60))
ok_c, why_c = coordinates_look_like_hierarchy_derived_from_elements(guessed, tree_has_it, raw)
print(f"  目测坐标 + OCR 开 -> accepted={ok_c}")
print(f"     reason={why_c}")

print()
print("=" * 72)
print("Q4  坏索引有没有被如实拒绝？")
print("=" * 72)
for bad in (0, -1, 4, 99):
    print(f"  resolve_ocr_index(results, {bad:>3}) = {resolve_ocr_index(raw, bad)}")
print(f"  resolve_ocr_index(None, 1) = {resolve_ocr_index(None, 1)}")

print()
print("=" * 72)
print("Q5  ocr_index 和 text_index 是两套编号")
print("=" * 72)
t = Target(text="搜索", text_index=2)
print(f"  Target(text='搜索', text_index=2) -> text_index={t.text_index}, ocr_index={t.ocr_index}")
print("  text_index 索引的是层级树里第几个同文本节点（0-based）；")
print("  ocr_index 索引的是 OCR 块里的行（1-based）。二者不可互换。")
print()
print("done.")
