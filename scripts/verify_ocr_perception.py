"""OCR 感知真机端到端验证。

跑法（需要真机在线）：
    .venv\\Scripts\\python.exe scripts\\verify_ocr_perception.py

回答四个问题，每个都用真机数据回答而不是推断：

1. 三条感知通道的开关是不是真的生效。
2. OCR 看到多少文本，其中多少是 UI 层级树完全没有的。
3. **OCR 独有的文本，用它的坐标能不能真的通过坐标校验** —— 这是 OCR 感知
   从「只读」变成「可操作」的分界线。改造前这里必然被拒。
4. 真按下去之后，屏幕是不是真的变了。

第 3 步同时带一个对照组：目测坐标在 OCR 开启时**仍必须被拒**。
把 OCR 加进合法来源很容易顺手把「目测」也放进来，这个对照就是防那个的。
"""

from __future__ import annotations

import asyncio
import base64
import io
import sys
import time
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image  # noqa: E402
from minitap.mobile_use.context import PerceptionConfig  # noqa: E402
from minitap.mobile_use.services.ocr import create_ocr_engine  # noqa: E402
from minitap.mobile_use.tools.bounds_validation import (  # noqa: E402
    coordinates_look_like_hierarchy_derived_from_elements,
)
from minitap.mobile_use.tools.types import Target  # noqa: E402
from minitap.mobile_use.utils.ocr_view import render_ocr_results  # noqa: E402
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds  # noqa: E402

DEVICE_ID = "89R8ZXNBGQIRJRLF"
OUT = Path(__file__).parent.parent / "probe_out"


def heading(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def to_base64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def main() -> int:
    import uiautomator2 as u2

    from minitap.mobile_use.clients.ui_automator_client import _parse_hierarchy_xml_to_elements

    device = u2.connect(DEVICE_ID)
    OUT.mkdir(exist_ok=True)

    heading("1) 两条感知通道的开关")
    for label, perception in [
        ("全开（默认）", PerceptionConfig()),
        ("仅层级树", PerceptionConfig(ocr=False)),
        ("仅 OCR", PerceptionConfig(ui_hierarchy=False)),
        ("全关", PerceptionConfig(ui_hierarchy=False, ocr=False)),
    ]:
        active = [
            name
            for name, on in (
                ("层级树", perception.ui_hierarchy),
                ("OCR", perception.ocr),
            )
            if on
        ]
        print(f"  {label:<12} -> {' + '.join(active) or '(无视觉感知)'}")

    heading("2) 真机 OCR vs UI 层级树")
    screenshot = cast(Image.Image, device.screenshot())
    screenshot.save(OUT / "verify_ocr.png")
    elements = _parse_hierarchy_xml_to_elements(device.dump_hierarchy(compressed=True))
    print(f"  截图 {screenshot.width}x{screenshot.height}   层级树节点 {len(elements)}")

    engine = create_ocr_engine()
    t0 = time.perf_counter()
    ocr_texts = engine.recognize(to_base64(screenshot))
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  OCR: {len(ocr_texts)} texts in {elapsed:.0f}ms (engine={engine.name})")

    tree_texts = set()
    for element in elements:
        for key in ("text", "content-desc"):
            value = (element.get(key) or "").strip()
            if value:
                tree_texts.add(value)

    only_ocr = [t for t in ocr_texts if t.text not in tree_texts]
    print(f"  OCR 独有（层级树里完全没有）: {len(only_ocr)}/{len(ocr_texts)}")
    for item in only_ocr[:10]:
        print(f"    {item.score:.2f} {item.bounds}  {item.text!r}")

    print("\n" + render_ocr_results(ocr_texts)[:700])

    heading("3) 坐标校验：OCR 框是否被接受")
    candidates = [t for t in only_ocr if t.bounds.width > 60 and t.bounds.height > 25]
    if not candidates:
        print("  !! 本屏没有够大的 OCR 独有文本，换一个视频/自绘 UI 的 App 再跑")
        return 1

    target_item = candidates[0]
    target = Target(
        text=target_item.text,
        bounds=target_item.bounds,
        resource_id=None,
        resource_id_index=None,
        text_index=0,
    )
    print(f"  目标: {target_item.text!r}  {target_item.bounds}")

    ok_off, reason_off = coordinates_look_like_hierarchy_derived_from_elements(
        target, elements, None
    )
    ok_on, reason_on = coordinates_look_like_hierarchy_derived_from_elements(
        target, elements, ocr_texts
    )
    print(f"  OCR 关闭 -> {'通过' if ok_off else '拒绝'}")
    if not ok_off and reason_off:
        print(f"             {reason_off[:150]}")
    print(f"  OCR 开启 -> {'通过' if ok_on else '拒绝'}")
    if not ok_on and reason_on:
        print(f"             {reason_on[:150]}")

    print("\n  对照组 —— 目测坐标（OCR 开启时仍必须拒绝）：")
    b = target_item.bounds
    guesses = [
        (
            "半分辨率目测",
            ElementBounds(x=b.x // 2, y=b.y // 2, width=b.width // 2, height=b.height // 2),
        ),
        (
            "原尺寸偏移 300px",
            ElementBounds(x=b.x + 300, y=b.y + 300, width=b.width, height=b.height),
        ),
        (
            "尺寸缩到 1/4",
            ElementBounds(x=b.x, y=b.y, width=max(1, b.width // 4), height=max(1, b.height // 4)),
        ),
    ]
    all_rejected = True
    for label, guess in guesses:
        ok, _ = coordinates_look_like_hierarchy_derived_from_elements(
            Target(
                bounds=guess,
                resource_id=None,
                resource_id_index=None,
                text=None,
                text_index=None,
            ),
            elements,
            ocr_texts,
        )
        all_rejected &= not ok
        print(f"    {label:<20} -> {'通过 (!! 有问题)' if ok else '拒绝'}")

    if not ok_on:
        print("\n  !! OCR 坐标仍被拒绝 —— OCR 感知不具备可操作性")
        return 1
    if not all_rejected:
        print("\n  !! 目测坐标被放行 —— 校验器被削弱了")
        return 1

    heading("4) 真按下去")
    before = cast(Image.Image, device.screenshot())
    before_pixels = before.tobytes()

    center = target_item.bounds.get_center()
    print(f"  tap ({center.x}, {center.y})  <-  {target_item.text!r}")
    device.click(center.x, center.y)
    time.sleep(3)

    after = cast(Image.Image, device.screenshot())
    after.save(OUT / "verify_ocr_after.png")
    changed = after.tobytes() != before_pixels

    after_texts = engine.recognize(to_base64(after))
    before_names = {t.text for t in ocr_texts}
    appeared = [t.text for t in after_texts if t.text not in before_names]

    print(f"  画面变化        : {'是' if changed else '否'}")
    print(f"  点击后新出现文本: {appeared[:8]}")

    heading("结论")
    print(f"  OCR 独有文本         : {len(only_ocr)}/{len(ocr_texts)}")
    print(f"  OCR 独有文本可点击   : {'是' if ok_on else '否'}")
    print(f"  目测坐标仍被拦截     : {'是' if all_rejected else '否'}")
    print(f"  点击后画面变化       : {'是' if changed else '否'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
