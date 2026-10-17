"""真机验证 **生产路径**：tools/mobile/tap.py 实际调用的异步校验器。

scripts/verify_ocr_perception.py 验的是同步内核 + 裸 device.click。
但真正跑的时候走的是 `coordinates_look_like_hierarchy_derived`（异步），
它多一段别的逻辑：惰性地在「层级树拒绝之后」才截屏跑 OCR。

**测试全绿 ≠ 这段没被跑过。** 这里显式验证三个分支：

    1. OCR 关闭 + 层级树拒绝        -> 拒绝，且不截屏（零额外开销）
    2. OCR 开启 + 层级树拒绝        -> 现截屏 + 现跑 OCR，再验一次
    3. OCR 开启 + 层级树本来就通过  -> 通过，且**绝不**截屏

跑法：.venv\\Scripts\\python.exe scripts\\verify_ocr_async_path.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).parent.parent))

from adbutils import AdbClient  # noqa: E402
from PIL import Image  # noqa: E402

from minitap.mobile_use.clients.ui_automator_client import (  # noqa: E402
    UIAutomatorClient,
    _parse_hierarchy_xml_to_elements,
)
from minitap.mobile_use.config import get_default_llm_config  # noqa: E402
from minitap.mobile_use.context import (  # noqa: E402
    DeviceContext,
    DevicePlatform,
    MobileUseContext,
    PerceptionConfig,
)
from minitap.mobile_use.services.ocr import create_ocr_engine  # noqa: E402
from minitap.mobile_use.tools.bounds_validation import (  # noqa: E402
    coordinates_look_like_hierarchy_derived,
)
from minitap.mobile_use.tools.types import Target  # noqa: E402

DEVICE_ID = "89R8ZXNBGQIRJRLF"


def heading(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def build_context(ocr_on: bool) -> MobileUseContext:
    return MobileUseContext(
        trace_id="verify-ocr-async",
        device=DeviceContext(
            host_platform="WINDOWS",
            mobile_platform=DevicePlatform.ANDROID,
            device_id=DEVICE_ID,
            device_width=1220,
            device_height=2712,
        ),
        adb_client=AdbClient(host="localhost", port=5037),
        ui_adb_client=UIAutomatorClient(device_id=DEVICE_ID),
        # 校验器不碰 LLM，但 MobileUseContext 强制要求这个字段。
        llm_config=get_default_llm_config(),
        perception=PerceptionConfig(ocr=ocr_on),
    )


def counting_take_screenshot() -> tuple[Any, list[int]]:
    """把 `UnifiedMobileController.take_screenshot` 换成计数版。

    不要用掐时间来判断「有没有多截一次屏」：真机上建 controller 连
    uiautomator2 就要 400~500ms，比 OCR 推理的抖动还大，掐时间必然误判。
    直接数调用次数才是行为断言。
    """
    from minitap.mobile_use.controllers import unified_controller as uc

    calls: list[int] = []
    original = uc.UnifiedMobileController.take_screenshot

    async def counting(self) -> str:
        calls.append(1)
        return await original(self)

    uc.UnifiedMobileController.take_screenshot = counting  # type: ignore[method-assign]
    return original, calls


async def main() -> int:
    import uiautomator2 as u2

    device = u2.connect(DEVICE_ID)

    heading("1) 找一条 OCR 独有、且够大的文本作为目标")
    screenshot = cast(Image.Image, device.screenshot())
    buf = io.BytesIO()
    screenshot.save(buf, format="PNG")
    engine = create_ocr_engine()
    ocr_texts = engine.recognize(base64.b64encode(buf.getvalue()).decode())
    elements = _parse_hierarchy_xml_to_elements(device.dump_hierarchy(compressed=True))

    tree_texts = [
        (element.get(key) or "").strip()
        for element in elements
        for key in ("text", "content-desc")
        if (element.get(key) or "").strip()
    ]
    # 子串也算命中：层级树常把相邻几行合成一个长节点，只做等值比较会误判成
    # 「OCR 独有」，那就证明不了 OCR 补的是层级树真的看不见的东西。
    only_ocr = [
        t for t in ocr_texts if not any(t.text in tree or tree in t.text for tree in tree_texts)
    ]
    candidates = [
        t for t in only_ocr if t.bounds.width >= 90 and t.bounds.height >= 35 and t.score >= 0.8
    ]
    candidates.sort(key=lambda t: -(t.bounds.width * t.bounds.height))
    if not candidates:
        print(
            f"  !! 没找到合适的 OCR 独有文本（OCR 共 {len(ocr_texts)} 条，独有 {len(only_ocr)} 条）"
        )
        return 1
    chosen = candidates[0]
    print(f"  目标: {chosen.text!r}")
    print(f"        {chosen.bounds}  conf={chosen.score:.2f}")

    original_take_screenshot, shots = counting_take_screenshot()
    target = Target(
        text=chosen.text,
        bounds=chosen.bounds,
        resource_id=None,
        resource_id_index=None,
        text_index=0,
    )
    failures: list[str] = []

    heading("2) OCR 关闭：层级树拒绝 -> 拒绝，且不得截屏")
    ctx_off = build_context(ocr_on=False)
    shots.clear()
    ok, reason = await coordinates_look_like_hierarchy_derived(ctx_off, target)
    print(f"  -> {'通过' if ok else '拒绝'}   截屏次数={len(shots)}")
    if not ok and reason:
        print(f"     {reason[:140]}")
    if ok:
        failures.append("OCR 关闭时竟然放行了层级树之外的坐标")
    if shots:
        failures.append(f"OCR 关闭时仍然截了 {len(shots)} 次屏")

    heading("3) OCR 开启：异步路径应当现截屏 + 现跑 OCR 并通过")
    ctx_on = build_context(ocr_on=True)
    shots.clear()
    ok, reason = await coordinates_look_like_hierarchy_derived(ctx_on, target)
    print(f"  -> {'通过' if ok else '拒绝'}   截屏次数={len(shots)}")
    if not ok and reason:
        print(f"     {reason[:200]}")
        failures.append("OCR 开启后坐标仍被拒绝")
    if len(shots) != 1:
        failures.append(f"期望恰好 1 次截屏，实际 {len(shots)} 次")

    heading("4) OCR 开启但层级树本来就通过：不得截屏")
    from minitap.mobile_use.utils.ui_hierarchy import get_bounds_for_element

    node = next(
        (e for e in elements if (e.get("text") or "").strip() and get_bounds_for_element(e)),
        None,
    )
    if node is None:
        print("  !! 层级树里没有可用节点，跳过本项")
    else:
        good = get_bounds_for_element(node)
        assert good is not None
        shots.clear()
        ok, _ = await coordinates_look_like_hierarchy_derived(
            ctx_on,
            Target(
                bounds=good,
                text=(node.get("text") or "").strip(),
                resource_id=None,
                resource_id_index=None,
                text_index=None,
            ),
        )
        print(
            f"  节点 {node.get('text')!r} {good} -> {'通过' if ok else '拒绝'}"
            f"   截屏次数={len(shots)}"
        )
        if not ok:
            failures.append("层级树本来就能通过的坐标被拒了")
        if shots:
            failures.append(f"命中路径白白多截了 {len(shots)} 次屏（惰性逻辑没生效）")

    from minitap.mobile_use.controllers import unified_controller as uc

    uc.UnifiedMobileController.take_screenshot = original_take_screenshot  # type: ignore[method-assign]

    heading("结论")
    if failures:
        for item in failures:
            print(f"  !! {item}")
        return 1
    print("  OCR 关闭 + 层级树拒绝  -> 拒绝，0 次截屏   OK")
    print("  OCR 开启 + 层级树拒绝  -> 通过，1 次截屏   OK")
    print("  OCR 开启 + 层级树命中  -> 通过，0 次截屏   OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
