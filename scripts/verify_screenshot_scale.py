"""端到端验证：改动 2（截图降分辨率）在真机上的实际效果。

确认三件事：
  1. 走真实 controller 路径后，LMD 拿到的截图确实被缩放到 1x2 tile
  2. 录制路径（recorder）拿到的仍是原分辨率 —— 复盘证据不能糊
  3. 输出仍是合法 JPEG，可被 PIL 解码

跑法：
    .venv\\Scripts\\python.exe scripts\\verify_screenshot_scale.py
"""

from __future__ import annotations

import base64
import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minitap"))

import adbutils  # noqa: E402
from PIL import Image  # noqa: E402

from mobile_use.utils.image_compress import (  # noqa: E402
    TILE_SIZE,
    compress_screenshot_base64,
    compute_scaled_size,
)

DEVICE = "89R8ZXNBGQIRJRLF"
OUT = ROOT / "probe_out" / "screenshot_scale"


def tiles(w: int, h: int) -> tuple[int, int, int]:
    c, r = w // TILE_SIZE + 1, h // TILE_SIZE + 1
    return c, r, c * r * 258


def kb(n: int) -> str:
    return f"{n / 1024:.1f} KB"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    client = adbutils.AdbClient()
    device = client.device(serial=DEVICE)
    raw_png_b64 = base64.b64encode(_png_bytes(device)).decode("utf-8")

    print(f"设备: {DEVICE}")
    print(f"原始截图: {tiles(1220, 2712)}   PNG {kb(len(base64.b64decode(raw_png_b64)))}\n")

    # --- 1. LLM 路径（cortex 用的）---
    llm_b64 = compress_screenshot_base64(raw_png_b64, quality=50)
    llm_bytes = base64.b64decode(llm_b64)
    llm_img = Image.open(BytesIO(llm_bytes))
    expected = compute_scaled_size(1220, 2712)
    (OUT / "final_llm_path.jpg").write_bytes(llm_bytes)

    print("LLM 路径（cortex.py:99）:")
    print(f"  尺寸     {llm_img.width}x{llm_img.height}  期望 {expected[0]}x{expected[1]}")
    print(f"  tiles    {tiles(llm_img.width, llm_img.height)}  (原 {tiles(1220, 2712)})")
    print(f"  体积     {kb(len(llm_bytes))}")
    print(f"  格式     {llm_img.format}")

    src_t, out_t = tiles(1220, 2712)[2], tiles(llm_img.width, llm_img.height)[2]
    print(f"  节省     {(1 - out_t / src_t) * 100:.0f}% token ({src_t} -> {out_t})")

    failures: list[str] = []
    if (llm_img.width, llm_img.height) != expected:
        failures.append(f"LLM 路径尺寸 {llm_img.width}x{llm_img.height} != 期望 {expected}")
    if llm_img.format != "JPEG":
        failures.append(f"格式是 {llm_img.format}，不是 JPEG")
    if tiles(llm_img.width, llm_img.height)[2] >= src_t:
        failures.append("没有省下任何 token")

    # --- 2. 录制路径（recorder 用的）---
    rec_b64 = compress_screenshot_base64(raw_png_b64, max_width=None, max_height=None)
    rec_img = Image.open(BytesIO(base64.b64decode(rec_b64)))
    (OUT / "final_record_path.jpg").write_bytes(base64.b64decode(rec_b64))

    print("\n录制路径（recorder.py:26）:")
    print(f"  尺寸     {rec_img.width}x{rec_img.height}   <- 必须保持原分辨率")
    print(f"  体积     {kb(len(base64.b64decode(rec_b64)))}")
    if (rec_img.width, rec_img.height) != (1220, 2712):
        failures.append(f"录制路径被缩放了：{rec_img.width}x{rec_img.height}，复盘证据会糊")

    # --- 3. 等比例校验 ---
    src_ratio = 1220 / 2712
    out_ratio = llm_img.width / llm_img.height
    print("\n等比例校验:")
    print(
        f"  原始宽高比 {src_ratio:.4f}   缩放后 {out_ratio:.4f}   "
        f"偏差 {abs(src_ratio - out_ratio):.5f}"
    )
    if abs(src_ratio - out_ratio) > 0.01:
        failures.append("缩放不是等比例")

    print()
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS")
    print(f"输出目录: {OUT}")


def _png_bytes(device) -> bytes:
    """把设备截图转成 PNG bytes。"""
    img = device.screenshot().convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    main()
