"""实测：真机截图在「不缩放 / 等比例缩小」两种情况下的体积与可读性。

回答三个问题，全部用真机数据，不靠估算：
  1. 真机截图的真实分辨率，以及截图分辨率是否等于设备分辨率（坐标系是否一致）
  2. quality=50 JPEG 在不同缩放档位下的字节数
  3. 缩放后 Gemini tile 计价公式给出的 token 数变化

跑法：
    .venv\\Scripts\\python.exe scripts\\probe_screenshot_scale.py
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minitap"))

import adbutils  # noqa: E402
from PIL import Image  # noqa: E402

DEVICE = "89R8ZXNBGQIRJRLF"
OUT_DIR = ROOT / "probe_out" / "screenshot_scale"

# Gemini / OpenAI 兼容视觉接口的通用 tile 计价：(w//768+1) * (h//768+1) * 258
TILE = 768
TOKENS_PER_TILE = 258

SCALES = [1.0, 0.75, 0.5, 0.4]


def gemini_tokens(w: int, h: int) -> int:
    return (w // TILE + 1) * (h // TILE + 1) * TOKENS_PER_TILE


def tile_dims(w: int, h: int) -> tuple[int, int]:
    """实际会被切成几列 x 几行 tile。"""
    return (w // TILE + 1), (h // TILE + 1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 设备真实分辨率
    client = adbutils.AdbClient()
    device = client.device(serial=DEVICE)
    wm_size = str(device.shell("wm size")).strip()
    print(f"wm size: {wm_size}")

    # 2. 抓一张真截图
    pil = device.screenshot()
    print(f"screenshot: {pil.width}x{pil.height} mode={pil.mode}")
    pil.save(OUT_DIR / "raw.png")

    print()
    hdr = (
        f"{'scale':>6s} {'WxH':>12s} {'tiles':>8s} {'tokens':>8s} "
        f"{'jpeg q50 B':>11s} {'vs raw':>8s} {'saved tok':>10s}"
    )
    print(hdr)
    print("-" * len(hdr))

    # 基准：原始 PNG 的字节数
    raw_png = OUT_DIR / "raw.png"
    png_bytes = raw_png.stat().st_size

    baseline_tokens = gemini_tokens(pil.width, pil.height)

    rows = []
    for s in SCALES:
        if s == 1.0:
            img = pil
        else:
            # 等比例缩小，关键：坐标契约仍然来自层级树，与本图无关
            nw, nh = int(pil.width * s), int(pil.height * s)
            img = pil.resize((nw, nh), Image.LANCZOS)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=50, optimize=True)
        jpg = buf.getvalue()

        cols, rws = tile_dims(img.width, img.height)
        tok = gemini_tokens(img.width, img.height)

        name = f"scale_{int(s * 100):03d}.jpg"
        (OUT_DIR / name).write_bytes(jpg)

        rows.append(
            f"{s:6.2f} {img.width:5d}x{img.height:<5d} {cols:3d}x{rws:<3d} "
            f"{tok:8,} {len(jpg):11,} {(1 - len(jpg) / png_bytes) * 100:7.1f}% "
            f"{(1 - tok / baseline_tokens) * 100:9.1f}%"
        )

    print("\n".join(rows))

    print()
    print(f"raw PNG: {png_bytes:,} B   baseline tokens: {baseline_tokens:,}")
    print(f"输出目录: {OUT_DIR}")
    print("人工比对：打开 scale_100.jpg 与 scale_050.jpg，看状态栏/按钮小字是否还可读。")


if __name__ == "__main__":
    main()
