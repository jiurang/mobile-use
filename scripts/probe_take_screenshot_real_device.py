"""真机实测：截图工具（`take_screenshot`）在真手机上到底能不能用。

回答四个问题，全部用真机数据：

1. 工具能不能拿到**当屏**的图（内容对不对 —— 和直接 adb 截图逐像素比）；
2. 图有没有被压到 tile 预算内（≈610x1356，1x2 tile ⇒ ~500 token）；
3. 工具结果是不是「一段文字 + 一个 image_url 块」（能直接发给 provider）；
4. 一次调用要花多久（模型每轮都在等这个）。

跑法：
    .venv\\Scripts\\python.exe scripts\\probe_take_screenshot_real_device.py

产物：
    probe_out/take_screenshot_real.jpg   —— 人眼可看的那张图
    probe_out/take_screenshot_real.json  —— 全部度量
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minitap"))

import adbutils  # noqa: E402
from PIL import Image, ImageChops, ImageStat  # noqa: E402

from minitap.mobile_use.clients.ui_automator_client import UIAutomatorClient  # noqa: E402
from minitap.mobile_use.config import initialize_llm_config  # noqa: E402
from minitap.mobile_use.context import DeviceContext, DevicePlatform, MobileUseContext  # noqa: E402
from minitap.mobile_use.tools.mobile.take_screenshot import get_take_screenshot_tool  # noqa: E402

DEVICE = "89R8ZXNBGQIRJRLF"
OUT_DIR = ROOT / "probe_out"

#: 视觉接口按 768px tile 计价：(w//768+1) * (h//768+1) * 258
TILE = 768
TOKENS_PER_TILE = 258


class _FakeState:
    """工具只需要 `asanitize_update`。"""

    async def asanitize_update(self, *, ctx, update, agent):
        return update


def _tokens(w: int, h: int) -> int:
    return (w // TILE + 1) * (h // TILE + 1) * TOKENS_PER_TILE


def _mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    """两张图的平均像素差（0~255）。用来判断「是不是同一屏」。"""
    size = (64, 64)
    small_a = a.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    small_b = b.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    diff = ImageChops.difference(small_a, small_b)
    return sum(ImageStat.Stat(diff).mean) / 3


async def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    adb = adbutils.AdbClient()
    dev = adb.device(serial=DEVICE)
    size = str(dev.shell("wm size"))
    print(f"设备 {DEVICE}  wm size: {size.strip()}")

    ctx = MobileUseContext(
        trace_id="probe-take-screenshot",
        device=DeviceContext(
            host_platform="WINDOWS",
            mobile_platform=DevicePlatform.ANDROID,
            device_id=DEVICE,
            device_width=1220,
            device_height=2712,
        ),
        adb_client=adb,
        ui_adb_client=UIAutomatorClient(device_id=DEVICE),
        llm_config=initialize_llm_config(),
    )

    # ---- 直接抓一张，作为「当屏」的基准 ----
    t0 = time.perf_counter()
    reference = dev.screenshot()
    ref_ms = round((time.perf_counter() - t0) * 1000)
    print(f"基准截图（adb 直抓）: {reference.width}x{reference.height}  {ref_ms}ms")

    # ---- 跑真正的工具 ----
    tool = get_take_screenshot_tool(ctx)
    t0 = time.perf_counter()
    out = await tool.coroutine(  # type: ignore[misc]
        agent_thought="探针：抓一张当屏图",
        tool_call_id="probe_call_1",
        state=_FakeState(),  # type: ignore[arg-type]
    )
    tool_ms = round((time.perf_counter() - t0) * 1000)

    message = out.update["executor_messages"][-1]
    blocks = message.content
    print(f"\n工具调用: {tool_ms}ms   状态: {message.status}")
    print(f"结果块类型: {[b.get('type') for b in blocks]}")

    assert isinstance(blocks, list), "内容必须是块列表"
    image_block = next(b for b in blocks if b.get("type") == "image_url")
    url = image_block["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,"), url[:40]
    b64 = url.split(",", 1)[1]
    jpeg = Image.open(BytesIO(base64.b64decode(b64)))

    print(
        f"图片: {jpeg.width}x{jpeg.height}  {len(b64) // 1024}KB base64  "
        f"≈{_tokens(jpeg.width, jpeg.height)} token"
    )

    diff = _mean_abs_diff(reference, jpeg)
    print(f"与基准图的平均像素差: {diff:.2f}/255 （越小越是同一屏）")

    (OUT_DIR / "take_screenshot_real.jpg").write_bytes(base64.b64decode(b64))

    result = {
        "device": DEVICE,
        "reference_size": [reference.width, reference.height],
        "reference_capture_ms": ref_ms,
        "tool_call_ms": tool_ms,
        "image_size": [jpeg.width, jpeg.height],
        "image_kb": len(b64) // 1024,
        "image_tokens": _tokens(jpeg.width, jpeg.height),
        "mean_abs_diff_vs_reference": round(diff, 2),
        "block_types": [b.get("type") for b in blocks],
        "status": message.status,
    }
    (OUT_DIR / "take_screenshot_real.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  => {OUT_DIR / 'take_screenshot_real.jpg'}")
    print(f"  => {OUT_DIR / 'take_screenshot_real.json'}")

    ok = diff < 12.0
    verdict = "图与当屏一致 ✅" if ok else "图与基准差太多 ❌"
    print(f"\n结论: {verdict}（像素差 {round(diff, 2)}）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
