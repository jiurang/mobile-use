"""截图压缩：给 LLM 用的那条路径做等比例缩小。

为什么要缩
----------
截图是每轮 cortex 请求里最贵的一块。真机实测（scripts/probe_screenshot_scale.py，
Redmi Turbo 4, 1220x2712）：

    scale   尺寸          tiles   tokens   JPEG q50
    1.00   1220x2712     2x4     2,064    172,223 B
    0.75    915x2034     2x3     1,548    117,449 B
    0.50    610x1356     1x2       516     69,174 B
    0.40    488x1084     1x2       516     50,542 B

视觉接口按 768px tile 计价：`(w//768+1) * (h//768+1) * 258`，所以缩放的意义在于
**跨过 tile 边界**：
- 1220x2712 -> 2x4 tile = 2064 token
- 768x1707  -> 1x3 tile = 1548 token（宽度过了，高度多吃一行，只省 25%）
- 610x1356  -> 1x2 tile =  516 token（省 75%）

注意 0.5 档与 0.4 档（488x1084）**同为 1x2 tile，同为 516 token** —— 在同一个
tile 内继续缩小只省带宽不省 token，纯属白丢清晰度。但跨到 1x1（640x768）确实能再省
一半，代价是清晰度。所以默认取 610x1356：省 75%，且已实测可读。

默认目标 610x1356（1x2 tile = 516 token，原 2064，省 75%）。
这个尺寸的可读性已在真机上人工比对过：底部导航、互动数字、正文文字、小红点数字
全部清晰；「开宝箱」角标与顶部灰字导航开始发虚（scripts/probe_screenshot_scale.py
的 scale_050.jpg 可自行查看）。

为什么缩放不影响点击
--------------------
点击坐标来自 UI 层级树（`cortex.md` 明写 "Can't reliably extract precise element
coordinates from pixels"），不依赖截图像素；且 `tools/bounds_validation.py`
会拒绝任何不与层级树对齐的坐标。所以这里等比例缩小是安全的。

**必须等比例**：只缩宽或只缩高会让图像与层级树坐标语义彻底错乱。

录制路径不受影响
----------------
`utils/recorder.py` 存的是事后复盘用的证据图，默认不缩放（`max_width=None`），
避免排查问题时拿到一张糊掉的截图。
"""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

# 视觉 tile 边长：`(w//768+1) * (h//768+1) * 258`。
TILE_SIZE = 768

# 默认长边上限 = 640。真机 1220x2712 等比缩到 610x1356，落在 1x2 tile = 516 token
# （原 2x4 = 2064，省 75%）。
#
# 为什么不是 768：这是实测踩过的坑。压到 768 宽会得到 768x1707，落在 1x3 tile
# = 1548 token —— 只省了 25%，因为高度又多吃了一行。要落进 2 行，高度必须 <= 1536，
# 对应宽度上限 690。取 640 是留了余量，同时仍是 1x2 tile。
# 见 scripts/probe_tile_target.py。
DEFAULT_MAX_WIDTH = 640

# 高度上限同样按 tile 边界取：<= 1536 才是 2 行。
DEFAULT_MAX_HEIGHT = 1536


def _flatten_to_rgb(image: Image.Image) -> Image.Image:
    """JPEG 不支持 alpha；带透明通道的图要先合成到白底。

    uiautomator / idb 出来的截图通常是 RGB，但 iOS 与部分云设备会带 alpha，
    直接存 JPEG 会报错或得到黑底。
    """
    if image.mode in ("RGB", "L"):
        return image
    if image.mode in ("RGBA", "LA"):
        rgb = Image.new("RGB", image.size, (255, 255, 255))
        rgb.paste(image, mask=image.split()[-1])
        return rgb
    return image.convert("RGB")


def compute_scaled_size(
    width: int,
    height: int,
    max_width: int | None = DEFAULT_MAX_WIDTH,
    max_height: int | None = DEFAULT_MAX_HEIGHT,
) -> tuple[int, int]:
    """等比例算出目标尺寸；已经够小就原样返回。

    两个上限同时生效时取更严格的那个 —— 两者都可能来自不同配置，
    不能各自独立地决定缩放比例，否则可能出现非等比例的结果。
    """
    scale = 1.0
    for limit, current in ((max_width, width), (max_height, height)):
        if limit and current > limit:
            scale = min(scale, limit / current)

    if scale >= 1.0:
        return width, height
    # 至少 1px，避免极端输入把尺寸压成 0
    return max(1, int(width * scale)), max(1, int(height * scale))


def compress_screenshot_base64(
    image_base64: str,
    quality: int = 50,
    max_width: int | None = DEFAULT_MAX_WIDTH,
    max_height: int | None = DEFAULT_MAX_HEIGHT,
) -> str:
    """把 base64 截图重编码为 JPEG，可选先等比例缩小。返回不带 data URI 前缀的 base64。

    Args:
        image_base64: 可以带 `data:image/...;base64,` 前缀，也可以是裸 base64。
        quality: JPEG 质量，1-95。
        max_width: 长边上限；None 表示不缩放。
        max_height: 短边上限；None 表示不缩放。
    """
    if image_base64.startswith("data:image"):
        image_base64 = image_base64.split(",", 1)[1]

    image = Image.open(BytesIO(base64.b64decode(image_base64)))
    image = _flatten_to_rgb(image)

    target_w, target_h = compute_scaled_size(image.width, image.height, max_width, max_height)
    if (target_w, target_h) != (image.width, image.height):
        # Pillow 9.1 起重采样枚举从 `Image.LANCZOS` 移到 `Image.Resampling.LANCZOS`。
        # 仓库锁的是新版 Pillow，直接用 Resampling；旧版环境不在支持范围内，
        # 与其加 getattr 兜底（pyright 仍会报属性错误），不如显式失败。
        image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
