"""截图等比例缩小的单测。

真机数据来源：scripts/probe_screenshot_scale.py（Redmi Turbo 4, 1220x2712）
    scale   尺寸          tiles   tokens
    1.00   1220x2712     2x4     2,064
    0.75    915x2034     2x3     1,548
    0.50    610x1356     1x2       516
    0.40    488x1084     1x2       516   <- 与 0.5 同 token，只省带宽
"""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from minitap.mobile_use.utils.image_compress import (
    DEFAULT_MAX_HEIGHT,
    DEFAULT_MAX_WIDTH,
    TILE_SIZE,
    compress_screenshot_base64,
    compute_scaled_size,
)


def _make_image(w: int, h: int, mode: str = "RGB") -> str:
    fill = {
        "RGB": (120, 180, 240),
        "RGBA": (120, 180, 240, 255),
        "L": 160,
    }[mode]
    img = Image.new(mode, (w, h), fill)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _dims_of(b64: str) -> tuple[int, int]:
    img = Image.open(BytesIO(base64.b64decode(b64)))
    return img.width, img.height


def _tiles(w: int, h: int) -> tuple[int, int]:
    return (w // TILE_SIZE + 1), (h // TILE_SIZE + 1)


class TestComputeScaledSize:
    def test_real_device_screen_is_downscaled(self):
        w, h = compute_scaled_size(1220, 2712)
        assert (w, h) != (1220, 2712), "1220 宽应被压到 tile 边长以内"
        assert w <= DEFAULT_MAX_WIDTH, f"宽 {w} 未降到 {DEFAULT_MAX_WIDTH}"

    def test_scale_is_proportional(self):
        """等比例是硬约束：只缩宽或只缩高会让图与层级树坐标语义错乱。"""
        w, h = compute_scaled_size(1220, 2712)
        src_ratio = 1220 / 2712
        out_ratio = w / h
        assert abs(src_ratio - out_ratio) < 0.005, f"宽高比 {out_ratio} 偏离原始 {src_ratio}"

    def test_small_image_is_untouched(self):
        assert compute_scaled_size(400, 800) == (400, 800)

    def test_at_default_limit_is_untouched(self):
        assert compute_scaled_size(640, 1536) == (640, 1536)

    def test_above_default_limit_is_scaled(self):
        w, h = compute_scaled_size(768, 768)
        assert w < 768 and h < 768

    def test_none_limit_disables_scaling(self):
        assert compute_scaled_size(1220, 2712, None, None) == (1220, 2712)

    def test_both_limits_take_the_stricter_one(self):
        """两个上限同时生效时必须取更严格的，否则可能出现非等比例。"""
        w, h = compute_scaled_size(1220, 2712, max_width=768, max_height=1600)
        assert abs(1220 / 2712 - w / h) < 0.005

    def test_never_collapses_to_zero(self):
        w, h = compute_scaled_size(10000, 20000, max_width=1, max_height=1)
        assert w >= 1 and h >= 1


class TestTokenSavings:
    def test_crosses_into_single_column_tile(self):
        """省 token 的关键是宽度压到 tile 以内 —— 1220 会占 2 列。"""
        _, h = compute_scaled_size(1220, 2712)
        src_cols, _ = _tiles(1220, 2712)
        out_cols, _ = _tiles(compute_scaled_size(1220, 2712)[0], h)
        assert src_cols == 2
        assert out_cols == 1, "缩放后仍占 2 列，等于没省"

    def test_default_lands_on_1x2_tile(self):
        """默认配置必须落在 1x2 tile = 516 token，而不是 1x3 = 1548。

        这是实测踩过的坑：宽度压到 768 会得到 768x1707，高度多吃一行 tile，
        token 只省 25%。要落进 2 行，宽度必须 <= 690（见 scripts/probe_tile_target.py）。
        """
        w, h = compute_scaled_size(1220, 2712)
        assert _tiles(w, h) == (1, 2), f"实际 {_tiles(w, h)}，应为 (1, 2)"

    def test_width_768_would_waste_a_tile_row(self):
        """锁住「按 TILE_SIZE 缩放」这个直觉是错的。"""
        w, h = compute_scaled_size(1220, 2712, max_width=768, max_height=1536)
        assert _tiles(w, h) == (1, 3), f"实际 {_tiles(w, h)}，预期 (1, 3)"
        # 而默认配置能拿到 2 行
        dw, dh = compute_scaled_size(1220, 2712)
        assert _tiles(dw, dh)[1] == 2

    def test_shrinking_within_same_tile_row_gains_nothing(self):
        """同一 tile 行数内继续缩小 = 白缩。

        真机实测：0.5 档 610x1356 与 0.4 档 488x1084 同为 1x2 tile = 516 token。
        尺寸差了 20%，token 一分没省，只白丢清晰度。

        锁住这个事实，防止有人以为「越小越好」而缩到同 tile 内更小的尺寸。
        """
        w1, h1 = compute_scaled_size(1220, 2712)          # 1x2
        w2, h2 = compute_scaled_size(1220, 2712, 500, 1200)  # 仍是 1x2
        assert _tiles(w1, h1) == _tiles(w2, h2), "token 一样，白缩了"
        assert w2 < w1

    def test_tile_row_reduction_requires_both_sides_small_enough(self):
        """到 1x1 tile 要求宽高都 <=768，等比缩放下这意味着极小尺寸。

        1220x2712 的宽高比是 2.22，想让高 <=768 就得把宽压到 345 —— 此时细节基本全丢。
        这解释了为什么默认停在 1x2：再省一半 token 的代价过大。
        """
        w, h = compute_scaled_size(1220, 2712, max_width=345, max_height=768)
        assert _tiles(w, h) == (1, 1), f"实际 {_tiles(w, h)}"
        # 但这已经把宽度压到 345，可读性不可接受
        assert w < 400
        # 而默认配置停在 1x2
        dw, dh = compute_scaled_size(1220, 2712)
        assert _tiles(dw, dh) == (1, 2)

    def test_default_height_bound_keeps_tile_count_reasonable(self):
        assert DEFAULT_MAX_HEIGHT >= TILE_SIZE


class TestCompress:
    def test_returns_jpeg_and_downscaled(self):
        out = compress_screenshot_base64(_make_image(1220, 2712))
        img = Image.open(BytesIO(base64.b64decode(out)))
        assert img.format == "JPEG"
        assert img.width <= TILE_SIZE

    def test_no_data_uri_prefix(self):
        """cortex 那边会自己拼 data URI 前缀，这里重复会导致前缀嵌套。"""
        out = compress_screenshot_base64(_make_image(400, 800))
        assert not out.startswith("data:")

    def test_accepts_input_with_data_uri_prefix(self):
        prefixed = "data:image/png;base64," + _make_image(400, 800)
        out = compress_screenshot_base64(prefixed)
        assert _dims_of(out) == (400, 800)

    def test_shrinks_bytes(self):
        big = _make_image(1220, 2712)
        out = compress_screenshot_base64(big)
        assert len(base64.b64decode(out)) < len(base64.b64decode(big))

    def test_can_disable_scaling_for_recording(self):
        """录制证据路径必须能拿到原分辨率，否则复盘时看不清小字。"""
        out = compress_screenshot_base64(_make_image(1220, 2712), max_width=None, max_height=None)
        assert _dims_of(out) == (1220, 2712)

    def test_rgba_is_flattened_not_crashed(self):
        """JPEG 不支持 alpha；iOS / 云设备的截图带透明通道，不能直接存。"""
        out = compress_screenshot_base64(_make_image(400, 800, mode="RGBA"))
        img = Image.open(BytesIO(base64.b64decode(out)))
        assert img.mode == "RGB"

    def test_grayscale_is_accepted(self):
        out = compress_screenshot_base64(_make_image(400, 800, mode="L"))
        assert Image.open(BytesIO(base64.b64decode(out))).format == "JPEG"
