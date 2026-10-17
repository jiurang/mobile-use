"""OCR 感知的单测。

覆盖三层，按「越靠下越依赖真机」排：

1. `utils/ocr_view.py` —— 纯渲染，无依赖。
2. `tools/bounds_validation.py` 的 OCR 分支 —— **这里最容易回归**，因为
   「把 OCR 框当合法来源」和「不能把目测坐标放进来」是一对相反的要求，
   改一边很容易悄悄毁掉另一边。
3. `services/ocr/` 的注册表与配置解析。

真机端到端（引擎真的跑 ONNX）不在这里，跑：
    .venv\\Scripts\\python.exe scripts\\verify_ocr.py
"""

from __future__ import annotations

import base64

import pytest

from minitap.mobile_use.context import PerceptionConfig
from minitap.mobile_use.services.ocr import (
    OcrConfig,
    OcrText,
    available_ocr_engines,
    clear_ocr_engine_cache,
    create_ocr_engine,
    register_ocr_engine,
)
from minitap.mobile_use.tools.bounds_validation import (
    coordinates_look_like_hierarchy_derived_from_elements,
    ocr_texts_as_elements,
)
from minitap.mobile_use.tools.types import Target
from minitap.mobile_use.utils.ocr_view import MAX_TEXT_CHARS, render_ocr_results
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds


# 真机形状：bounds 是字符串 "[x1,y1][x2,y2]"，见 test_bounds_validation 的说明。
def _el(bounds: str, **extra) -> dict:
    return {"bounds": bounds, **extra}


def _target(x: int, y: int, w: int, h: int) -> Target:
    return Target(
        bounds=ElementBounds(x=x, y=y, width=w, height=h),
        resource_id=None,
        resource_id_index=None,
        text=None,
        text_index=None,
    )


def _ocr(text: str, x: int, y: int, w: int, h: int, score: float = 0.95) -> OcrText:
    return OcrText(text=text, score=score, bounds=ElementBounds(x=x, y=y, width=w, height=h))


def _body_lines(block: str) -> list[str]:
    """渲染结果的数据行（跳过 header 和可能存在的空行）。"""
    return [line for line in block.splitlines()[1:] if line.strip()]


# 视频 App 的层级树：只有系统栏和右侧的互动按钮。视频文案 / 顶部导航 / 相关搜索
# 这类文本**完全没有节点** —— 这是 OCR 存在的理由，fixture 必须如实反映。
VIDEO_DUMP = [
    _el("[0,0][1220,2712]", **{"class": "android.widget.FrameLayout"}),
    _el("[0,0][1220,250]", **{"class": "android.widget.FrameLayout"}),
    _el("[1071,1332][1168,1429]", text="2381"),
    _el("[1071,1555][1162,1624]", text="146"),
    _el("[1071,1780][1149,1828]", text="435"),
    _el("[641,2464][760,2510]", text="开宝箱"),
    _el("[52,2536][212,2608]", text="首页"),
    _el("[781,2538][908,2610]", text="消息"),
]


class TestOcrView:
    def test_empty_input_renders_nothing(self):
        """没有结果时返回空串，调用方据此不发消息 —— 而不是发一个空块。"""
        assert render_ocr_results(None) == ""
        assert render_ocr_results([]) == ""

    def test_line_format_is_indexed_and_has_bounds(self):
        block = render_ocr_results([_ocr("首页", 52, 2536, 160, 72, score=0.97)])
        assert "[1]" in block
        assert '"首页"' in block
        assert "(52,2536,160x72)" in block
        assert "0.97" in block

    def test_indices_are_one_based_and_sequential(self):
        block = render_ocr_results([_ocr("a", 1, 1, 1, 1), _ocr("b", 2, 2, 2, 2)])
        lines = _body_lines(block)
        assert len(lines) == 2
        assert lines[0].startswith("[1] ")
        assert lines[1].startswith("[2] ")

    def test_long_text_is_truncated(self):
        """一整屏评论那种长文本不能把 token 撑爆。"""
        long_text = "很" * 500
        block = render_ocr_results([_ocr(long_text, 0, 0, 10, 10)])
        line = _body_lines(block)[0]
        assert long_text not in block
        assert "…" in line
        assert len(line) < MAX_TEXT_CHARS + 80

    def test_non_ascii_is_not_escaped(self):
        """中文不该变成 \\uXXXX —— 实测转义会让每行多出十几个 token。"""
        assert "首页" in render_ocr_results([_ocr("首页", 0, 0, 1, 1)])

    def test_quotes_in_text_stay_escaped(self):
        """text 里带引号时仍然要 json 转义，否则会破坏一行的结构。"""
        block = render_ocr_results([_ocr('say "hi"', 0, 0, 1, 1)])
        assert '\\"hi\\"' in block


class TestOcrIsAValidCoordinateSource:
    """OCR 框是像素测量值，不是目测值，必须能通过坐标校验。"""

    def test_ocr_only_text_is_accepted(self):
        """视频文案：层级树里零命中，OCR 命中 -> 接受。

        没有这一条，OCR 感知就只剩「读」的能力。
        """
        target = _target(33, 2394, 839, 57)  # "在南宁万象城吃到了我的人生三...展开"
        ok, reason = coordinates_look_like_hierarchy_derived_from_elements(
            target, VIDEO_DUMP, [_ocr("在南宁万象城吃到了我的人生三...展开", 33, 2394, 839, 57)]
        )
        assert ok, f"OCR 坐标被误拒: {reason}"

    def test_same_text_is_accepted_even_if_size_drifted(self):
        """OCR 框因为 unclip 会比层级树框略大，尺寸漂移不该误伤。"""
        target = _target(40, 2528, 180, 88)  # 真实 (52,2536,160x72) 外扩一圈
        ok, reason = coordinates_look_like_hierarchy_derived_from_elements(
            target, VIDEO_DUMP, [_ocr("首页", 52, 2536, 160, 72)]
        )
        assert ok, f"尺寸漂移被误拒: {reason}"

    def test_without_ocr_the_same_target_is_rejected(self):
        """OCR 关掉时行为必须和改造前完全一致 —— 这是回归保护的核心断言。"""
        target = _target(33, 2394, 839, 57)
        ok, reason = coordinates_look_like_hierarchy_derived_from_elements(target, VIDEO_DUMP, None)
        assert not ok, "OCR 关闭时竟然放行了层级树之外的坐标"
        assert reason and "resource_id" in reason

    def test_empty_ocr_list_behaves_like_disabled(self):
        target = _target(33, 2394, 839, 57)
        ok, _ = coordinates_look_like_hierarchy_derived_from_elements(target, VIDEO_DUMP, [])
        assert not ok


class TestEyeballedCoordinatesStillRejected:
    """把 OCR 加进合法来源，不能顺手把「目测坐标」也放进来。

    注意这里刻意让 OCR 框**避开**目测坐标：坐标和某个测量框重合时它就该通过
    —— 那说明坐标确实是量出来的，不是猜的。判据是「有没有测量框支持」，
    不是「这个坐标看起来准不准」。
    """

    def test_half_scale_guess_is_still_rejected_even_with_ocr(self):
        """模型在 0.5 倍截图上目测、忘换算 —— OCR 在场也必须拦下。"""
        target = _target(20, 1250, 570, 50)
        ocr = [_ocr("顶部导航", 175, 156, 570, 66)]
        ok, reason = coordinates_look_like_hierarchy_derived_from_elements(target, VIDEO_DUMP, ocr)
        assert not ok, "半分辨率目测坐标被 OCR 通道放行了"
        assert reason and "IoU" in reason

    def test_coordinates_matching_a_measured_ocr_box_are_accepted(self):
        """与 OCR 框完全重合 -> 该通过。坐标是测出来的，不算目测。"""
        target = _target(175, 156, 570, 66)
        ok, reason = coordinates_look_like_hierarchy_derived_from_elements(
            target, VIDEO_DUMP, [_ocr("顶部导航", 175, 156, 570, 66)]
        )
        assert ok, f"与 OCR 框重合的坐标被误拒: {reason}"

    def test_random_mid_screen_guess_is_still_rejected(self):
        """落在两个已知框正中间的瞎猜坐标。"""
        target = _target(500, 2000, 200, 200)
        ok, _ = coordinates_look_like_hierarchy_derived_from_elements(
            target, VIDEO_DUMP, [_ocr("开宝箱", 641, 2464, 119, 46)]
        )
        assert not ok, "瞎猜的中间坐标被放行了"

    def test_occlusion_note_in_reason_mentions_both_sources(self):
        target = _target(20, 1250, 570, 50)
        ok, reason = coordinates_look_like_hierarchy_derived_from_elements(
            target, VIDEO_DUMP, [_ocr("开宝箱", 641, 2464, 119, 46)]
        )
        assert not ok
        assert reason and "OCR" in reason


class TestOcrTextsAsElements:
    def test_shape_is_accepted_by_get_bounds(self):
        """复用层级树的 IoU 路径，形状必须对得上。"""
        from minitap.mobile_use.utils.ui_hierarchy import get_bounds_for_element

        elements = ocr_texts_as_elements([_ocr("首页", 52, 2536, 160, 72)])
        bounds = get_bounds_for_element(elements[0])
        assert bounds == ElementBounds(x=52, y=2536, width=160, height=72)

    def test_text_is_carried_over_for_error_messages(self):
        elements = ocr_texts_as_elements([_ocr("首页", 1, 2, 3, 4)])
        assert elements[0]["text"] == "首页"

    def test_none_and_empty(self):
        assert ocr_texts_as_elements(None) == []
        assert ocr_texts_as_elements([]) == []


class TestPerceptionConfig:
    def test_all_three_channels_default_on(self):
        p = PerceptionConfig()
        assert p.screenshot and p.ui_hierarchy and p.ocr

    def test_channels_are_independent(self):
        p = PerceptionConfig(screenshot=False)
        assert not p.screenshot and p.ui_hierarchy and p.ocr

    def test_ocr_config_is_not_shared_between_instances(self):
        """default_factory 而非可变默认值 —— 否则改一个会污染所有实例。"""
        a = PerceptionConfig()
        b = PerceptionConfig()
        assert a.ocr_config is not b.ocr_config


class TestOcrRegistry:
    def test_builtin_engine_is_registered(self):
        assert "paddle_onnx" in available_ocr_engines()

    def test_unknown_engine_lists_alternatives(self):
        with pytest.raises(ValueError) as exc:
            create_ocr_engine(OcrConfig(engine="does-not-exist"))
        assert "paddle_onnx" in str(exc.value)

    def test_engine_instances_are_cached(self):
        """onnxruntime 加载两个模型要几百毫秒，每轮重建会翻好几倍开销。"""
        clear_ocr_engine_cache()
        calls = []

        class FakeEngine:
            name = "fake"

            def recognize(self, image_base64: str) -> list[OcrText]:
                return []

        def factory(config: OcrConfig):
            calls.append(config)
            return FakeEngine()

        register_ocr_engine("fake", factory)
        try:
            first = create_ocr_engine(OcrConfig(engine="fake"))
            second = create_ocr_engine(OcrConfig(engine="fake"))
            assert first is second
            assert len(calls) == 1
        finally:
            from minitap.mobile_use.services.ocr.registry import _REGISTRY

            _REGISTRY.pop("fake", None)

    def test_different_config_gets_its_own_instance(self):
        clear_ocr_engine_cache()
        try:
            a = create_ocr_engine(OcrConfig(engine="paddle_onnx", det_limit_side_len=960))
            b = create_ocr_engine(OcrConfig(engine="paddle_onnx", det_limit_side_len=1280))
            assert a is not b
        except Exception:
            # 没装 [ocr] 依赖或没有模型时无法构造引擎，这里不假装通过。
            pytest.skip("OCR 引擎不可构造（缺依赖或缺模型）")

    def test_get_ocr_engine_fails_open(self):
        """引擎加载失败只能降级，不能把整个 agent 搞挂。"""
        from minitap.mobile_use.services.ocr import get_ocr_engine

        class FakeCtx:
            perception = PerceptionConfig(ocr_config=OcrConfig(engine="does-not-exist"))

        assert get_ocr_engine(FakeCtx()) is None  # type: ignore[arg-type]


class TestOcrTextModel:
    def test_bounds_reuse_the_ui_hierarchy_shape(self):
        """能直接填进 Target.bounds，不需要在工具层转形状。"""
        text = _ocr("首页", 1, 2, 3, 4)
        bounds = text.bounds
        assert bounds is not None
        target = Target(
            bounds=bounds,
            text=text.text,
            resource_id=None,
            resource_id_index=None,
            text_index=0,
        )
        assert target.bounds is not None
        assert target.bounds.get_center() == bounds.get_center()

    def test_base64_with_data_uri_prefix_is_accepted(self):
        """截图在 contextor 里就是带前缀的形式。"""
        from minitap.mobile_use.services.ocr.paddle_onnx import PaddleOcrOnnxEngine

        raw = base64.b64encode(b"not-an-image").decode()
        assert PaddleOcrOnnxEngine._decode(f"data:image/png;base64,{raw}") is None

    def test_target_accepts_ocr_text_directly(self):
        """模型侧用法：把 OCR 行原样抄进 target。"""
        item = _ocr("首页", 10, 20, 100, 40)
        target = Target(
            bounds=item.bounds,
            text=item.text,
            resource_id=None,
            resource_id_index=None,
            text_index=0,
        )
        assert target.text == "首页"
