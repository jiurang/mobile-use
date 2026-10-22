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
    def test_ui_and_ocr_default_on(self):
        p = PerceptionConfig()
        assert p.ui_hierarchy and p.ocr

    def test_screenshot_defaults_to_on_demand(self):
        """截图默认**按需**（2026-10-21 改，2026-10-22 改成显式枚举）。

        截图是最贵的一块，而且永远不可缓存：它在 prompt 末尾，base64 长度随
        画面变 → 请求总长每轮都变 → provider 的缓存单元匹配不上 → 归零。
        绝大多数决策靠元素表 + OCR 就够，图只在颜色/图标/遮挡这些地方才需要。
        """
        p = PerceptionConfig()
        assert p.screenshot == "on_demand"
        assert p.screenshot_channel == "on_demand"
        assert p.can_see_screenshot is True

    def test_channels_are_independent(self):
        """关掉一条不影响另外两条 —— 而且「关掉」是**字符串**不是假值。

        ⚠️ 以前 `screenshot` 是 bool，`not p.screenshot` 判的就是「关着」；
        现在它是枚举字符串（`"off"` 是**真值**）⇒ 用 `can_see_screenshot` 判。
        """
        p = PerceptionConfig(screenshot="off")
        assert p.can_see_screenshot is False
        assert p.ui_hierarchy and p.ocr

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


class _FakeOrt:
    """只提供 `resolve_providers` 用到的那一个接口。"""

    def __init__(self, available: list[str]):
        self._available = available

    def get_available_providers(self) -> list[str]:
        return list(self._available)


class _FakeSession:
    def __init__(self, providers: list[str]):
        self._providers = providers

    def get_providers(self) -> list[str]:
        return self._providers


class TestResolveProviders:
    def _resolve(self, device: str, available: list[str]):
        from minitap.mobile_use.services.ocr.paddle_onnx import resolve_providers

        return resolve_providers(OcrConfig(device=device), _FakeOrt(available))

    def test_cpu_only_runtime_never_requests_cuda(self):
        """装的是 CPU 版 onnxruntime 时，auto 必须老实要 CPU。"""
        providers, expected, explicit = self._resolve("auto", ["CPUExecutionProvider"])
        assert providers == ["CPUExecutionProvider"]
        assert expected == "CPUExecutionProvider"
        assert explicit is False

    def test_auto_prefers_cuda_when_available(self):
        providers, expected, _ = self._resolve(
            "auto", ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        assert expected == "CUDAExecutionProvider"
        # CPU 必须一起给：ORT 是节点级回退，只给 GPU 时遇到没实现的算子直接报错。
        assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_auto_does_not_pick_dml(self):
        """DirectML 实测比 CPU 慢，还要额外装包，auto 不能选它。

        选了就是「用户没要求 GPU，却因为有独显被悄悄换到更慢的后端」。
        """
        providers, expected, _ = self._resolve(
            "auto", ["DmlExecutionProvider", "CPUExecutionProvider"]
        )
        assert expected == "CPUExecutionProvider"
        assert providers == ["CPUExecutionProvider"]

    def test_explicit_cuda_on_cpu_only_runtime_stays_explicit(self):
        """点名了 cuda 却没拿到：仍然要报成「显式但没兑现」，让上层 warn。

        不能因为「反正降级了」就标成 auto —— 那会把用户的显式要求
        静默吞掉，日志里只留一条 info。
        """
        _, expected, explicit = self._resolve("cuda", ["CPUExecutionProvider"])
        assert expected == "CPUExecutionProvider"
        assert explicit is True

    def test_explicit_device_wins_over_auto(self):
        _, expected, explicit = self._resolve(
            "cpu", ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        assert expected == "CPUExecutionProvider"
        assert explicit is True

    def test_unknown_device_falls_back_to_auto_without_raising(self):
        """拼错的 device 名不该把整轮感知搞挂。"""
        providers, expected, explicit = self._resolve(
            "gpu-please", ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        assert expected == "CUDAExecutionProvider"
        assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert explicit is False

    def test_empty_available_still_yields_cpu(self):
        providers, expected, _ = self._resolve("cuda", [])
        assert providers == ["CPUExecutionProvider"]
        assert expected == "CPUExecutionProvider"

    def test_device_is_case_and_space_insensitive(self):
        _, expected, explicit = self._resolve(
            "  CUDA ", ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        assert expected == "CUDAExecutionProvider"
        assert explicit is True


class TestVerifyProvider:
    """ORT 拿到 GPU provider 后初始化失败会**静默**退回 CPU。

    不对账的话，「配了 GPU」和「跑在 GPU 上」完全是两件事，
    而代码看起来一切正常 —— 这是 GPU 化最容易出现的假成功。
    """

    def test_matching_backend_is_returned(self):
        from minitap.mobile_use.services.ocr.paddle_onnx import verify_provider

        session = _FakeSession(["CUDAExecutionProvider", "CPUExecutionProvider"])
        assert verify_provider(session, "CUDAExecutionProvider", True, "cuda") == (
            "CUDAExecutionProvider"
        )

    def test_silent_fallback_to_cpu_is_detected(self):
        from minitap.mobile_use.services.ocr.paddle_onnx import verify_provider

        session = _FakeSession(["CPUExecutionProvider"])
        assert verify_provider(session, "CUDAExecutionProvider", True, "cuda") == (
            "CPUExecutionProvider"
        )

    def test_empty_provider_list_does_not_raise(self):
        from minitap.mobile_use.services.ocr.paddle_onnx import verify_provider

        session = _FakeSession([])
        assert verify_provider(session, "CUDAExecutionProvider", True, "cuda") == "(none)"


class TestResolveDevice:
    """`配置 -> 环境变量 -> auto` 的优先级，和 `resolve_model_dir` 同一套。

    有环境变量入口是故意的：换后端要连带 dll 路径、调 onnxruntime 包，
    这些都在环境里摆着，不该要求使用者改代码。
    """

    @staticmethod
    def _resolve(device: str | None, monkeypatch, env: str | None = None):
        from minitap.mobile_use.services.ocr.paddle_onnx import resolve_device

        if env is None:
            monkeypatch.delenv("MOBILE_USE_OCR_DEVICE", raising=False)
        else:
            monkeypatch.setenv("MOBILE_USE_OCR_DEVICE", env)
        return resolve_device(OcrConfig(device=device))

    def test_default_is_auto(self, monkeypatch):
        assert self._resolve(None, monkeypatch) == "auto"

    def test_env_var_is_used_when_config_is_unset(self, monkeypatch):
        assert self._resolve(None, monkeypatch, env="cuda") == "cuda"

    def test_explicit_config_beats_env_var(self, monkeypatch):
        """在代码里写死 device 时，环境变量**不能**覆盖它 ——
        否则 CI 里设一次 MOBILE_USE_OCR_DEVICE 就悄悄改掉了本地显式配置。
        """
        assert self._resolve("cpu", monkeypatch, env="cuda") == "cpu"

    def test_is_normalized(self, monkeypatch):
        assert self._resolve("  CUDA ", monkeypatch) == "cuda"

    def test_blank_env_var_falls_back_to_auto(self, monkeypatch):
        assert self._resolve(None, monkeypatch, env="   ") == "auto"


class TestRegisterCudaDllDirs:
    def test_no_env_means_no_handles(self, monkeypatch):
        from minitap.mobile_use.services.ocr import paddle_onnx

        monkeypatch.delenv("MOBILE_USE_CUDA_DLL_DIR", raising=False)
        before = len(paddle_onnx._DLL_DIR_HANDLES)
        assert paddle_onnx.register_cuda_dll_dirs() == []
        assert len(paddle_onnx._DLL_DIR_HANDLES) == before

    def test_missing_dir_is_returned_but_does_not_raise(self, monkeypatch, tmp_path):
        """目录不存在**不能抛** —— dll 缺失会表现为降级到 CPU，
        在这里崩等于把「降级上报」变成「硬拦截」。

        PATH 方式对不存在的目录天然无害（LoadLibrary 跳过），所以照样返回。
        """
        from minitap.mobile_use.services.ocr import paddle_onnx

        monkeypatch.setattr(paddle_onnx, "_DLL_DIR_HANDLES", [], raising=False)
        monkeypatch.setenv("MOBILE_USE_CUDA_DLL_DIR", str(tmp_path / "nope"))
        assert paddle_onnx.register_cuda_dll_dirs() == [str(tmp_path / "nope")]

    def test_path_env_var_is_the_thing_that_matters(self, monkeypatch, tmp_path):
        """**关键**：ORT 用原生 `LoadLibrary`，不认 `add_dll_directory`。

        真正让 dll 被找到的是改进程内 PATH，不是注册 dll 目录。少这一步的
        症状是：日志明明打了 `cuda_dll_dirs=...`，建 session 仍然报
        `cublasLt64_12.dll` 缺失（实测踩过）。
        """
        import os

        from minitap.mobile_use.services.ocr import paddle_onnx

        monkeypatch.setattr(paddle_onnx, "_DLL_DIR_HANDLES", [], raising=False)
        monkeypatch.setenv("MOBILE_USE_CUDA_DLL_DIR", str(tmp_path))
        monkeypatch.setenv("PATH", "C:\\windows")

        registered = paddle_onnx.register_cuda_dll_dirs()
        assert registered == [str(tmp_path)]
        assert str(tmp_path) in os.environ["PATH"].split(os.pathsep)
        assert len(paddle_onnx._DLL_DIR_HANDLES) == 1  # 句柄必须留活

    def test_existing_path_entries_are_not_duplicated(self, monkeypatch, tmp_path):
        import os

        from minitap.mobile_use.services.ocr import paddle_onnx

        monkeypatch.setattr(paddle_onnx, "_DLL_DIR_HANDLES", [], raising=False)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setenv("MOBILE_USE_CUDA_DLL_DIR", str(tmp_path))

        paddle_onnx.register_cuda_dll_dirs()
        assert os.environ["PATH"].split(os.pathsep).count(str(tmp_path)) == 1


class _FakeRecSession:
    """记录每次 run 的输入形状和内容，用来断言批处理真的把调用压少了。"""

    def __init__(self):
        self.calls: list[tuple[int, int]] = []
        self.blobs: list[object] = []

    def run(self, _outputs, feed):
        import numpy as np

        blob = feed["x"]
        self.calls.append((int(blob.shape[0]), int(blob.shape[3])))
        self.blobs.append(blob)
        return [np.zeros((blob.shape[0], 8, 4), dtype=np.float32)]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]


def _crops(count: int, width: int = 30):
    import numpy as np

    return [np.full((20, width, 3), 128, dtype=np.uint8) for _ in range(count)]


# crop 高 20px 时按自身高宽比缩放出来的目标宽度：ceil(20 * 320 / 48) = 134。
_EXPECTED_TARGET_W = 134


class TestRecBatching:
    """批处理是 GPU 化的前提：CUDA 单次 run 约 40ms，CPU 只要 6.2ms。

    不把 26 次调用压成 1~2 次，GPU 会比 CPU 慢 4 倍。
    """

    KEYS = ["a", "b"]

    @staticmethod
    def _engine(mode: str, session: _FakeRecSession):
        from minitap.mobile_use.services.ocr.paddle_onnx import PaddleOcrOnnxEngine
        from minitap.mobile_use.services.ocr.types import OcrConfig

        engine = _BatchHarness(OcrConfig(rec_batch=mode), session)
        engine._recognize_crops = PaddleOcrOnnxEngine._recognize_crops.__get__(engine)
        return engine

    def test_serial_calls_once_per_crop(self):
        session = _FakeRecSession()
        self._engine("serial", session)._recognize_crops(_crops(5))
        assert len(session.calls) == 5

    def test_pad_collapses_to_a_single_call(self):
        session = _FakeRecSession()
        self._engine("pad", session)._recognize_crops(_crops(5))
        assert session.calls[0][0] == 5

    def test_resize_collapses_to_a_single_call(self):
        session = _FakeRecSession()
        self._engine("resize", session)._recognize_crops(_crops(5))
        assert session.calls[0][0] == 5

    def test_pad_keeps_each_row_at_its_own_width(self):
        """pad 不额外拉伸窄框 —— 这是选它做默认的原因。

        `_rec_row` 的宽度只由 **height** 决定（PaddleOCR 的 rec 就是按固定高 48
        + 比例宽训练的），所以矮框算出的 target_w 就窄。pad 保留每行自己算出的
        宽度（**与 serial 逐像素一致**），只把右侧补黑到组内最大宽度；
        而 `resize` 会把所有行强行拉到组内最大宽度，矮框被横向拉长。

        这里用 height=20（target_w=134）和 height=48（target_w=320）两行来证明：
        batch 宽度是 320，但矮行只有前 134 列是真内容，右边全是填充。
        """
        import numpy as np

        session = _FakeRecSession()
        short = np.full((20, 20, 3), 128, dtype=np.uint8)
        tall = np.full((48, 200, 3), 128, dtype=np.uint8)
        self._engine("pad", session)._recognize_crops([short, tall])

        batch_n, batch_w = session.calls[0]
        assert batch_n == 2
        assert batch_w == 320  # 组内最大 = 48px 高那行的 target_w

        blob = session.blobs[0]
        from minitap.mobile_use.services.ocr.paddle_onnx import _REC_PAD_VALUE

        # 矮行：前 134 列有内容，之后全是黑色填充（说明没被拉成 320 宽）
        assert np.all(blob[0, :, :, 134:] == _REC_PAD_VALUE)
        assert not np.all(blob[0, :, :, :134] == _REC_PAD_VALUE)
        # 高行：整行填满，没有填充
        assert not np.all(blob[1] == _REC_PAD_VALUE)

    def test_resize_stretches_short_rows_to_common_width(self):
        """对照组：resize 会把矮行也填满 320 宽 —— 几何被改动了。"""
        import numpy as np

        session = _FakeRecSession()
        short = np.full((20, 20, 3), 128, dtype=np.uint8)
        tall = np.full((48, 200, 3), 128, dtype=np.uint8)
        self._engine("resize", session)._recognize_crops([short, tall])

        blob = session.blobs[0]
        from minitap.mobile_use.services.ocr.paddle_onnx import _REC_PAD_VALUE

        assert session.calls[0] == (2, 320)
        # 没有黑色填充，因为两行都被拉满了
        assert not np.any(blob == _REC_PAD_VALUE)

    def test_pad_fill_value_is_black_not_mid_grey(self):
        """填充值是归一化后的纯黑 (-1)；用 0 会被 CTC 当成中灰内容读出来。"""
        from minitap.mobile_use.services.ocr.paddle_onnx import _REC_PAD_VALUE

        assert _REC_PAD_VALUE == -1.0

    def test_single_crop_still_works_in_pad_mode(self):
        session = _FakeRecSession()
        assert len(self._engine("pad", session)._recognize_crops(_crops(1))) == 1
        assert session.calls[0][0] == 1

    def test_empty_input_is_safe(self):
        assert self._engine("pad", _FakeRecSession())._recognize_crops([]) == []

    def test_default_is_pad(self):
        """schema 默认是 None（「未设置」），实际生效值才是 pad。

        区分这两者很重要：None 才能读环境变量，默认写成 "pad" 的话
        `MOBILE_USE_OCR_REC_BATCH` 永远不生效。
        """
        from minitap.mobile_use.services.ocr.types import resolve_rec_batch

        assert OcrConfig().rec_batch is None
        assert resolve_rec_batch(OcrConfig()) == "pad"


class TestResolveRecBatch:
    """批处理方式必须先归一化再用。

    改造前直接拿裸字符串比 `== "serial"`，于是 `rec_batch="SERIAL"` 会掉进
    else 分支被当成 pad —— **识别结果悄悄变了**（标点、长句都可能不同），
    而且没有任何提示。
    """

    @staticmethod
    def _resolve(raw: str | None, monkeypatch, env: str | None = None):
        from minitap.mobile_use.services.ocr.types import resolve_rec_batch

        if env is None:
            monkeypatch.delenv("MOBILE_USE_OCR_REC_BATCH", raising=False)
        else:
            monkeypatch.setenv("MOBILE_USE_OCR_REC_BATCH", env)
        return resolve_rec_batch(OcrConfig(rec_batch=raw))

    def test_legal_values_pass_through(self, monkeypatch):
        for mode in ("serial", "pad", "resize"):
            assert self._resolve(mode, monkeypatch) == mode

    def test_case_and_whitespace_are_normalized(self, monkeypatch):
        assert self._resolve("SERIAL", monkeypatch) == "serial"
        assert self._resolve("Resize", monkeypatch) == "resize"
        assert self._resolve("  pad  ", monkeypatch) == "pad"

    def test_illegal_value_falls_back_and_warns(self, monkeypatch, caplog):
        """降级必须有 warning —— 静默降级在这里等于悄悄改识别结果。"""
        import logging

        from minitap.mobile_use.utils.logger import get_logger

        underlying = get_logger("minitap.mobile_use.services.ocr.types").logger
        with caplog.at_level(logging.WARNING, logger=underlying.name):
            assert self._resolve("bogus", monkeypatch) == "pad"
        assert any("rec_batch" in r.getMessage() for r in caplog.records)

    def test_blank_means_unset_and_does_not_warn(self, monkeypatch, caplog):
        """空值 = 没设置，回落到默认是正常的，不该刷警告。"""
        import logging

        from minitap.mobile_use.utils.logger import get_logger

        underlying = get_logger("minitap.mobile_use.services.ocr.types").logger
        with caplog.at_level(logging.WARNING, logger=underlying.name):
            assert self._resolve("", monkeypatch) == "pad"
            assert self._resolve(None, monkeypatch) == "pad"
        assert not any("rec_batch" in r.getMessage() for r in caplog.records)

    def test_env_var_is_used_when_config_is_unset(self, monkeypatch):
        assert self._resolve(None, monkeypatch, env="serial") == "serial"

    def test_explicit_config_beats_env_var(self, monkeypatch):
        assert self._resolve("resize", monkeypatch, env="serial") == "resize"

    def test_normalization_actually_changes_the_run_count(self, monkeypatch):
        """回归：`"SERIAL"` 之前会被当成 pad（1 次调用），归一化后必须是 8 次。"""
        from minitap.mobile_use.services.ocr.paddle_onnx import PaddleOcrOnnxEngine

        counts = {}
        for raw in ("serial", "SERIAL", "pad", "resize", "RESIZE"):
            session = _FakeRecSession()
            engine = _BatchHarness(OcrConfig(rec_batch=raw), session)
            engine._recognize_crops = PaddleOcrOnnxEngine._recognize_crops.__get__(engine)
            engine._recognize_crops(_crops(8))
            counts[raw] = len(session.calls)
        assert counts["SERIAL"] == counts["serial"] == 8
        assert counts["RESIZE"] == counts["resize"] == 1
        assert counts["pad"] == 1


class TestDeviceIdReachesOnnxRuntime:
    """`device_id` 之前是纯装饰：写进了缓存 key，文档也写着「多卡指定哪张卡」，
    但**从来没传给 onnxruntime** —— 多卡机器上改它毫无作用，也不报错。
    """

    @staticmethod
    def _options(providers: list[str], device_id: int = 0):
        from minitap.mobile_use.services.ocr.paddle_onnx import resolve_provider_options

        return resolve_provider_options(OcrConfig(device_id=device_id), providers)

    def test_gpu_provider_receives_device_id(self):
        options = self._options(["CUDAExecutionProvider", "CPUExecutionProvider"], 2)
        assert options[0] == {"device_id": "2"}

    def test_directml_provider_receives_device_id(self):
        options = self._options(["DmlExecutionProvider", "CPUExecutionProvider"], 1)
        assert options[0] == {"device_id": "1"}

    def test_cpu_provider_does_not(self):
        """CPU EP 不认 device_id，给了会被 ORT 判为非法 key。"""
        assert self._options(["CPUExecutionProvider"], 2) == [{}]

    def test_length_always_matches_providers(self):
        for providers in (
            ["CPUExecutionProvider"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CUDAExecutionProvider", "CPUExecutionProvider", "DmlExecutionProvider"],
        ):
            assert len(self._options(providers)) == len(providers)

    def test_options_are_actually_passed_to_the_session(self, monkeypatch, tmp_path):
        """端到端：确认 provider_options 真的交给了 InferenceSession 构造函数。"""
        import minitap.mobile_use.services.ocr.paddle_onnx as paddle_onnx

        captured: list[dict] = []

        class FakeSession:
            def __init__(self, path, options=None, providers=None, provider_options=None):
                captured.append({"providers": providers, "provider_options": provider_options})

            def get_providers(self):
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        class FakeOptions:
            """SessionOptions 会被写 log_severity_level / intra_op_num_threads。"""

            def __init__(self):
                self.log_severity_level = 0
                self.intra_op_num_threads = 0

        class FakeOrt:
            SessionOptions = FakeOptions
            InferenceSession = FakeSession

            @staticmethod
            def get_available_providers():
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        for name in ("det.onnx", "rec.onnx", "keys.txt"):
            (tmp_path / name).write_bytes(b"x")

        monkeypatch.setattr(paddle_onnx, "_require_deps", lambda: (None, FakeOrt, None))
        paddle_onnx.PaddleOcrOnnxEngine(OcrConfig(model_dir=tmp_path, device="cuda", device_id=3))
        assert len(captured) == 2  # det + rec
        for call in captured:
            assert call["provider_options"][0] == {"device_id": "3"}


class TestCacheKeyCoversEveryField:
    """加字段时忘了加进缓存 key = 「改配置没反应但也不报错」。

    这是最难查的一类 bug，所以用测试守住：key 必须覆盖全部配置字段
    （含经 `resolve_*` 间接读取的）。
    """

    @staticmethod
    def _fields_in_key() -> set[str]:
        import inspect
        import re

        from minitap.mobile_use.services.ocr.registry import _cache_key
        from minitap.mobile_use.services.ocr import types as ocr_types

        found = set(re.findall(r"config\.(\w+)", inspect.getsource(_cache_key)))
        # 自动发现所有 resolve_* 辅助函数，而不是硬编码名单 —— resolve_engine
        # 是后加的，硬编码过一次就会在它下一次被漏掉时静默失去保护。
        for name, fn in vars(ocr_types).items():
            if not name.startswith("resolve_") or not callable(fn):
                continue
            try:
                found |= set(re.findall(r"config\.(\w+)", inspect.getsource(fn)))
            except (OSError, TypeError):
                # C 扩展/内建对象没有源码，跳过即可（真 resolve_* 都是纯 Python）。
                continue
        return found

    def test_no_field_is_missing_from_the_cache_key(self):
        missing = set(OcrConfig.model_fields) - self._fields_in_key()
        assert not missing, (
            f"这些字段不在 OCR 引擎缓存 key 里，改了它们不会换实例：{sorted(missing)}。"
            " 在 registry._cache_key 里补上。"
        )

    def test_key_must_not_bypass_the_resolvers(self):
        """key 里只能出现**解析后**的值，直接读原始字段是同一个 bug 换个写法。

        `config.device` 在走环境变量时是 None —— 用它算 key 的话，
        `=cuda` 和 `=cpu` 算出同一个 key，改配置不报错却没反应。
        上面的「字段出现过」检查抓不到这种，因为它只认字段名不认取值方式。
        """
        import ast
        import inspect

        from minitap.mobile_use.services.ocr.registry import _cache_key

        # 用 AST 只看真实的属性访问，不看注释 —— 注释里为了说明问题会提到
        # `config.device`，按正则扫文本会把说明文字当成代码。
        tree = ast.parse(inspect.getsource(_cache_key))
        used = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        for field in ("device", "rec_batch"):
            assert field not in used, (
                f"_cache_key 直接读了 config.{field}，绕过了解析函数。"
                " 环境变量路径下它是 None，会让不同后端算出同一个 key。"
            )

    def test_max_texts_is_covered(self):
        """max_texts 曾被漏掉 —— 当时靠实例各自持有 config 才侥幸正确。"""
        from minitap.mobile_use.services.ocr.registry import _cache_key

        assert _cache_key(OcrConfig(max_texts=3)) != _cache_key(OcrConfig(max_texts=80))


class _BatchHarness:
    """只借用 `_recognize_crops` 的最小宿主，避免真去建 ONNX session。"""

    def __init__(self, config, rec_sess):
        self._config = config
        self._rec_sess = rec_sess
        self._keys = ["a", "b"]

    _recognize_crops = None  # 由 _engine 绑定真实实现


class TestDeviceIsPartOfTheCacheKey:
    def test_same_config_different_device_are_different_instances(self):
        """换后端必须换实例，否则改配置后还拿着 CPU session 在跑。"""
        clear_ocr_engine_cache()
        calls = []

        class FakeEngine:
            name = "fake"

            def __init__(self, config: OcrConfig):
                self.device = config.device

            def recognize(self, image_base64: str) -> list[OcrText]:
                return []

        def factory(config: OcrConfig):
            calls.append(config.device)
            return FakeEngine(config)

        register_ocr_engine("fake", factory)
        try:
            a = create_ocr_engine(OcrConfig(engine="fake", device="cpu"))
            b = create_ocr_engine(OcrConfig(engine="fake", device="cuda"))
            assert a is not b
            assert calls == ["cpu", "cuda"]
        finally:
            from minitap.mobile_use.services.ocr.registry import _REGISTRY

            _REGISTRY.pop("fake", None)

    def test_device_id_also_part_of_key(self):
        from minitap.mobile_use.services.ocr.registry import _cache_key

        a = _cache_key(OcrConfig(device="cuda", device_id=0))
        b = _cache_key(OcrConfig(device="cuda", device_id=1))
        assert a != b

    def test_env_var_rec_batch_is_part_of_the_key(self, monkeypatch):
        """和 device 同理：rec_batch 也有环境变量入口，所以也必须进 key。

        只有 device 有这层测试时是漏的 —— 覆盖缺口和 bug 一模一样。
        """
        from minitap.mobile_use.services.ocr.registry import _cache_key

        monkeypatch.setenv("MOBILE_USE_OCR_REC_BATCH", "serial")
        with_serial = _cache_key(OcrConfig())
        monkeypatch.setenv("MOBILE_USE_OCR_REC_BATCH", "resize")
        with_resize = _cache_key(OcrConfig())
        assert with_serial != with_resize

    def test_env_var_device_is_part_of_the_key(self, monkeypatch):
        """回归：走环境变量时 `config.device` 是 None。

        如果缓存 key 直接用它，`MOBILE_USE_OCR_DEVICE=cuda` 和 `=cpu` 会算出
        同一个 key，于是设成 cpu 之后拿到的还是上一次那个 CUDA 实例 ——
        **没有任何报错**，只是静默跑错了后端。实测踩过。
        """
        from minitap.mobile_use.services.ocr.registry import _cache_key

        monkeypatch.setenv("MOBILE_USE_OCR_DEVICE", "cuda")
        with_cuda = _cache_key(OcrConfig())
        monkeypatch.setenv("MOBILE_USE_OCR_DEVICE", "cpu")
        with_cpu = _cache_key(OcrConfig())
        assert with_cuda != with_cpu

    def test_env_var_change_returns_a_fresh_engine(self, monkeypatch):
        """不只是 key 不同，还要保证真能换掉实例。"""
        clear_ocr_engine_cache()
        seen = []

        class FakeEngine:
            name = "fake"

            def __init__(self, config: OcrConfig):
                self.requested = config.device

            def recognize(self, image_base64: str) -> list[OcrText]:
                return []

        def factory(config: OcrConfig):
            from minitap.mobile_use.services.ocr.types import resolve_device

            seen.append(resolve_device(config))
            return FakeEngine(config)

        register_ocr_engine("fake", factory)
        try:
            monkeypatch.setenv("MOBILE_USE_OCR_DEVICE", "cuda")
            create_ocr_engine(OcrConfig(engine="fake"))
            monkeypatch.setenv("MOBILE_USE_OCR_DEVICE", "cpu")
            create_ocr_engine(OcrConfig(engine="fake"))
            assert seen == ["cuda", "cpu"]
        finally:
            from minitap.mobile_use.services.ocr.registry import _REGISTRY

            _REGISTRY.pop("fake", None)


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
