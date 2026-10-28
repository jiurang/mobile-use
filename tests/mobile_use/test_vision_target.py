"""视觉主体扫描（``scan_visual_subjects``）的离线回归测试。

覆盖四件事：

1. 归一化框 → 像素框的换算（越界夹紧、空框拒绝）；
2. 扫描输出 → 编号表的映射（过滤空描述 / 非法框，截断到 20 项，异常降级）；
3. Executor 的 ``Target.vision_index`` 反解（命中、越界、无缓存、指纹过期）；
4. 提示词 / 工具集契约（中英模板、开关）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "minitap"))

import pytest  # noqa: E402

from minitap.mobile_use.services import vision  # noqa: E402
from minitap.mobile_use.tools.bounds_validation import (  # noqa: E402
    coordinates_look_like_hierarchy_derived_from_elements,
)
from minitap.mobile_use.tools.types import Target  # noqa: E402
from minitap.mobile_use.tools.utils import resolve_target  # noqa: E402
from minitap.mobile_use.utils.screen_elements import scan_cache_fingerprint  # noqa: E402
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds  # noqa: E402

_PROMPTS = Path(__file__).resolve().parents[2] / "minitap" / "mobile_use" / "agents" / "executor"


def test_normalized_box_maps_to_pixels():
    bounds = vision.bounds_from_normalized_box(0, 0, 500, 1000, 1000, 2000)
    assert bounds == ElementBounds(x=0, y=0, width=500, height=2000)


def test_normalized_box_is_clamped_to_the_screen():
    bounds = vision.bounds_from_normalized_box(-50, -50, 1050, 1200, 1000, 2000)
    assert bounds == ElementBounds(x=0, y=0, width=1000, height=2000)


@pytest.mark.parametrize(
    "box",
    [
        (500, 0, 500, 1000),  # 宽度为 0
        (0, 1000, 1000, 1000),  # 高度为 0
        (0, 0, 0, 0),  # 整个框退化成一个点
    ],
)
def test_empty_boxes_are_rejected(box):
    assert vision.bounds_from_normalized_box(*box, 1000, 2000) is None


# ---------------------------------------------------------------------------
# 扫描：列表质量与降级
# ---------------------------------------------------------------------------


def _vision_ctx() -> SimpleNamespace:
    return SimpleNamespace(
        device=SimpleNamespace(device_width=1000, device_height=2000),
        trace_id="test",
        prompt_language="en",
    )


def _patch_vision_pipeline(monkeypatch, response=None, error: Exception | None = None) -> None:
    class _Runner:
        async def ainvoke(self, messages):
            if error is not None:
                raise error
            return response

    async def _invoke(coro, **kwargs):
        return await coro

    monkeypatch.setattr(vision, "compress_screenshot_base64", lambda raw, quality: "img")
    monkeypatch.setattr(vision, "get_structured_llm", lambda **kwargs: _Runner())
    monkeypatch.setattr(vision, "invoke_llm_with_timeout_message", _invoke)


@pytest.mark.asyncio
async def test_scan_maps_subjects_to_a_numbered_table(monkeypatch):
    response = vision.VisionScanOutput(
        subjects=[
            vision.VisionSubject(
                description="gear icon, top right", x1=900, y1=100, x2=980, y2=200
            ),
            vision.VisionSubject(description="", x1=0, y1=0, x2=100, y2=100),
            vision.VisionSubject(description="broken box", x1=500, y1=500, x2=500, y2=600),
            vision.VisionSubject(
                description="photo of a red-roofed house", x1=100, y1=400, x2=900, y2=800
            ),
        ]
    )
    _patch_vision_pipeline(monkeypatch, response)

    elements, reason = await vision.scan_screen_subjects(_vision_ctx(), "raw")

    assert reason == ""
    assert [element.index for element in elements] == [1, 2]
    assert elements[0].label == "gear icon, top right"
    assert elements[0].bounds == ElementBounds(x=900, y=200, width=80, height=200)
    assert elements[1].label == "photo of a red-roofed house"
    assert elements[1].bounds == ElementBounds(x=100, y=800, width=800, height=800)


@pytest.mark.asyncio
async def test_scan_truncates_to_the_subject_limit(monkeypatch):
    subjects = [
        vision.VisionSubject(
            description=f"icon {i}", x1=0, y1=i, x2=100, y2=i + 10
        )
        for i in range(vision.MAX_SUBJECTS + 5)
    ]
    _patch_vision_pipeline(monkeypatch, vision.VisionScanOutput(subjects=subjects))

    elements, reason = await vision.scan_screen_subjects(_vision_ctx(), "raw")

    assert reason == ""
    assert len(elements) == vision.MAX_SUBJECTS
    assert elements[-1].index == vision.MAX_SUBJECTS


@pytest.mark.asyncio
async def test_scan_failure_degrades_to_empty_with_a_reason(monkeypatch):
    _patch_vision_pipeline(monkeypatch, error=RuntimeError("model does not accept images"))

    elements, reason = await vision.scan_screen_subjects(_vision_ctx(), "raw")

    assert elements == []
    assert "model does not accept images" in reason


def test_scan_result_tells_the_model_how_to_address_a_subject():
    elements = [
        vision.VisionElement(
            index=1,
            label="gear icon, top right",
            bounds=ElementBounds(x=900, y=100, width=80, height=100),
        )
    ]

    result = vision.render_vision_scan_result(elements, language="en")

    assert '{"vision_index": <i>}' in result
    assert '[1] "gear icon, top right"' in result


def test_empty_scan_result_does_not_invite_invention():
    result = vision.render_vision_scan_result([], language="en")
    assert "no visual subjects" in result.lower()
    assert "Do not invent" in result


# ---------------------------------------------------------------------------
# Target.vision_index 反解
# ---------------------------------------------------------------------------


def _state_with(elements, fingerprint: str | None):
    return SimpleNamespace(
        vision_elements=elements,
        vision_scan_fingerprint=fingerprint,
        latest_screen_elements=None,
        latest_ocr_results=None,
    )


def test_vision_index_resolves_to_bounds_and_label():
    element = vision.VisionElement(
        index=1,
        label="gear icon, top right",
        bounds=ElementBounds(x=10, y=20, width=30, height=40),
    )
    state = _state_with([element], scan_cache_fingerprint(None, None))

    target, error = resolve_target(state, Target(vision_index=1))  # type: ignore[arg-type]

    assert error is None
    assert target.bounds == element.bounds
    assert target.text == "gear icon, top right"


def test_vision_index_without_a_scan_is_rejected():
    state = _state_with([], None)

    _target, error = resolve_target(state, Target(vision_index=1))  # type: ignore[arg-type]

    assert error is not None
    assert "call scan_visual_subjects first" in error


def test_vision_index_out_of_range_is_rejected():
    element = vision.VisionElement(
        index=1, label="only one", bounds=ElementBounds(x=0, y=0, width=10, height=10)
    )
    state = _state_with([element], scan_cache_fingerprint(None, None))

    _target, error = resolve_target(state, Target(vision_index=2))  # type: ignore[arg-type]

    assert error is not None
    assert "out of range" in error
    assert "1..1" in error


def test_a_stale_visual_list_is_rejected():
    element = vision.VisionElement(
        index=1, label="gear", bounds=ElementBounds(x=0, y=0, width=10, height=10)
    )
    state = _state_with([element], "a-fingerprint-from-another-screen")

    _target, error = resolve_target(state, Target(vision_index=1))  # type: ignore[arg-type]

    assert error is not None
    assert "previous screen" in error
    assert "scan_visual_subjects again" in error


def test_vision_bounds_bypass_the_iou_gate():
    """视觉框不参与 IoU 校验：它指向的可能就是层级树里没有的东西。"""
    target = Target(
        vision_index=1,
        bounds=ElementBounds(x=5, y=5, width=10, height=10),
    )
    elements = [{"bounds": {"x": 500, "y": 500, "width": 100, "height": 100}}]

    ok, reason = coordinates_look_like_hierarchy_derived_from_elements(target, elements)

    assert ok, reason


# ---------------------------------------------------------------------------
# 提示词 / 工具集契约
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", ["executor.md", "executor-zh.md"])
def test_prompts_teach_the_vision_index_contract(filename):
    """提示词里必须教 `vision_index`，并且不再有旧的定位失败回灌宏。"""
    text = (_PROMPTS / filename).read_text(encoding="utf-8")
    assert "vision_index" in text
    assert "macro vision_subjects_message(" in text
    assert "vision_failure_message" not in text


def _wrappers_ctx(vision_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        perception=SimpleNamespace(vision=vision_enabled),
        video_recording_enabled=False,
    )


def test_scan_tool_is_only_offered_when_the_switch_is_on():
    from minitap.mobile_use.tools.index import VISION_WRAPPERS, executor_wrappers

    assert VISION_WRAPPERS[0] in executor_wrappers(_wrappers_ctx(True))  # type: ignore[arg-type]
    assert VISION_WRAPPERS[0] not in executor_wrappers(_wrappers_ctx(False))  # type: ignore[arg-type]


def test_scan_tool_dispatches_without_any_target_description():
    """模型只能喊「扫一下」，不能带「找什么」—— dispatcher 必须接受空参数。"""
    from minitap.mobile_use.agents.cortex.dispatcher import dispatch_decisions_with_reason
    from minitap.mobile_use.tools.index import VISION_WRAPPERS, get_tools_from_wrappers

    tools = get_tools_from_wrappers(_wrappers_ctx(True), VISION_WRAPPERS)  # type: ignore[arg-type]

    calls, reason = dispatch_decisions_with_reason(
        '[{"action": "scan_visual_subjects"}]',
        "the element table does not show the icon I need",
        tools,
    )

    assert reason is None
    assert calls is not None
    assert calls[0]["name"] == "scan_visual_subjects"
    assert calls[0]["args"]["agent_thought"] == "the element table does not show the icon I need"
