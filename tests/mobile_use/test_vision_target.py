"""视觉主体扫描（``scan_visual_subjects``）的离线回归测试。

覆盖五件事：

1. 归一化框 → 像素框的换算（越界夹紧、空框拒绝）；
2. 扫描输出 → 编号表的映射（过滤空描述 / 非法框，截断到 20 项，异常降级）；
3. 感知帧：epoch 派生、视觉列表的继承与作废、扫描只读帧里的图；
4. Executor 的 ``Target.vision_index`` 反解（命中、越界、无列表、无帧）；
5. 提示词 / 工具集契约（中英模板、开关、回执不重复整张表）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "minitap"))

import pytest  # noqa: E402

from minitap.mobile_use.perception import frame as screen_frame  # noqa: E402
from minitap.mobile_use.services import vision  # noqa: E402
from minitap.mobile_use.tools.bounds_validation import (  # noqa: E402
    coordinates_look_like_hierarchy_derived_from_elements,
)
from minitap.mobile_use.tools.types import Target  # noqa: E402
from minitap.mobile_use.tools.utils import resolve_target  # noqa: E402
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
        vision.VisionSubject(description=f"icon {i}", x1=0, y1=i, x2=100, y2=i + 10)
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


def test_scan_receipt_tells_the_model_how_to_address_a_subject():
    """回执只报数量，编号表由每轮 prompt 渲染 —— 不能两处都列一遍。"""
    result = vision.render_scan_receipt(1, language="en")

    assert '{"vision_index": <i>}' in result
    assert "1 subject(s)" in result
    assert "[1]" not in result


def test_empty_scan_result_does_not_invite_invention():
    result = vision.render_scan_receipt(0, language="en")
    assert "no visual subjects" in result.lower()
    assert "Do not invent" in result


# ---------------------------------------------------------------------------
# 感知帧：一帧一图、列表随帧继承 / 作废
# ---------------------------------------------------------------------------


def _frame(
    *,
    epoch: str = "screen-a",
    elements=None,
    vision_elements=None,
    screenshot: str = "frame-image",
    screen_state: str | None = "on",
) -> screen_frame.ScreenFrame:
    return screen_frame.ScreenFrame(
        epoch=epoch,
        screenshot_base64=screenshot,
        width=1000,
        height=2000,
        screen_elements=elements or [],
        screen_state=screen_state,
        vision_elements=vision_elements or [],
    )


def _ctx_with(frame_obj) -> SimpleNamespace:
    return SimpleNamespace(screen_frame=frame_obj)


def test_same_screen_keeps_the_visual_list():
    element = screen_frame.VisionElement(
        index=1, label="gear", bounds=ElementBounds(x=0, y=0, width=10, height=10)
    )
    ctx = _ctx_with(_frame(vision_elements=[element]))

    screen_frame.publish_frame(ctx, _frame(epoch="screen-a"))

    assert ctx.screen_frame.vision_elements == [element]


def test_a_new_screen_drops_the_visual_list():
    element = screen_frame.VisionElement(
        index=1, label="gear", bounds=ElementBounds(x=0, y=0, width=10, height=10)
    )
    ctx = _ctx_with(_frame(vision_elements=[element]))

    screen_frame.publish_frame(ctx, _frame(epoch="screen-b"))

    assert ctx.screen_frame.vision_elements == []


def test_an_empty_text_channel_makes_every_frame_distinct():
    """两条文字通道都空时不能把上一屏的视觉列表一直继承下去。"""
    first = screen_frame.frame_epoch(None, None)
    second = screen_frame.frame_epoch(None, None)

    assert first.startswith("empty@")
    assert second.startswith("empty@")
    assert first != second


@pytest.mark.asyncio
async def test_scan_reads_the_frame_image_instead_of_taking_a_new_one(monkeypatch):
    frame_obj = _frame(screenshot="the-frame-image")
    captured: dict[str, object] = {}

    async def _fake_scan(ctx, screenshot_base64, **kwargs):
        captured["image"] = screenshot_base64
        captured["size"] = (kwargs.get("screen_width"), kwargs.get("screen_height"))
        return [], ""

    monkeypatch.setattr(vision, "scan_screen_subjects", _fake_scan)

    elements, reason = await vision.scan_frame(SimpleNamespace(), frame_obj)

    assert reason == ""
    assert elements == []
    assert captured["image"] == "the-frame-image"
    assert captured["size"] == (1000, 2000)


@pytest.mark.asyncio
async def test_scan_without_a_frame_degrades_with_a_reason():
    elements, reason = await vision.scan_frame(SimpleNamespace(), None)

    assert elements == []
    assert "frame is unavailable" in reason


@pytest.mark.asyncio
async def test_scan_on_a_screen_that_turned_off_is_refused(monkeypatch):
    frame_obj = _frame(screen_state="off")

    async def _fake_scan(ctx, screenshot_base64, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("vision model must not be called on a black screenshot")

    monkeypatch.setattr(vision, "scan_screen_subjects", _fake_scan)

    elements, reason = await vision.scan_frame(SimpleNamespace(), frame_obj)

    assert elements == []
    assert "screen is off" in reason


@pytest.mark.parametrize("language", ["en", "zh"])
def test_scan_prompt_defines_granularity_and_box_rules(language):
    """提示词必须讲清「最小有意义单位」和框的尺度，而不只是「列出图标」。"""
    text = vision.scan_system_prompt(SimpleNamespace(prompt_language=language))

    assert "0-1000" in text
    assert "30-120" in text or "30~120" in text
    assert "smallest" in text.lower() or "最小有意义单位" in text
    assert "container" in text.lower() or "容器" in text
    assert str(vision.MAX_SUBJECTS) in text


def test_scan_prompt_reads_the_chinese_variant():
    text = vision.scan_system_prompt(SimpleNamespace(prompt_language="zh"))

    assert "视觉主体" in text
    assert "归一化" in text


def test_scan_prompt_limit_follows_the_constant(monkeypatch):
    monkeypatch.setattr(vision, "MAX_SUBJECTS", 7)

    text = vision.scan_system_prompt(SimpleNamespace(prompt_language="en"))

    assert "at most 7" in text


# ---------------------------------------------------------------------------
# Target.vision_index 反解
# ---------------------------------------------------------------------------


def _state():
    return SimpleNamespace(latest_screen_elements=None, latest_ocr_results=None)


def test_vision_index_resolves_to_bounds_and_label():
    element = screen_frame.VisionElement(
        index=1,
        label="gear icon, top right",
        bounds=ElementBounds(x=10, y=20, width=30, height=40),
    )
    frame_obj = _frame(vision_elements=[element])

    target, error = resolve_target(_state(), Target(vision_index=1), frame=frame_obj)

    assert error is None
    assert target.bounds == element.bounds
    assert target.text == "gear icon, top right"


def test_vision_index_without_a_scan_is_rejected():
    frame_obj = _frame()

    _target, error = resolve_target(_state(), Target(vision_index=1), frame=frame_obj)

    assert error is not None
    assert "call scan_visual_subjects first" in error


def test_vision_index_out_of_range_is_rejected():
    element = screen_frame.VisionElement(
        index=1, label="only one", bounds=ElementBounds(x=0, y=0, width=10, height=10)
    )
    frame_obj = _frame(vision_elements=[element])

    _target, error = resolve_target(_state(), Target(vision_index=2), frame=frame_obj)

    assert error is not None
    assert "out of range" in error
    assert "1..1" in error


def test_vision_index_without_a_frame_is_rejected():
    _target, error = resolve_target(_state(), Target(vision_index=1))

    assert error is not None
    assert "frame is unavailable" in error


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


@pytest.mark.parametrize("language", ["en", "zh"])
def test_prompts_teach_the_vision_index_contract(language):
    """提示词里必须教 `vision_index`，并且不再有旧的定位失败回灌宏。"""
    message = (_PROMPTS / f"prompts/{language}/messages/per-turn/vision-subjects.md").read_text(
        encoding="utf-8"
    )
    assert "vision_index" in message
    assert "{{ rows }}" in message
    assert "vision_failure_message" not in message


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
