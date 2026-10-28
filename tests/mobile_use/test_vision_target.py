"""视觉定位（``Target.vision``）的离线回归测试。

覆盖三件事：

1. 归一化框 → 像素框的换算（越界夹紧、空框拒绝）；
2. Executor 派发前的解析：成功改写 target（索引让位）、失败原样返回并带原因；
3. 视觉框不走 IoU 门限 —— 它和 OCR 框一样是「量出来的」，不是从截图目测的。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "minitap"))

import pytest  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from minitap.mobile_use.agents.executor.prompt import (  # noqa: E402
    rewrite_history_tool_calls,
)
from minitap.mobile_use.services import vision  # noqa: E402
from minitap.mobile_use.tools.bounds_validation import (  # noqa: E402
    coordinates_look_like_hierarchy_derived_from_elements,
)
from minitap.mobile_use.tools.types import Target  # noqa: E402
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


@pytest.mark.asyncio
async def test_vision_target_is_rewritten_to_bounds(monkeypatch):
    """成功时：框写进 bounds，`index` / `ocr_index` 必须让位。

    留着旧索引的话，工具侧的 ``resolve_target`` 会用元素表那一行覆盖视觉框，
    等于白定位一次。
    """

    async def fake_locate(ctx, description):
        assert description == "右上角的齿轮图标"
        return ElementBounds(x=10, y=20, width=30, height=40), ""

    monkeypatch.setattr(vision, "locate_target_by_vision", fake_locate)

    actions = [{"action": "tap", "target": {"index": 7, "vision": "右上角的齿轮图标"}}]
    resolved, reason = await vision.resolve_vision_targets(object(), actions)  # type: ignore[arg-type]

    assert reason == ""
    target = resolved[0]["target"]
    assert target["bounds"] == {"x": 10, "y": 20, "width": 30, "height": 40}
    # 旧的定位字段全部让位：工具在视觉框之前会先用它们
    assert "index" not in target and "ocr_index" not in target
    # 出处要留着：坐标校验靠它放行，失败账本靠它认身份
    assert target["vision"] == "右上角的齿轮图标"
    # 输入动作不能被就地改写
    assert actions[0]["target"] == {"index": 7, "vision": "右上角的齿轮图标"}


@pytest.mark.asyncio
async def test_vision_failure_returns_the_actions_untouched(monkeypatch):
    async def fake_locate(ctx, description):
        return None, "nothing matching '齿轮' is visible on the screen"

    monkeypatch.setattr(vision, "locate_target_by_vision", fake_locate)

    actions = [{"action": "tap", "target": {"vision": "齿轮"}}]
    resolved, reason = await vision.resolve_vision_targets(object(), actions)  # type: ignore[arg-type]

    assert reason == "nothing matching '齿轮' is visible on the screen"
    assert resolved == actions


@pytest.mark.asyncio
async def test_actions_without_vision_are_untouched(monkeypatch):
    """没写 vision 的动作根本不进视觉通道（不能平白多一次截图 + LLM 调用）。"""

    async def explode(ctx, description):  # pragma: no cover - 被调用即失败
        raise AssertionError("no vision call expected")

    monkeypatch.setattr(vision, "locate_target_by_vision", explode)

    actions = [{"action": "tap", "target": {"index": 3}}, {"action": "back"}]
    resolved, reason = await vision.resolve_vision_targets(object(), actions)  # type: ignore[arg-type]

    assert reason == ""
    assert resolved == actions


def test_vision_bounds_bypass_the_iou_gate():
    """视觉框不参与 IoU 校验：它指向的可能就是层级树里没有的东西。"""
    target = Target(
        vision="右上角的齿轮图标",
        bounds=ElementBounds(x=5, y=5, width=10, height=10),
    )
    elements = [{"bounds": {"x": 500, "y": 500, "width": 100, "height": 100}}]

    ok, reason = coordinates_look_like_hierarchy_derived_from_elements(target, elements)
    assert ok, reason


def test_target_accepts_a_vision_description():
    assert Target(vision="gear").vision == "gear"


def test_history_hides_the_resolved_bounds_for_vision_targets():
    """历史里只留模型自己写的 vision 描述，不留框架填的 bounds。

    真机实测（2026-10-28）：bounds 留在历史里，模型把它读成「我给过坐标」，
    连着三轮怀疑视觉没生效、反复重试。
    """
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "tap",
                "args": {
                    "target": {
                        "vision": "gear",
                        "bounds": {"x": 1, "y": 2, "width": 3, "height": 4},
                    },
                    "agent_thought": "tapping the gear",
                },
                "id": "c1",
            }
        ],
    )

    rewritten = rewrite_history_tool_calls([message])
    action = rewritten[0].tool_calls[0]["args"]["actions"][0]

    assert action["target"] == {"vision": "gear"}


# ---------------------------------------------------------------------------
# 置信度门限：真机实测里「屏上根本没有的目标」也会被强行给框，必须挡掉
# ---------------------------------------------------------------------------


def _vision_ctx() -> SimpleNamespace:
    return SimpleNamespace(
        device=SimpleNamespace(device_width=1000, device_height=2000),
        trace_id="test",
        prompt_language="en",
    )


class _FakeController:
    def __init__(self, ctx) -> None:
        pass

    async def get_screen_power_state(self) -> str:
        return "on"


async def _fake_capture(ctx) -> str:
    return "raw-screenshot"


async def _fake_invoke(coro, **kwargs):
    return await coro


def _patch_vision_pipeline(monkeypatch, response) -> None:
    class _Runner:
        async def ainvoke(self, messages):
            return response

    monkeypatch.setattr(vision, "UnifiedMobileController", _FakeController)
    monkeypatch.setattr(vision, "capture_screen_base64", _fake_capture)
    monkeypatch.setattr(vision, "compress_screenshot_base64", lambda raw, quality: "img")
    monkeypatch.setattr(vision, "get_structured_llm", lambda **kwargs: _Runner())
    monkeypatch.setattr(vision, "invoke_llm_with_timeout_message", _fake_invoke)


@pytest.mark.asyncio
async def test_low_confidence_box_is_rejected(monkeypatch):
    """相似但不是的东西（置信度低）不能拿去点 —— 视觉框是跳过 IoU 校验的。"""
    response = vision.VisionLocationOutput(found=True, confidence=20, x1=0, y1=0, x2=500, y2=1000)
    _patch_vision_pipeline(monkeypatch, response)

    bounds, reason = await vision.locate_target_by_vision(_vision_ctx(), "蓝色鲸鱼")

    assert bounds is None
    assert "confidence 20/100" in reason


@pytest.mark.asyncio
async def test_confident_box_is_accepted(monkeypatch):
    response = vision.VisionLocationOutput(found=True, confidence=90, x1=0, y1=0, x2=500, y2=1000)
    _patch_vision_pipeline(monkeypatch, response)

    bounds, reason = await vision.locate_target_by_vision(_vision_ctx(), "齿轮图标")

    assert reason == ""
    assert bounds == ElementBounds(x=0, y=0, width=500, height=2000)


@pytest.mark.asyncio
async def test_not_found_is_rejected(monkeypatch):
    _patch_vision_pipeline(monkeypatch, vision.VisionLocationOutput(found=False, confidence=99))

    bounds, reason = await vision.locate_target_by_vision(_vision_ctx(), "蓝色鲸鱼")

    assert bounds is None
    assert "nothing matching" in reason


@pytest.mark.parametrize("filename", ["executor.md", "executor-zh.md"])
def test_prompts_teach_the_vision_field(filename):
    """提示词里必须能看见 `{"vision": ...}` 和失败回灌宏（中英两份对齐）。"""
    text = (_PROMPTS / filename).read_text(encoding="utf-8")
    assert '{"vision":' in text
    assert "macro vision_failure_message(" in text
