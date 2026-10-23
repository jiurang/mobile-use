"""tap 的熔断与失败记账。

真机实测里点击失败几乎全是同一类：**目标已经不在屏幕上了**（横屏全屏页被
视频播完带走，模型思考 17 秒才动手）。模型看到这种失败会误诊成「索引指错
人了」，然后在同一个目标上反复打转。

所以这里守两件事：
1. 同一个目标连着失败到阈值 → 工具**不再执行**，直接要求换目标；
2. 失败要记进账本、成功要清账 —— 账本是下一轮提示词里「这些目标已经没了」
   的唯一来源。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from minitap.mobile_use.graph.state import State
from minitap.mobile_use.tools.mobile import tap as tap_module
from minitap.mobile_use.tools.types import Target
from minitap.mobile_use.utils.failed_targets import FailedTarget
from minitap.mobile_use.utils.screen_elements import ScreenElement
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds

W, H = 1220, 2712


def make_state(elements: list[ScreenElement] | None = None) -> State:
    return State(
        messages=[],
        initial_goal="g",
        subgoal_plan=[],
        latest_ui_hierarchy=None,
        latest_ocr_results=None,
        focused_app_info=None,
        device_date=None,
        structured_decisions=None,
        complete_subgoals_by_ids=[],
        executor_messages=[],
        cortex_last_thought=None,
        agents_thoughts=[],
        latest_screen_elements=elements,
    )


def element(index: int, resource_id: str = "com.app:id/like", text: str = "点赞") -> ScreenElement:
    return ScreenElement(
        index=index,
        kind="ui",
        label=text,
        text="",
        resource_id=resource_id,
        display_id=resource_id.split("/")[-1],
        class_name="android.widget.ImageView",
        bounds=ElementBounds(x=100, y=200, width=80, height=80),
        flags="C",
    )


@pytest.fixture
def wired(monkeypatch):
    """tap 的外部依赖换成桩：实时 dump 里**没有**目标（模拟「东西已经没了」）。"""
    calls: dict = {"taps": [], "dumps": 0}

    class StubController:
        def __init__(self, ctx):
            pass

        async def tap_at(self, x: int, y: int):
            calls["taps"].append((x, y))
            return SimpleNamespace(error=None)

        async def tap_element(self, resource_id=None, text=None, index=0):
            return SimpleNamespace(error="no element")

    class StubLive:
        def __init__(self, ctx):
            pass

        async def get_ui_elements(self):
            calls["dumps"] += 1
            return []  # 当屏上没有这个 id 了

    async def live_not_found(ctx, resource_id, index=0, hint=None):
        calls["dumps"] += 1
        return None, "the element is no longer on the screen"

    async def fake_validate(ctx, target, min_iou=0.9, ocr_texts=None):
        return True, None

    async def passthrough(self, ctx, update, agent=None):
        return update

    monkeypatch.setattr(tap_module, "UnifiedMobileController", StubController)
    monkeypatch.setattr(tap_module, "resolve_live_bounds_by_id", live_not_found)
    monkeypatch.setattr(tap_module, "coordinates_look_like_hierarchy_derived", fake_validate)
    monkeypatch.setattr(State, "asanitize_update", passthrough)
    return calls


def run_tap(ctx, state: State, target: Target):
    tool = tap_module.get_tap_tool(ctx)
    return asyncio.run(
        tool.coroutine(
            agent_thought="thought",
            target=target,
            tool_call_id="call-1",
            state=state,
        )
    )


@pytest.fixture
def ctx():
    return SimpleNamespace(
        device=SimpleNamespace(device_width=W, device_height=H),
        perception=SimpleNamespace(ocr=True),
    )


def _failed_targets(cmd) -> list[FailedTarget]:
    return cmd.update.get("failed_targets", [])


def test_a_failure_is_recorded_in_the_ledger(ctx, wired):
    state = make_state([element(1)])
    cmd = run_tap(ctx, state, Target(index=1))
    assert not wired["taps"], "目标不在屏上时不该真的点下去"
    streak = _failed_targets(cmd)
    assert len(streak) == 1
    assert streak[0].index == 1
    assert streak[0].count == 1


def test_the_same_target_twice_is_blocked(ctx, wired):
    """连着失败两次后，工具不再执行它 —— 要求模型换目标。"""
    state = make_state([element(1)])

    # 第一次：去当屏找了一次，没找到
    cmd1 = run_tap(ctx, state, Target(index=1))
    assert wired["dumps"] == 1, "第一次失败该去当屏找一次"
    assert wired["taps"] == [], "目标不在屏上时不该真的点下去"

    # 第二次：模型又报了同一个目标（允许重试一次，人也会这么做）
    state.failed_targets = cmd1.update["failed_targets"]
    cmd2 = run_tap(ctx, state, Target(index=1))
    assert wired["dumps"] == 2
    assert _failed_targets(cmd2)[0].count == 2

    # 第三次：熔断 —— 不再白花一次 live dump，也不再点
    state.failed_targets = cmd2.update["failed_targets"]
    cmd3 = run_tap(ctx, state, Target(index=1))
    assert wired["dumps"] == 2, "熔断后不该再白花一次 live dump"
    assert wired["taps"] == [], "熔断后不该真的点下去"
    message = cmd3.update["agents_thoughts"][-1]
    assert "already failed" in message
    assert "re-read" in message


def test_a_success_clears_the_ledger(ctx, monkeypatch):
    """成功即清账：模型换了目标，「连续失败」到此为止。"""
    state = make_state([element(1)])
    state.failed_targets = [
        FailedTarget(key="index=9|id=com.app:id/old", index=9, resource_id="com.app:id/old")
    ]

    class StubController:
        def __init__(self, ctx):
            pass

        async def tap_at(self, x: int, y: int):
            return SimpleNamespace(error=None)

        async def tap_element(self, resource_id=None, text=None, index=0):
            return SimpleNamespace(error="no element")

    async def found(ctx, resource_id, index=0, hint=None):
        return ElementBounds(x=10, y=20, width=30, height=40), ""

    async def fake_validate(ctx, target, min_iou=0.9, ocr_texts=None):
        return True, None

    async def passthrough(self, ctx, update, agent=None):
        return update

    monkeypatch.setattr(tap_module, "UnifiedMobileController", StubController)
    monkeypatch.setattr(tap_module, "resolve_live_bounds_by_id", found)
    monkeypatch.setattr(tap_module, "coordinates_look_like_hierarchy_derived", fake_validate)
    monkeypatch.setattr(State, "asanitize_update", passthrough)

    cmd = run_tap(ctx, state, Target(index=1))
    assert _failed_targets(cmd) == []
    assert "was successful" in cmd.update["agents_thoughts"][-1]
