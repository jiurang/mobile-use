"""tap 能不能直接把 OCR 的 `[i]` 当成工具参数用。

`resolve_ocr_index` 的编号规则在 `test_ocr_index_contract.py` 里守着。这里守
的是**接线** —— 真正容易坏、又不会自己报错的那几根线：

1. `state.latest_ocr_results` 在 executor 执行时还在不在（cortex 曾经把它清空，
   清空之后索引就无从还原，而 tap 不会报错，只是这个字段永远失效）。
2. 还原出来的 bounds 有没有真的走到 `tap_at`（而不是被静默丢掉）。
3. 索引越界时是不是报错，而不是悄悄点别的地方。

不接真设备：controller 和坐标校验都被替换掉，只验 tap 自己的逻辑。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from minitap.mobile_use.graph.state import State
from minitap.mobile_use.services.ocr import OcrText
from minitap.mobile_use.tools.mobile import tap as tap_module
from minitap.mobile_use.tools.types import Target
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds

W, H = 1220, 2712


def ocr(text: str, x: int, y: int, w: int = 200, h: int = 60) -> OcrText:
    return OcrText(text=text, score=0.95, bounds=ElementBounds(x=x, y=y, width=w, height=h))


def make_state(ocr_results: list[OcrText] | None) -> State:
    return State(
        initial_goal="g",
        subgoal_plan=[],
        latest_ui_hierarchy=None,
        latest_ocr_results=ocr_results,
        focused_app_info=None,
        device_date=None,
        complete_subgoals_by_ids=[],
        executor_messages=[],
        agents_thoughts=[],
    )


@pytest.fixture
def wired(monkeypatch):
    """把 tap 的外部依赖换成可观测的桩，返回调用记录。"""
    calls: dict = {"taps": [], "validated": [], "elements": []}

    class StubController:
        def __init__(self, ctx):
            pass

        async def tap_at(self, x: int, y: int):
            calls["taps"].append((x, y))
            return SimpleNamespace(error=None)

        async def tap_element(self, resource_id=None, text=None, index=0):
            calls["elements"].append((resource_id, text, index))
            return SimpleNamespace(error="no element")

    async def fake_validate(ctx, target, min_iou=0.9, ocr_texts=None):
        calls["validated"].append({"bounds": target.bounds, "ocr_texts": ocr_texts})
        return True, None

    async def passthrough(self, ctx, update, agent=None):
        return update

    monkeypatch.setattr(tap_module, "UnifiedMobileController", StubController)
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


class TestOcrIndexResolvesToACoordinateTap:
    def test_index_two_taps_the_second_ocr_box(self, ctx, wired):
        results = [ocr("A", 100, 200), ocr("B", 700, 2000), ocr("C", 300, 900)]
        run_tap(ctx, make_state(results), Target(ocr_index=2))
        assert wired["taps"] == [(800, 2030)]  # B 的中心

    def test_index_one_is_the_first_line(self, ctx, wired):
        results = [ocr("A", 100, 200), ocr("B", 700, 2000)]
        run_tap(ctx, make_state(results), Target(ocr_index=1))
        assert wired["taps"] == [(200, 230)]

    def test_last_index_is_reachable(self, ctx, wired):
        results = [ocr(f"t{i}", i * 50, i * 50) for i in range(1, 11)]
        run_tap(ctx, make_state(results), Target(ocr_index=10))
        assert len(wired["taps"]) == 1

    def test_index_alone_is_a_valid_selector(self, ctx, wired):
        """只给 ocr_index 也算有效定位 —— 不需要再补 text/bounds。"""
        results = [ocr("A", 100, 200)]
        result = run_tap(ctx, make_state(results), Target(ocr_index=1))
        assert wired["taps"] == [(200, 230)]
        assert "successful" in str(result.update["executor_messages"][0].content)

    def test_state_ocr_is_handed_to_the_bounds_validator(self, ctx, wired):
        """校验器要用**同一份** results，否则会现截现识拿另一份数据判合法坐标。"""
        results = [ocr("A", 100, 200), ocr("B", 700, 2000)]
        run_tap(ctx, make_state(results), Target(ocr_index=2))
        assert wired["validated"]
        # 用 == 而不是 is：pydantic 构造 State 时会复制 list，但内容必须一致
        assert wired["validated"][0]["ocr_texts"] == results

    def test_result_message_echoes_the_resolved_text(self, ctx, wired):
        """回显命中的文本，模型能看见索引到底点到了什么。"""
        results = [ocr("A", 100, 200), ocr("全屏观看", 700, 2000)]
        result = run_tap(ctx, make_state(results), Target(ocr_index=2))
        assert "全屏观看" in str(result.update["executor_messages"][0].content)

    def test_explicit_bounds_still_work_alongside(self, ctx, wired):
        """给了 bounds 就用 bounds，索引不参与 —— 不改变原有行为。"""
        results = [ocr("A", 100, 200), ocr("B", 700, 2000)]
        run_tap(
            ctx,
            make_state(results),
            Target(bounds=ElementBounds(x=50, y=60, width=20, height=20)),
        )
        assert wired["taps"] == [(60, 70)]


class TestBadIndexIsReportedNotSwallowed:
    def test_out_of_range_reports_the_valid_range(self, ctx, wired):
        results = [ocr("A", 10, 10), ocr("B", 20, 20)]
        result = run_tap(ctx, make_state(results), Target(ocr_index=9))
        msg = result.update["executor_messages"][0]
        content = str(msg.content)
        assert "ocr_index=9" in content
        assert "1..2" in content
        assert msg.status == "error"

    def test_zero_is_rejected_not_clamped(self, ctx, wired):
        """0 是 1-based 用错了，绝不能当成第一条 —— 那会静默点错一格。"""
        results = [ocr("A", 100, 200), ocr("B", 700, 2000)]
        result = run_tap(ctx, make_state(results), Target(ocr_index=0))
        assert wired["taps"] == []
        assert "ocr_index=0" in str(result.update["executor_messages"][0].content)

    def test_missing_ocr_results_says_so(self, ctx, wired):
        result = run_tap(ctx, make_state(None), Target(ocr_index=2))
        assert "no OCR results available" in str(result.update["executor_messages"][0].content)
        assert wired["taps"] == []

    def test_bad_index_still_falls_back_to_text(self, ctx, wired):
        """索引废了不该让整次点击报废 —— 链上还有别的定位器。"""
        results = [ocr("A", 10, 10)]
        run_tap(ctx, make_state(results), Target(ocr_index=99, text="提交"))
        assert wired["elements"] == [(None, "提交", 0)]

    def test_attempts_are_recorded_for_debugging(self, ctx, wired):
        results = [ocr("A", 10, 10)]
        result = run_tap(ctx, make_state(results), Target(ocr_index=99))
        attempts = result.update["executor_messages"][0].additional_kwargs["attempts"]
        assert any(a["selector"] == "ocr_index=99" for a in attempts)
