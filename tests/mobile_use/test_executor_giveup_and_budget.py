"""「做不到」的出口 + 「没有进展」的检测 + 预算耗尽的优雅收尾。

三个问题，一个根因（2026-10-26 真机实测，mobile-manager execution 880）
-------------------------------------------------------------------------

任务「淘宝红包签到」跑了 **45 分钟 / 198 次 LLM 调用 / 409 万 input token**
之后崩在 ``GraphRecursionError: Recursion limit of 752 reached``。

崩之前它卡在「去头条刷热点领现金(0/1)」这一条上重复了 **70 步**：
打开头条 → 等 → 滑 → 返回 → 回淘宝核对 → 又是 0/1 → 再来一遍。

为什么会这样 —— 三个洞叠在一起：

1. **「做不到」在契约里不可表达。** executor 唯一能结束子目标的方式是
   ``done: true``，而 ``done`` 的语义是「目标达成」，
   ``determine_outcome`` 把它**无条件**映射成 ``completed``
   ⇒ 模型「做不到」时只剩继续试 ⇒ 原地打转。
2. **没人说「你没有进展」。** 那些动作**全都执行成功**，
   失败账本里一条事实都没有 ⇒ 熔断不触发。
3. **撞上限是崩，不是收尾。** 撞 ``recursion_limit`` 只留下一句
   「递归上限到了」—— 没有一句人话说清跑到哪了、完成了什么。

这个文件把三个洞分别钉住。
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "minitap"))

import pytest  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from minitap.mobile_use.agents.executor.executor import ExecutorNode  # noqa: E402
from minitap.mobile_use.agents.executor.progress import (  # noqa: E402
    MAX_NUDGES_PER_SUBGOAL,
    action_fingerprint,
    detect_stall,
    screen_state_fingerprint,
    should_nudge,
)
from minitap.mobile_use.agents.executor.prompt import (  # noqa: E402
    build_executor_messages,
    render_stall_notice,
)
from minitap.mobile_use.agents.executor.types import (  # noqa: E402
    ExecutorOutput,
    determine_outcome,
)
from minitap.mobile_use.agents.planner.types import Subgoal, SubgoalStatus  # noqa: E402
from minitap.mobile_use.agents.reporter.reporter import (  # noqa: E402
    build_executor_report,
    determine_outcome_for,
    failure_reason_for,
)
from minitap.mobile_use.config import LLM, LLMConfig, LLMConfigUtils, LLMWithFallback  # noqa: E402
from minitap.mobile_use.context import (  # noqa: E402
    DeviceContext,
    DevicePlatform,
    MobileUseContext,
)
from minitap.mobile_use.graph.state import State  # noqa: E402
from minitap.mobile_use.prompt_language import PromptLanguage  # noqa: E402
from minitap.mobile_use.sdk.budget import (  # noqa: E402
    BudgetExhaustedError,
    build_budget_report,
)
from minitap.mobile_use.utils.screen_elements import ScreenElement  # noqa: E402
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds  # noqa: E402


def _ctx() -> MobileUseContext:
    def llm() -> LLMWithFallback:
        return LLMWithFallback(
            provider="openai", model="gpt-5", fallback=LLM(provider="openai", model="o4-mini")
        )

    return MobileUseContext(
        trace_id="test",
        device=DeviceContext(
            host_platform="WINDOWS",
            mobile_platform=DevicePlatform.ANDROID,
            device_id="test",
            device_width=1220,
            device_height=2712,
        ),
        llm_config=LLMConfig(
            planner=llm(),
            orchestrator=llm(),
            contextor=llm(),
            cortex=llm(),
            executor=llm(),
            utils=LLMConfigUtils(outputter=llm(), hopper=llm()),
        ),
    )


def _elements(n: int = 3, *, tag: str = "row") -> list[ScreenElement]:
    return [
        ScreenElement(
            index=i,
            kind="ui",
            label=f"{tag} {i}",
            resource_id=f"com.app:id/{tag}{i}",
            bounds=ElementBounds(x=0, y=0, width=100, height=50),
            flags="C",
        )
        for i in range(1, n + 1)
    ]


def _state(**overrides) -> State:
    base = {
        "initial_goal": "read the wifi name",
        "subgoal_plan": [],
        "latest_ui_hierarchy": None,
        "latest_ocr_results": None,
        "latest_screen_elements": None,
        "focused_app_info": None,
        "device_date": None,
        "complete_subgoals_by_ids": [],
        "agents_thoughts": [],
        "executor_messages": [],
        "scratchpad": {},
    }
    base.update(overrides)
    return State(**base)


# ---------------------------------------------------------------------------
# 洞 1 —— 「做不到」必须能表达
# ---------------------------------------------------------------------------


def test_give_up_is_never_completed():
    """模型说「做不到」⇒ outcome 必须是 ``failed``，**不能**是 ``completed``。

    判据：``done=True`` + ``gave_up=True`` ⇒ ``failed``。
    （``done=True`` 单独出现时仍然是 ``completed`` —— 那是「做成了」。）
    """
    assert determine_outcome(done=True, gave_up=True) == "failed"
    assert determine_outcome(done=False, gave_up=True) == "failed"
    # 「做成了」的老路不变
    assert determine_outcome(done=True) == "completed"


def test_reporter_outcome_is_failed_when_executor_gave_up():
    """Reporter 是**代码**判 outcome 的（不经 LLM）—— 放弃必须落在那里。

    状态里同时有 ``executor_done=True``（收尾）和 ``executor_failed=True``
    （做不到）⇒ outcome 必须是 ``failed``。
    """
    state = _state(executor_done=True, executor_failed=True)
    assert determine_outcome_for(state) == "failed"

    # 对照：没有放弃标记时，done=True 仍然是「完成」
    assert determine_outcome_for(_state(executor_done=True)) == "completed"


def test_failure_reason_is_copied_verbatim():
    """「做不到的原因」是**事实**：模型说的那句，原样搬上去，不改写。"""
    state = _state(
        executor_done=True,
        executor_failed=True,
        executor_messages=[
            AIMessage(content="tried browsing the hot list three times, the counter stayed 0/1"),
        ],
    )
    assert failure_reason_for(state) == (
        "tried browsing the hot list three times, the counter stayed 0/1"
    )
    # 不是「放弃」就没有这个字段
    assert failure_reason_for(_state(executor_done=True)) is None


@pytest.mark.asyncio
async def test_executor_node_writes_the_gave_up_flag(monkeypatch):
    """模型说「做不到」⇒ 节点写 ``executor_done`` + ``executor_failed``，停止动作。

    ⚠️ **不发动作**是硬要求：模型说了做不到还继续点，就是原地打转的开始。
    """

    async def _decide(self, *, state, goal, stall_notice: str = ""):
        return ExecutorOutput(
            actions=[],
            agent_thought="the counter never moves",
            failed=True,
            failed_reason="the counter stayed 0/1 after three different routes",
        )

    monkeypatch.setattr(ExecutorNode, "_decide", _decide)

    state = _state(executor_goal="complete every doable task")
    update = await ExecutorNode(_ctx())(state)

    assert update["executor_done"] is True
    assert update["executor_failed"] is True
    messages = update["executor_messages"]
    assert any(
        "counter stayed 0/1" in (m.content or "") for m in messages if isinstance(m, AIMessage)
    )


@pytest.mark.asyncio
async def test_give_up_flows_into_the_report(monkeypatch):
    """端到端一格：executor 说「做不到」⇒ 汇报的 outcome 是 ``failed``。

    这是**接线**的测试：executor 节点 → 汇报构建（``build_executor_report``）。
    中间任何一环漏了 ``executor_failed``，汇报就会变成 ``completed``
    （把「做不到」说成「做成了」—— 那是最坏的一种错）。
    """
    async def _decide(self, *, state, goal, stall_notice: str = ""):
        return ExecutorOutput(
            actions=[],
            agent_thought="the counter never moves",
            failed=True,
            failed_reason="tried 3 routes, counter stayed 0/1",
        )

    monkeypatch.setattr(ExecutorNode, "_decide", _decide)

    state = _state(executor_goal="complete every doable task")
    update = await ExecutorNode(_ctx())(state)

    merged = state.model_dump()
    merged.update(update)
    report = build_executor_report(State(**merged), _ctx())
    assert report.outcome == "failed", "「做不到」不能被判成 completed"
    assert report.failure_reason == "tried 3 routes, counter stayed 0/1"
    assert "做不到" in report.summary


@pytest.mark.asyncio
async def test_cortex_clears_the_give_up_flag_and_progress_when_issuing():
    """换目标时，上一次的「做不到」标记和进展指纹都要清。

    留着的话，下一个子目标的汇报会**继承**上一次的 failed ⇒ 一个子目标
    说做不到，后面每一个都被连坐。
    """
    from minitap.mobile_use.agents.cortex.cortex import CortexNode
    from minitap.mobile_use.agents.cortex.types import CortexOutput

    class _L:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, messages):
            return CortexOutput(
                next_subgoal="do the next thing",
                completion_reason=None,
                subgoal_completed=False,
            )

    import minitap.mobile_use.agents.cortex.cortex as cortex_mod

    monkey = pytest.MonkeyPatch()
    monkey.setattr(cortex_mod, "get_llm", lambda **kwargs: _L())
    try:
        state = _state(
            executor_failed=True,
            executor_progress=[{"action": "tap", "screen": "s", "nudged": False}],
            subgoal_plan=[
                Subgoal(
                    id="a",
                    description="first",
                    status=SubgoalStatus.PENDING,
                )
            ],
        )
        update = await CortexNode(_ctx())(state)
    finally:
        monkey.undo()

    assert update["executor_failed"] is False
    assert update["executor_progress"] == []


# ---------------------------------------------------------------------------
# 洞 2 —— 「没有进展」要变成看得见的事实
# ---------------------------------------------------------------------------


def test_screen_state_fingerprint_ignores_changing_numbers():
    """计数器变了**不算**换页 —— 否则「回到了同一页」永远认不出来。"""
    a = _elements(3, tag="row")
    b = _elements(3, tag="row")
    assert screen_state_fingerprint(a) == screen_state_fingerprint(b)

    different = _elements(3, tag="other")
    assert screen_state_fingerprint(a) != screen_state_fingerprint(different)


def test_action_fingerprint_normalizes_the_element_index():
    """元素表每屏重编号 ⇒ 同一个按钮在两屏上是两个 index。

    留着 index，「重复点同一个按钮」就永远认不出来。
    """
    a = action_fingerprint({"action": "tap", "target": {"index": 7}})
    b = action_fingerprint({"action": "tap", "target": {"index": 23}})
    assert a == b == "tap"

    # 但「打开哪个 app」是身份的一部分，必须留着
    assert (
        action_fingerprint({"action": "launch_app", "app_name": "今日头条"})
        == "launch_app:app_name=今日头条"
    )


def test_detect_stall_flags_a_repeated_screen():
    """连续回到同一屏 ⇒ 报「没有进展」。"""
    entries = [{"action": f"tap:{i}", "screen": "same"} for i in range(8)]
    signal = detect_stall(entries)
    assert signal is not None
    assert signal.kind == "screen"
    assert signal.repeats >= 3


def test_detect_stall_flags_a_repeated_action():
    """同一个动作重复 ⇒ 报「没有进展」。"""
    entries = [{"action": "press_key:key=Back", "screen": f"s{i}"} for i in range(8)]
    signal = detect_stall(entries)
    assert signal is not None
    assert signal.kind == "action"


def test_detect_stall_is_quiet_while_progressing():
    """真在推进时**不能**报 —— 误报会让模型乱改策略。"""
    entries = [
        {"action": f"tap:{i}", "screen": f"screen-{i}"} for i in range(8)
    ]
    assert detect_stall(entries) is None


def test_detect_stall_needs_a_full_window():
    """步数不够时不下结论。"""
    assert detect_stall([{"action": "tap", "screen": "s"}] * 3) is None


def test_nudge_is_throttled():
    """提醒是信息，不是限制 —— 但也不能每步都念叨。"""
    entries = [{"action": "tap", "screen": "same", "nudged": False} for _ in range(8)]
    signal = detect_stall(entries)
    assert signal is not None
    assert should_nudge(signal) is True

    # 已经提醒过 MAX 次 ⇒ 不再提醒
    entries = [
        {"action": "tap", "screen": "same", "nudged": True} for _ in range(8)
    ]
    signal = detect_stall(entries)
    assert signal is not None
    assert signal.nudges_sent == 8
    assert should_nudge(signal) is False
    assert MAX_NUDGES_PER_SUBGOAL >= 1


def test_stall_notice_tells_the_model_it_may_give_up():
    """提醒必须**教**模型怎么退出 —— 「没有进展」本身没有用。

    提醒里必须出现 ``failed`` 这个出口，否则模型还是只会继续试。
    """
    notice = render_stall_notice(
        kind="screen",
        repeats=4,
        window=8,
        fingerprint="abc123",
        vision_enabled=True,
        language=PromptLanguage.EN,
    )
    assert "failed" in notice
    assert "4 times" in notice


def test_stall_notice_is_localized():
    """中英各自成句 —— 描述句不能再硬编码在 Python 里（英文 prompt 会混中文）。"""
    kwargs = {
        "kind": "screen",
        "repeats": 4,
        "window": 8,
        "fingerprint": "abc123",
        "vision_enabled": True,
    }
    en = render_stall_notice(**kwargs, language=PromptLanguage.EN)
    zh = render_stall_notice(**kwargs, language=PromptLanguage.ZH)

    assert "same page" in en
    assert not any("\u4e00" <= ch <= "\u9fff" for ch in en), f"英文块里混进了中文：{en}"
    assert "同一个页面" in zh


def test_stall_notice_does_not_ask_for_vision_when_vision_is_off():
    """视觉关掉时 ``scan_visual_subjects`` 根本没注册，不能教模型去调它。"""
    notice = render_stall_notice(
        kind="action",
        repeats=3,
        window=8,
        fingerprint="tap:target",
        vision_enabled=False,
        language=PromptLanguage.EN,
    )
    assert "scan_visual_subjects" not in notice
    assert "failed" in notice


def test_stall_notice_is_placed_after_the_local_history():
    """提醒块每轮变 ⇒ 必须排在局部历史**之后**（前缀缓存只覆盖连续前缀）。"""
    messages = build_executor_messages(
        ctx=_ctx(),
        goal="do the thing",
        screen_elements=_elements(2),
        local_messages=[AIMessage(content="old step")],
        device_state_lines=[],
        stall_notice="[no progress] same page again",
    )
    texts = [str(m.content) for m in messages]
    history_at = next(i for i, t in enumerate(texts) if "old step" in t)
    # ⚠️ 用「提醒的正文」当标记，不能用 "no progress" ——
    # system prompt 里本来就有一句 "re-checking produces no progress"。
    notice_at = next(i for i, t in enumerate(texts) if "same page again" in t)
    assert notice_at > history_at, "提醒块必须排在局部历史之后"


# ---------------------------------------------------------------------------
# 洞 3 —— 撞上限要收尾，不是崩
# ---------------------------------------------------------------------------


def _subgoal(sid: str, status: SubgoalStatus, description: str = "do a thing") -> Subgoal:
    return Subgoal(id=sid, description=description, status=status)


def test_budget_report_lists_what_was_done():
    """收尾报告必须说清：完成了哪些、停在哪个、跑了多少步。"""
    state = _state(
        subgoal_plan=[
            _subgoal("a", SubgoalStatus.SUCCESS, "进入红包签到页"),
            _subgoal("b", SubgoalStatus.SUCCESS, "罗列赚元宝任务"),
            _subgoal("c", SubgoalStatus.PENDING, "依次完成可做的赚元宝任务"),
        ],
        executor_goal="依次完成可做的赚元宝任务",
        executor_steps=74,
        total_steps=186,
    )
    report = build_budget_report(state, max_steps=80, recursion_limit=752)

    assert "预算用完" in report
    assert "186" in report  # 实际跑了多少动作步
    assert "752" in report  # 上限是多少个图步
    assert "进入红包签到页" in report
    assert "依次完成可做的赚元宝任务" in report
    assert "完成 2 个" in report
    # 最后一句要说清「是预算问题，不是任务做不到」
    assert "预算" in report


def test_budget_report_says_which_subgoal_it_stopped_on():
    state = _state(
        subgoal_plan=[_subgoal("c", SubgoalStatus.PENDING, "最后一个子目标")],
        executor_goal="最后一个子目标",
        executor_steps=74,
        total_steps=186,
    )
    report = build_budget_report(state, max_steps=80, recursion_limit=752)
    assert "74 步" in report


def test_agent_catches_recursion_limit_and_reports_instead_of_crashing():
    """``GraphRecursionError`` 必须被单独接住 ⇒ 收尾报告 + ``BudgetExhaustedError``。

    ⚠️ 仍然是「失败」（目标确实没达成，不能说成完成），
    但失败的原因从「Recursion limit of 752 reached」变成了一份人能读的���告。
    """
    from minitap.mobile_use.sdk.agent import Agent

    src = inspect.getsource(Agent._run_task)
    assert "except GraphRecursionError" in src
    assert "build_budget_report" in src
    assert "BudgetExhaustedError" in src


def test_budget_exhausted_error_carries_the_report():
    err = BudgetExhaustedError("预算耗尽", report="做了 2 个子目标")
    assert err.report == "做了 2 个子目标"
    assert "预算耗尽" in str(err)
