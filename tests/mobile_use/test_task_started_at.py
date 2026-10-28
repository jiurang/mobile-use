"""「任务开始时间」这个提示词块的单测。

它是什么
--------
Executor 和 Cortex 的提示词里，紧挨系统提示词之后有一条单独的消息，写着
这次任务是什么时候开始的。

为什么是**单独一条**，而不是塞进系统提示词的内容里
----------------------------------------------------
**系统提示词必须逐字节稳定** —— 它是整段请求里最大的一块，塞进去一个
每任务都不同的值 ⇒ 它变成「每任务唯一」⇒ 同一 persona 的不同任务之间
不再共享那段前缀缓存。

单独一条就没这个问题：系统提示词一个字不动，时间信息照给。

为什么它**不会**打断缓存
------------------------
任务开始时间**整场运行不变** ⇒ 它和它前面的系统提示词一样，是前缀的
一部分，永远命中。它排在局部历史**之前**也没关系 —— 变的从来不是它。

四条硬约束
----------
1. 时间块在**系统提示词之后**（不是之前，也不是系统提示词里面）；
2. 系统提示词的内容**逐字节不变**（加不加时间块都一样）；
3. 时间未知（``None``）⇒ **不发**这条消息（不能发空块）；
4. 中英文各说各的（模板里写文案，代码只递数据）。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from minitap.mobile_use.agents.cortex.cortex import CortexNode
from minitap.mobile_use.agents.executor.prompt import build_executor_messages
from minitap.mobile_use.config import LLM, LLMConfig, LLMConfigUtils, LLMWithFallback
from minitap.mobile_use.context import DeviceContext, DevicePlatform, MobileUseContext
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.prompt_language import PromptLanguage
from minitap.mobile_use.utils.screen_elements import ScreenElement
from minitap.mobile_use.utils.time import format_task_started_at
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds

TZ = timezone(timedelta(hours=8))
STARTED = datetime(2026, 10, 26, 22, 16, 33, tzinfo=TZ)


def _llm() -> LLMWithFallback:
    return LLMWithFallback(
        provider="openai", model="gpt-5", fallback=LLM(provider="openai", model="o4-mini")
    )


def _ctx(**kw) -> MobileUseContext:
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
            planner=_llm(),
            orchestrator=_llm(),
            contextor=_llm(),
            cortex=_llm(),
            executor=_llm(),
            utils=LLMConfigUtils(outputter=_llm(), hopper=_llm()),
        ),
        **kw,
    )


def _elements(n: int = 3) -> list[ScreenElement]:
    return [
        ScreenElement(
            index=i,
            kind="ui",
            label=f"Row {i}",
            bounds=ElementBounds(x=0, y=0, width=100, height=50),
            flags="C",
        )
        for i in range(1, n + 1)
    ]


def _executor_messages(ctx: MobileUseContext):
    return build_executor_messages(
        ctx=ctx,
        goal="用「传奇」再搜一次",
        screen_elements=_elements(),
        local_messages=[],
        device_state_lines=["Device date: 2026-10-26 22:20:00 CST 2026"],
    )


class TestFormatting:
    def test_it_carries_the_timezone_offset(self):
        """带时区偏移 —— 提示词里同时有设备时间，模型要能判断两者是不是同一个钟。"""
        assert format_task_started_at(STARTED) == "2026-10-26 22:16:33+08:00"

    def test_a_naive_datetime_is_read_as_local_time(self):
        naive = datetime(2026, 10, 26, 22, 16, 33)
        formatted = format_task_started_at(naive)
        assert formatted is not None
        assert formatted.startswith("2026-10-26 22:16:33")

    def test_unknown_time_is_none(self):
        assert format_task_started_at(None) is None


class TestExecutorBlock:
    def test_it_comes_right_after_the_system_prompt(self):
        msgs = _executor_messages(_ctx(task_started_at=STARTED))

        assert msgs[0].type == "system"
        assert "Task started at" in str(msgs[1].content), (
            f"时间块必须紧跟系统提示词，实际第 1 条是 {str(msgs[1].content)[:60]!r}"
        )
        assert "2026-10-26 22:16:33+08:00" in str(msgs[1].content)

    def test_the_goal_still_comes_after_it(self):
        """目标在时间块**之后** —— 目标的措辞是子目标级的，位置不动。"""
        msgs = _executor_messages(_ctx(task_started_at=STARTED))

        goal_idx = next(i for i, m in enumerate(msgs) if "Your goal" in str(m.content))
        assert goal_idx == 2, f"目标的��置变了：{goal_idx}"

    def test_the_system_prompt_is_byte_identical(self):
        """系统提示词**一个字都不许变** —— 它是跨任务共享的那一段缓存。"""
        with_time = _executor_messages(_ctx(task_started_at=STARTED))
        without_time = _executor_messages(_ctx(task_started_at=None))

        assert str(with_time[0].content) == str(without_time[0].content)

    def test_no_block_when_the_time_is_unknown(self):
        msgs = _executor_messages(_ctx(task_started_at=None))

        assert not any("Task started at" in str(m.content) for m in msgs)
        assert msgs[1].type == "human"
        assert "Your goal" in str(msgs[1].content), "没有时间块时，目标的��置不变"

    def test_the_chinese_template_says_it_in_chinese(self):
        msgs = _executor_messages(
            _ctx(task_started_at=STARTED, prompt_language=PromptLanguage.ZH)
        )

        assert "任务开始时间" in str(msgs[1].content)
        assert "2026-10-26 22:16:33+08:00" in str(msgs[1].content)

    def test_it_is_stable_across_rounds(self):
        """同一场运行里，时间块**逐字节不变** ⇒ 它永远是前缀的一部分。"""
        ctx = _ctx(task_started_at=STARTED)
        first = _executor_messages(ctx)
        second = _executor_messages(ctx)

        assert str(first[1].content) == str(second[1].content)


class TestCortexBlock:
    """Cortex 那边：同样紧跟系统提示词，系统提示词同样一字不改。"""

    @staticmethod
    def _capture(ctx: MobileUseContext) -> list:
        """跑一次 cortex 节点，把发给 LLM 的消息截下来。"""
        from minitap.mobile_use.agents.cortex import cortex as cortex_mod

        captured: list = []

        class _StubLLM:
            def with_structured_output(self, schema):
                return self

            async def ainvoke(self, messages, *a, **kw):
                captured.extend(messages)

                class _Out:
                    next_subgoal = "继续推进"
                    completion_reason = None
                    subgoal_completed = False

                return _Out()

        class _StubFactory:
            def __call__(self, *a, **kw):
                return _StubLLM()

        original = cortex_mod.get_llm
        cortex_mod.get_llm = _StubFactory()  # type: ignore[assignment]
        try:
            state = State(
                messages=[],
                initial_goal="用「传奇」再搜一次",
                subgoal_plan=[],
                latest_ui_hierarchy=None,
                latest_ocr_results=None,
                latest_screen_elements=None,
                focused_app_info=None,
                device_date=None,
                structured_decisions=None,
                complete_subgoals_by_ids=[],
                agents_thoughts=[],
                remaining_steps=10,
                executor_messages=[],
                cortex_last_thought=None,
                scratchpad={},
            )
            asyncio.run(CortexNode(ctx)(state))
        finally:
            cortex_mod.get_llm = original  # type: ignore[assignment]
        return captured

    def test_it_comes_right_after_the_system_prompt(self):
        captured = self._capture(_ctx(task_started_at=STARTED))

        assert captured, "cortex 节点没发出任何消息"
        assert captured[0].type == "system"
        assert "Task started at" in str(captured[1].content), (
            f"时间块必须紧跟系统提示词，实际第 1 条是 {str(captured[1].content)[:60]!r}"
        )
        assert "2026-10-26 22:16:33+08:00" in str(captured[1].content)

    def test_the_system_prompt_is_byte_identical(self):
        with_time = self._capture(_ctx(task_started_at=STARTED))
        without_time = self._capture(_ctx(task_started_at=None))

        assert str(with_time[0].content) == str(without_time[0].content)

    def test_no_block_when_the_time_is_unknown(self):
        captured = self._capture(_ctx(task_started_at=None))

        assert not any("Task started at" in str(m.content) for m in captured)
