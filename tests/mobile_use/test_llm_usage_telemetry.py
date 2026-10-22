"""LLM token 埋点的单测。

要锁住三件事：
  1. usage 能从 langchain 的各种响应形状里读出来（复用 TUI 已验证的解析）
  2. 埋点不因异常打断主流程 —— 这是最关键的：统计代码永远不能把任务带崩
  3. agent 名与 fallback 标记正确透传
"""

from __future__ import annotations

import asyncio
from typing import Any, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from minitap.mobile_use.services import llm as llm_module
from minitap.mobile_use.utils.llm_usage import extract_usage, record_raw_response


class _GraphState(TypedDict):
    """最小 graph state —— 只为测「节点边界」。"""

    value: int


class _Obj:
    """鸭子类型桩，用属性模拟各类响应形状。"""

    def __init__(self, **attrs) -> None:
        self.usage_metadata = None
        self.llm_output = None
        self.response_metadata = None
        self.generations = None
        for key, value in attrs.items():
            setattr(self, key, value)


class TestExtractUsage:
    def test_standard_usage_metadata(self):
        assert extract_usage(_Obj(usage_metadata={"input_tokens": 100, "output_tokens": 20})) == (
            100,
            20,
        )

    def test_openai_style_llm_output(self):
        resp = _Obj(llm_output={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3}})
        assert extract_usage(resp) == (7, 3)

    def test_gemini_style_response_metadata(self):
        resp = _Obj(
            response_metadata={"usage": {"promptTokenCount": 11, "candidatesTokenCount": 5}}
        )
        assert extract_usage(resp) == (11, 5)

    def test_unwraps_llm_result(self):
        """on_llm_end / ainvoke 拿到的是 LLMResult，字段在 generations 里。"""
        inner = _Obj(usage_metadata={"input_tokens": 42, "output_tokens": 9})
        gen = _Obj(message=inner)
        assert extract_usage(_Obj(generations=[[gen]])) == (42, 9)

    def test_missing_usage_returns_none(self):
        """读不到时返回 None 而不是 (0,0) —— 后者会让统计看起来像「免费」。"""
        assert extract_usage(_Obj()) is None
        assert extract_usage(None) is None

    def test_broken_values_do_not_raise(self):
        resp = _Obj(usage_metadata={"input_tokens": "nonsense", "output_tokens": None})
        assert extract_usage(resp) == (0, 0)

    def test_negative_values_are_clamped(self):
        resp = _Obj(usage_metadata={"input_tokens": -5, "output_tokens": -1})
        assert extract_usage(resp) == (0, 0)


class TestInvokeRecordsUsage:
    """invoke_llm_with_timeout_message 的埋点行为。"""

    @pytest.fixture
    def captured(self, monkeypatch):
        events: list[dict] = []

        def fake_capture(**kwargs):
            events.append(kwargs)

        monkeypatch.setattr(llm_module.telemetry, "capture_llm_usage", fake_capture)
        return events

    @staticmethod
    def _response(**attrs) -> Any:
        return _Obj(usage_metadata={"input_tokens": 100, "output_tokens": 25}, **attrs)

    def test_records_usage_with_agent_name(self, captured):
        async def call():
            return self._response()

        result = asyncio.run(
            llm_module.invoke_llm_with_timeout_message(call(), agent_name="cortex", task_id="t1")
        )
        assert result is not None
        assert len(captured) == 1
        assert captured[0]["agent"] == "cortex"
        assert captured[0]["input_tokens"] == 100
        assert captured[0]["output_tokens"] == 25
        assert captured[0]["used_fallback"] is False

    def test_fallback_flag_is_propagated(self, captured):
        async def call():
            return self._response()

        asyncio.run(
            llm_module.invoke_llm_with_timeout_message(
                call(), agent_name="executor", task_id="t1", used_fallback=True
            )
        )
        assert captured[0]["used_fallback"] is True

    def test_skips_recording_without_agent_name(self, captured):
        """没传 agent 名就不记 —— 避免把调用归到 'unknown' 污染统计。"""

        async def call():
            return self._response()

        asyncio.run(llm_module.invoke_llm_with_timeout_message(call(), task_id="t1"))
        assert captured == []

    def test_missing_usage_records_nothing_not_crash(self, captured):
        """provider 不返回 usage 时**不记这一行**（而不是记 0），且不能抛异常。

        契约变更（2026-10-21）
        --------------------
        原来这里记 0，理由是「provider 根本没回报 usage，不是我们丢的」。但
        **「provider 没报」和「我们没拿到」在实现上分不开**，所以改成
        **读不到就不报这一行**。

        ⚠️ 2026-10-21 更正：本段原来写着「`_record_usage` 在 LangGraph 的图里
        压根收不到响应（ContextVar 跨不过图的任务上下文）」「图里的 usage 由
        `services/usage_bridge.py` 的 graph-config callback 上报」——
        **两处都是假的**：`usage_bridge.py` 从未存在；捕获槽装的是同一个 list
        对象，跨图可见。真机 graph 实测（`probe_out/executor_e2e.json`）：
        executor 21 次调用 / 120,772 input / 65,672 cached，全是真实非零值。
        当时「每一行都是 0」的真因是客户端没开 `stream_usage=True`，
        已在 `services/llm.py` 修好。

        现在的口径与 manager 一致（`manager-server` 的 api-contract.md：
        「整次执行没有上报过 usage 时这些字段全是 null——『没上报』不等于
        『零消耗』」）：**读不到就不报这一行**。
        """

        async def call():
            return _Obj()

        asyncio.run(llm_module.invoke_llm_with_timeout_message(call(), agent_name="planner"))
        assert captured == []

    def test_result_returned_even_when_recording_explodes(self, monkeypatch):
        """埋点自身抛异常时，主流程必须照常返回结果。"""

        def boom(**kwargs):
            raise RuntimeError("telemetry down")

        monkeypatch.setattr(llm_module.telemetry, "capture_llm_usage", boom)

        async def call():
            return self._response()

        result = asyncio.run(
            llm_module.invoke_llm_with_timeout_message(call(), agent_name="cortex")
        )
        assert result is not None, "埋点异常吞掉了 LLM 结果"
        assert result.usage_metadata["input_tokens"] == 100

    def test_timeout_path_still_records(self, captured):
        """走超时分支（先打印提示再等结果）时同样要记 usage。"""

        async def slow():
            await asyncio.sleep(0.05)
            return self._response()

        result = asyncio.run(
            llm_module.invoke_llm_with_timeout_message(
                slow(), timeout_seconds=0, agent_name="video_analyzer"
            )
        )
        assert result is not None
        assert len(captured) == 1
        assert captured[0]["duration_seconds"] >= 0


class TestUsageSurvivesGraphBoundary:
    """**图内的 LLM 调用也必须上报真实 token**（不许记 0）。

    这条锁的是一个被反复转述、却从没被验证过的结论。仓库里曾到处写着：

        「在 LangGraph 的图里那个捕获槽根本收不到东西（ContextVar 跨不过图的
         任务上下文），图里的 usage 由 `services/usage_bridge.py` 的 graph-config
         callback 上报。」

    两处都是假的：

    * `services/usage_bridge.py` **从未存在**（git 全历史无此文件）；
    * 捕获槽**跨得过去** —— 槽里装的是**同一个 list 对象**，LangGraph 的节点
      任务继承父 context，append 外面看得到。

    当时「每一行都记成 0」的真因是客户端没开 ``stream_usage=True``
    （provider 压根没报 usage），不是 ContextVar。

    代价很实在：照那条结论去实现，图里的成本数据全是 0 ⇒ 缓存命中率算不出来
    ⇒ §11.2 的验收判据直接失效。所以这里用最小 LangGraph 把它钉死。

    真机侧佐证：`probe_out/executor_e2e.json`（真机 graph 运行）里
    executor 21 次调用 / 120,772 in / 65,672 cached / 18,114 out 全是真实值。
    """

    @pytest.fixture
    def captured(self, monkeypatch):
        events: list[dict] = []

        def fake_capture(**kwargs):
            events.append(kwargs)

        monkeypatch.setattr(llm_module.telemetry, "capture_llm_usage", fake_capture)
        return events

    @staticmethod
    def _raw_response() -> Any:
        """一条带 usage 的原始响应（provider 的真实形状）。"""
        return _Obj(usage_metadata={"input_tokens": 1234, "output_tokens": 56})

    def test_graph_node_records_real_tokens(self, captured):
        """节点里经 graph 调 LLM ⇒ 上报**非零** token。"""

        async def call():
            record_raw_response(self._raw_response())
            # 返回体**身上没有** usage —— 和 with_structured_output 一样，
            # usage 只藏在原始响应里。这正是最容易被误判成「读不到」的那一格。
            return {"parsed": "yes"}

        async def node(state):
            await llm_module.invoke_llm_with_timeout_message(
                call(), agent_name="executor", task_id="t1"
            )
            return {"value": state["value"] + 1}

        async def run():
            g = StateGraph(_GraphState)
            g.add_node("n", node)
            g.add_edge(START, "n")
            g.add_edge("n", END)
            await g.compile().ainvoke({"value": 0}, config={"recursion_limit": 10})

        asyncio.run(run())

        assert len(captured) == 1, "图内调用没有上报 —— 埋点在图边界上断了"
        assert captured[0]["input_tokens"] == 1234
        assert captured[0]["output_tokens"] == 56
        assert captured[0]["agent"] == "executor"

    def test_graph_node_records_real_tokens_when_result_carries_usage(self, captured):
        """结果自带 usage_metadata 时同样上报（对照形态）。"""

        async def call():
            record_raw_response(self._raw_response())
            return self._raw_response()

        async def node(state):
            await llm_module.invoke_llm_with_timeout_message(
                call(), agent_name="cortex", task_id="t1"
            )
            return {"value": state["value"] + 1}

        async def run():
            g = StateGraph(_GraphState)
            g.add_node("n", node)
            g.add_edge(START, "n")
            g.add_edge("n", END)
            await g.compile().ainvoke({"value": 0}, config={"recursion_limit": 10})

        asyncio.run(run())

        assert len(captured) == 1
        assert captured[0]["input_tokens"] == 1234
