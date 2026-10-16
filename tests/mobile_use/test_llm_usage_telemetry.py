"""LLM token 埋点的单测。

要锁住三件事：
  1. usage 能从 langchain 的各种响应形状里读出来（复用 TUI 已验证的解析）
  2. 埋点不因异常打断主流程 —— 这是最关键的：统计代码永远不能把任务带崩
  3. agent 名与 fallback 标记正确透传
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from minitap.mobile_use.services import llm as llm_module
from minitap.mobile_use.utils.llm_usage import extract_usage


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

    def test_missing_usage_records_zeros_not_crash(self, captured):
        """provider 不返回 usage 时记 0，不能抛异常。"""
        async def call():
            return _Obj()

        asyncio.run(llm_module.invoke_llm_with_timeout_message(call(), agent_name="planner"))
        assert len(captured) == 1
        assert captured[0]["input_tokens"] == 0
        assert captured[0]["output_tokens"] == 0

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
