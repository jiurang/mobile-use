"""前缀缓存命中量必须能被读到并进 telemetry。

背景
----
`extract_usage` 只认 input / output 两个数。**缓存命中量是第三个数**：
provider 报的是「本次输入里有多少 token 直接读了缓存、没有重算」，它是 input
的子集。把它并进 input 会算错成本，丢掉它则遥测里**根本没有命中率这个指标**
—— 而「提示词分层到底有没有用」只能靠它证明。

2026-10-21 实测（MiniMax M3.1-Flash，OpenAI 兼容端点）：
    usage.prompt_tokens_details.cached_tokens = 10,477
同一个 body 换个尾部内容再发，立刻掉到 2,329 —— 这个数就是击穿现场。

这里覆盖各家的字段名 + 端到端进 telemetry，不连网、不花钱。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from minitap.mobile_use.services import telemetry as telemetry_module
from minitap.mobile_use.services.llm import (
    _install_usage_capture,
    invoke_llm_with_timeout_message,
)
from minitap.mobile_use.utils.llm_usage import extract_cache_tokens


class _Msg:
    """带原始 OpenAI 风格 usage 的假响应。"""

    def __init__(self, usage: dict):
        self.usage_metadata = None
        self.llm_output = None
        self.response_metadata = {"token_usage": usage}


def test_reads_openai_compatible_prompt_tokens_details():
    msg = _Msg({"prompt_tokens": 10_590, "prompt_tokens_details": {"cached_tokens": 10_411}})
    assert extract_cache_tokens(msg) == 10_411


def test_reads_deepseek_prompt_cache_hit_tokens():
    msg = _Msg({"prompt_tokens": 10_590, "prompt_cache_hit_tokens": 2_329})
    assert extract_cache_tokens(msg) == 2_329


def test_reads_anthropic_cache_read_input_tokens():
    msg = _Msg({"input_tokens": 10_590, "cache_read_input_tokens": 7_351})
    assert extract_cache_tokens(msg) == 7_351


def test_reads_langchain_normalized_input_token_details():
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 10_590,
            "output_tokens": 64,
            "total_tokens": 10_654,
            "input_token_details": {"cache_read": 8_876},
        },
    )
    assert extract_cache_tokens(message) == 8_876


def test_absent_cache_field_is_zero():
    assert extract_cache_tokens(_Msg({"prompt_tokens": 100})) == 0
    assert extract_cache_tokens(None) == 0


def test_unwraps_llm_result_generations():
    """LLMResult 的 usage 藏在 generations[0][0].message 上。"""

    class _Result:
        def __init__(self, message):
            self.generations = [[ChatGeneration(message=message)]]

    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 1,
            "total_tokens": 11,
            "input_token_details": {"cache_read": 512},
        },
    )
    assert extract_cache_tokens(_Result(message)) == 512


class _StubChat(ChatOpenAI):
    """返回带缓存命中量的假响应。"""

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="ok",
                        usage_metadata={
                            "input_tokens": 10_590,
                            "output_tokens": 64,
                            "total_tokens": 10_654,
                            "input_token_details": {"cache_read": 8_876},
                        },
                    )
                )
            ]
        )


@pytest.fixture
def recorded_calls(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        telemetry_module.telemetry, "capture_llm_usage", lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setenv("MINITAP_API_KEY", "sk-dummy")
    return calls


@pytest.mark.asyncio
async def test_cached_tokens_reach_telemetry(recorded_calls):
    _install_usage_capture()
    llm = _StubChat(model="stub", api_key="sk-dummy")

    async def one_call():
        await llm.ainvoke([{"role": "user", "content": "hi"}])
        return "done"

    await invoke_llm_with_timeout_message(
        one_call(), agent_name="cortex", task_id="t1", timeout_seconds=5
    )

    assert len(recorded_calls) == 1
    call = recorded_calls[0]
    assert call["agent"] == "cortex"
    assert call["input_tokens"] == 10_590
    assert call["output_tokens"] == 64
    assert call["cached_tokens"] == 8_876
