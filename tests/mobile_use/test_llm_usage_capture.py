"""结构化输出路径必须能把真实 token 用量送进 telemetry。

背景：cortex / planner / orchestrator 走的都是 `with_structured_output`，
调用方拿到的是 pydantic 对象，原始 AIMessage 落不到手里 —— 于是
`_record_usage` 只能记 0，telemetry 里这几个 agent 的 token 恒为 0，
任何"省了多少 token"的结论都无法验证。

修法是一层 ContextVar 捕获槽 + 客户端类上 generate/agenerate 的薄包装。
这里用离线 stub 模型把整条链路跑通，不连网、不花钱。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from minitap.mobile_use.services import telemetry as telemetry_module
from minitap.mobile_use.services.llm import (
    AnthropicCacheControlChat,
    _install_usage_capture,
    invoke_llm_with_timeout_message,
)


class _StructuredOutput(BaseModel):
    """假装成结构化输出返回的 pydantic 对象：身上没有 usage。"""

    decisions: str | None = None


class _StubChat(ChatOpenAI):
    """只实现 `_agenerate`，返回带 usage_metadata 的假响应。"""

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="ok",
                        usage_metadata={
                            "input_tokens": 123,
                            "output_tokens": 45,
                            "total_tokens": 168,
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
async def test_structured_path_records_real_usage(recorded_calls):
    _install_usage_capture()
    llm = _StubChat(model="stub", api_key="sk-dummy")

    async def structured_call():
        # 模拟 with_structured_output：原始 AIMessage 在客户端侧被捕获，
        # 调用方只拿到 pydantic 对象。
        await llm.ainvoke([HumanMessage("hi")])
        return _StructuredOutput()

    result = await invoke_llm_with_timeout_message(
        structured_call(), agent_name="cortex", task_id="t1", timeout_seconds=5
    )

    assert isinstance(result, _StructuredOutput)
    assert len(recorded_calls) == 1
    call = recorded_calls[0]
    assert call["agent"] == "cortex"
    assert call["input_tokens"] == 123
    assert call["output_tokens"] == 45


@pytest.mark.asyncio
async def test_capture_slot_does_not_leak(recorded_calls):
    """两次连续调用各记各的；第二次不该把第一次的响应也算进来。"""
    _install_usage_capture()
    llm = _StubChat(model="stub", api_key="sk-dummy")

    async def one_call():
        await llm.ainvoke([HumanMessage("hi")])
        return _StructuredOutput()

    for _ in range(2):
        await invoke_llm_with_timeout_message(
            one_call(), agent_name="cortex", task_id="t1", timeout_seconds=5
        )

    assert len(recorded_calls) == 2
    assert all(c["input_tokens"] == 123 for c in recorded_calls)


def test_usage_capture_is_installed_idempotently():
    _install_usage_capture()
    _install_usage_capture()
    for name in ("generate", "agenerate"):
        assert getattr(getattr(ChatOpenAI, name), "_usage_capture_wrapped", False)


def test_anthropic_payload_carries_cache_control():
    client = AnthropicCacheControlChat(
        model_name="claude-sonnet-4-6", api_key="sk-dummy", max_tokens=1024
    )
    system_text = "You are the Cortex. " * 100
    payload = client._get_request_payload(
        [SystemMessage(content=system_text), HumanMessage(content="hi")]
    )
    system = payload["system"]
    assert isinstance(system, list), f"system 未被包成 block 列表: {system!r}"
    assert system[-1]["text"] == system_text
    assert system[-1]["cache_control"] == {"type": "ephemeral"}
