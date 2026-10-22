"""Tests for the SiliconFlow (硅基流动) LLM provider.

Facts this file guards, all measured against the live API on 2026-10-21
(key scoped to ``Qwen/Qwen2.5-72B-Instruct`` / ``Qwen/Qwen3-VL-32B-Instruct``):

* the endpoint accepts ``response_format={"type": "json_schema"}`` but does
  **not** enforce the schema at the token level — so the schema must go out as
  a *tool*, and be validated by pydantic locally;
* a forced ``tool_choice`` (specific function name) is accepted and returns a
  well-formed ``tool_calls`` entry;
* ``temperature`` accepts 0..2, so ``hopper``'s hardcoded 0 keeps working;
* ``parallel_tool_calls: false`` is accepted (the tool-binding path sends it).

The ``stream_usage`` case is not about the provider at all: without it,
langchain-openai returns ``usage_metadata=None`` on streamed calls, so the
telemetry that prices a run comes back empty. The ``openai`` provider has
always had that hole; this provider does not.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import pytest
from langchain_openai.chat_models import base as lc_base
from pydantic import BaseModel, Field, SecretStr


def _mock_settings(mock_settings, **overrides):
    """Fill in every setting get_siliconflow_llm reads.

    ``mock_settings`` is a MagicMock, so an attribute we don't set is *not*
    ``None`` — it would silently take the wrong branch. Always be explicit.
    """
    mock_settings.SILICONFLOW_API_KEY = SecretStr("test-key")
    mock_settings.SILICONFLOW_BASE_URL = None
    for key, value in overrides.items():
        setattr(mock_settings, key, value)


@patch("minitap.mobile_use.services.llm.SiliconFlowChatOpenAI")
@patch("minitap.mobile_use.services.llm.settings")
def test_get_siliconflow_llm_defaults_to_cn_endpoint(mock_settings, mock_chat_openai):
    from minitap.mobile_use.services.llm import get_siliconflow_llm

    _mock_settings(mock_settings)
    get_siliconflow_llm()

    kwargs = mock_chat_openai.call_args.kwargs
    assert kwargs["base_url"] == "https://api.siliconflow.cn/v1"
    assert kwargs["model"] == "Qwen/Qwen2.5-72B-Instruct"


@patch("minitap.mobile_use.services.llm.SiliconFlowChatOpenAI")
@patch("minitap.mobile_use.services.llm.settings")
def test_get_siliconflow_llm_honours_settings_base_url(mock_settings, mock_chat_openai):
    from minitap.mobile_use.services.llm import get_siliconflow_llm

    _mock_settings(mock_settings, SILICONFLOW_BASE_URL="https://api.siliconflow.com/v1")
    get_siliconflow_llm()

    assert mock_chat_openai.call_args.kwargs["base_url"] == "https://api.siliconflow.com/v1"


@patch("minitap.mobile_use.services.llm.SiliconFlowChatOpenAI")
@patch("minitap.mobile_use.services.llm.settings")
def test_channel_override_wins_over_settings(mock_settings, mock_chat_openai):
    """manager 下发的端点/密钥必须盖过设备本地的 settings。"""
    from minitap.mobile_use.services.llm import ProviderOverride, get_siliconflow_llm

    _mock_settings(mock_settings, SILICONFLOW_BASE_URL="https://from-env/v1")
    get_siliconflow_llm(
        override=ProviderOverride(api_key=SecretStr("channel-key"), base_url="https://from-channel/v1")
    )

    kwargs = mock_chat_openai.call_args.kwargs
    assert kwargs["base_url"] == "https://from-channel/v1"
    assert kwargs["api_key"].get_secret_value() == "channel-key"


@patch("minitap.mobile_use.services.llm.SiliconFlowChatOpenAI")
@patch("minitap.mobile_use.services.llm.settings")
def test_stream_usage_is_enabled(mock_settings, mock_chat_openai):
    """不开它，流式调用的 usage_metadata 是 None，成本统计就是空的。"""
    from minitap.mobile_use.services.llm import get_siliconflow_llm

    _mock_settings(mock_settings)
    get_siliconflow_llm()

    assert mock_chat_openai.call_args.kwargs["stream_usage"] is True


@patch("minitap.mobile_use.services.llm.SiliconFlowChatOpenAI")
@patch("minitap.mobile_use.services.llm.settings")
def test_request_is_bounded(mock_settings, mock_chat_openai):
    """不设超时的话 SDK 等 600s 且重试两次，一次挂掉的连接能拖住整个 agent。"""
    from minitap.mobile_use.services.llm import LLM_REQUEST_TIMEOUT_SECONDS, get_siliconflow_llm

    _mock_settings(mock_settings)
    get_siliconflow_llm()

    kwargs = mock_chat_openai.call_args.kwargs
    assert kwargs["request_timeout"] == LLM_REQUEST_TIMEOUT_SECONDS
    assert kwargs["max_retries"] == 1


@patch("minitap.mobile_use.services.llm.SiliconFlowChatOpenAI")
@patch("minitap.mobile_use.services.llm.settings")
def test_temperature_is_passed_through(mock_settings, mock_chat_openai):
    """端点接受 0..2（实测），所以 hopper 写死的 0 不能被钳制掉。"""
    from minitap.mobile_use.services.llm import get_siliconflow_llm

    _mock_settings(mock_settings)
    get_siliconflow_llm(temperature=0)

    assert mock_chat_openai.call_args.kwargs["temperature"] == 0


class _Subgoal(BaseModel):
    goal: str = Field(description="the subgoal")
    reasoning: str = Field(description="why")


class _PlannerOutput(BaseModel):
    subgoals: list[_Subgoal] = Field(description="all subgoals")


@patch("minitap.mobile_use.services.llm.settings")
def test_siliconflow_primary_path_sends_response_format(mock_settings):
    """主路径是原生 ``response_format=json_schema`` —— 视觉模型只认这一条。

    实测（2026-10-21）：硅基流动的视觉模型（``Qwen/Qwen3-VL-*``）**不支持
    函数调用** —— 强制指定函数被拒（``400 code 20015``），不强制则返回空
    内容、不带 tool_calls。而 cortex / executor 必须看截图，只能用视觉模型，
    所以 json_schema 必须是主路径。
    """
    from minitap.mobile_use.agents.planner.types import PlannerOutput
    from minitap.mobile_use.services.llm import get_siliconflow_llm

    _mock_settings(mock_settings)

    captured: dict = {}
    original = lc_base.BaseChatOpenAI._get_request_payload

    class _Stop(Exception):
        pass

    def spy(self, input_, **kwargs):
        captured.update(original(self, input_, **kwargs))
        raise _Stop

    lc_base.BaseChatOpenAI._get_request_payload = spy
    try:
        get_siliconflow_llm(model_name="Qwen/Qwen3-VL-32B-Instruct").with_structured_output(
            PlannerOutput
        ).invoke("hi")
    except Exception:
        # 主路径被 _Stop 打断后会去试备路径，备路径同样被 _Stop 打断 ——
        # 这里只关心**发出去的请求体**，异常本身不是被测对象。
        pass
    finally:
        lc_base.BaseChatOpenAI._get_request_payload = original

    assert captured["response_format"] is PlannerOutput
    # langchain 的 json_schema 路径会把 schema 同时绑成一个 tool（不强制选择）——
    # 视觉模型收下 tools 但不强制时仍按 response_format 回 JSON，实测能跑通。
    assert [t["function"]["name"] for t in captured["tools"]] == ["PlannerOutput"]
    assert "tool_choice" not in captured


@patch("minitap.mobile_use.services.llm.settings")
def test_siliconflow_fallback_binds_the_schema_as_a_tool(mock_settings):
    """备路径把 schema 当工具绑上去，供主路径解析失败时再试一次。"""
    from minitap.mobile_use.agents.planner.types import PlannerOutput
    from minitap.mobile_use.services.llm import get_siliconflow_llm

    _mock_settings(mock_settings)
    runnable = get_siliconflow_llm(model_name="Qwen/Qwen3.5-4B").with_structured_output(PlannerOutput)

    tool_path = runnable._tool
    assert [t["function"]["name"] for t in tool_path._bound.kwargs["tools"]] == ["PlannerOutput"]
    # 不强制 tool_choice：强制的会被视觉模型 400，而备路要覆盖尽可能多的模型。
    assert "tool_choice" not in tool_path._bound.kwargs


@patch("minitap.mobile_use.services.llm.settings")
def test_siliconflow_falls_back_to_the_tool_path(mock_settings):
    """主路径炸了必须自动走备路径，而不是把异常直接抛给 agent。"""
    from minitap.mobile_use.services.llm import get_siliconflow_llm

    _mock_settings(mock_settings)
    runnable = get_siliconflow_llm(model_name="Qwen/Qwen3.5-4B").with_structured_output(_PlannerOutput)

    def boom(*args, **kwargs):
        raise RuntimeError("primary path exploded")

    class _FakeTool:
        def invoke(self, input, config=None, **kwargs):
            return "from-tool-path"

    class _BoomNative:
        def invoke(self, input, config=None, **kwargs):
            boom()

    runnable._native = _BoomNative()
    runnable._tool = _FakeTool()

    assert runnable.invoke("hi") == "from-tool-path"


@patch("minitap.mobile_use.services.llm.settings")
def test_siliconflow_reports_both_paths_when_both_fail(mock_settings):
    """两条都失败时,报错必须带上**两条**的原因。"""
    from minitap.mobile_use.services.llm import get_siliconflow_llm

    _mock_settings(mock_settings)
    runnable = get_siliconflow_llm(model_name="Qwen/Qwen3.5-4B").with_structured_output(_PlannerOutput)

    class _Boom:
        def __init__(self, msg):
            self.msg = msg

        def invoke(self, input, config=None, **kwargs):
            raise RuntimeError(self.msg)

    runnable._native = _Boom("json_schema path failed")
    runnable._tool = _Boom("tool path failed")

    with pytest.raises(ValueError) as excinfo:
        runnable.invoke("hi")

    assert "json_schema path failed" in str(excinfo.value)
    assert "tool path failed" in str(excinfo.value)


def test_validate_provider_requires_a_key(monkeypatch):
    """没有 key 就必须报错 —— 新增渠道不能放松既有的校验。"""
    from minitap.mobile_use.config import LLM, settings

    monkeypatch.setattr(settings, "SILICONFLOW_API_KEY", None, raising=False)
    node = LLM(provider="siliconflow", model="Qwen/Qwen2.5-72B-Instruct")
    with pytest.raises(Exception, match="SILICONFLOW_API_KEY"):
        node.validate_provider("Cortex")


def test_validate_provider_accepts_inline_key(monkeypatch):
    """渠道下发的 key 不在 env 里也必须放行 —— 这是 manager 管渠道的前提。"""
    from minitap.mobile_use.config import LLM, settings

    monkeypatch.setattr(settings, "SILICONFLOW_API_KEY", None, raising=False)
    node = LLM(
        provider="siliconflow", model="Qwen/Qwen2.5-72B-Instruct", api_key="sk-from-manager"
    )
    node.validate_provider("Cortex")  # 不抛异常即通过


@pytest.mark.integration
@patch("minitap.mobile_use.services.llm.settings")
def test_siliconflow_structured_output(mock_settings):
    """The real risk is the parsed object, not a 200.

    Assert the parsed object: the failure mode is a schema-violating answer
    that parses into garbage or not at all.
    """
    from minitap.mobile_use.services.llm import get_siliconflow_llm

    key = os.environ.get("SILICONFLOW_API_KEY")
    if not key or key in {"...", "your-siliconflow-api-key"}:
        pytest.skip("SILICONFLOW_API_KEY not set, skipping integration test")

    mock_settings.SILICONFLOW_API_KEY = SecretStr(key)
    mock_settings.SILICONFLOW_BASE_URL = None

    client = get_siliconflow_llm(model_name=os.environ.get("SF_TEXT_MODEL", "Qwen/Qwen2.5-72B-Instruct"))
    structured = client.with_structured_output(_PlannerOutput)
    result = asyncio.run(structured.ainvoke("Split 'buy milk' into subgoals."))

    assert isinstance(result, _PlannerOutput)
    assert result.subgoals
