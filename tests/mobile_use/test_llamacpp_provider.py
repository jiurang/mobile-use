"""Tests for the llama.cpp / llama-server LLM provider.

这个 provider 的每个参数都是**实测**出来的（2026-10-22，本机 llama-server
b11782，模型 ``unsloth/Qwen3.5-0.8B-GGUF:Q6_K``）。这些用例守的就是那几条
实测结论 —— 少一样都会让结构化输出直接失败：

* **思考链默认开着，而且会吃光 token 预算。** 不关掉时模型把输出全写进
  ``reasoning_content``，``content`` 恒为空 → 解析必失败。决定性对照
  （真实 schema + 真实客户端）::

      思考 ON  + max_tokens=256 → LengthFinishReasonError（JSON 没写完）
      思考 ON  + 不限长         → 挂死（实测 20s / 300s 两次超时都没返回）
      思考 OFF + 不限长         → 2.8s 正常解析出对象

* **不限长 = 挂死。** llama.cpp 的 ``n_predict`` 默认 -1，而思考链会一直续写。
  所以 ``max_tokens`` 必须有界。

* **结构化输出走 json_schema 语法约束**（llama.cpp 用 GBNF 硬约束）：
  实测 5 个角色的 schema 全部解析成功；「schema 当工具绑」那条路在 0.8B 上
  会「不调工具、直接写散文」（实测 1/2 失败）。

* **本地 server 默认不鉴权** —— 所以这个 provider **不要求** API Key
  （与 openai / minimax / deepseek 相反）。
"""

from __future__ import annotations

import asyncio
import os
import socket
from types import SimpleNamespace

import pytest
from langchain_openai.chat_models import base as lc_base
from pydantic import BaseModel, Field, SecretStr


# ---------------------------------------------------------------------------
# 1. 工厂参数：端点 / 思考开关 / 有界性
# ---------------------------------------------------------------------------


def test_get_llamacpp_llm_defaults_to_local_endpoint():
    """默认端点是本机 llama-server。"""
    from minitap.mobile_use.services.llm import get_llamacpp_llm

    client = get_llamacpp_llm()
    assert client.openai_api_base == "http://127.0.0.1:9931/v1"


def test_get_llamacpp_llm_honours_settings_base_url(monkeypatch):
    """端点可以改（server 不在本机默认端口时）。"""
    from minitap.mobile_use.config import settings
    from minitap.mobile_use.services.llm import get_llamacpp_llm

    monkeypatch.setattr(settings, "LLAMACPP_BASE_URL", "http://10.0.0.5:9931/v1", raising=False)
    client = get_llamacpp_llm()
    assert client.openai_api_base == "http://10.0.0.5:9931/v1"


def test_thinking_is_off_by_default():
    """思考链默认关 —— 这是这个 provider 能用的**前提**。

    开着思考时 content 恒为空，结构化输出必然失败。
    """
    from minitap.mobile_use.services.llm import get_llamacpp_llm

    client = get_llamacpp_llm()
    assert client.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_max_tokens_is_bounded():
    """不限长 = 挂死。llama.cpp 的 n_predict 默认 -1。"""
    from minitap.mobile_use.services.llm import get_llamacpp_llm

    client = get_llamacpp_llm()
    assert client.max_tokens is not None
    assert client.max_tokens > 0


def test_request_is_bounded():
    """不设超时的话 SDK 等 600s 且重试两次，一次挂死的连接能拖住整个 agent。"""
    from minitap.mobile_use.services.llm import LLM_REQUEST_TIMEOUT_SECONDS, get_llamacpp_llm

    client = get_llamacpp_llm()
    assert client.request_timeout == LLM_REQUEST_TIMEOUT_SECONDS
    assert client.max_retries == 1


def test_stream_usage_is_enabled():
    """不开它，流式调用的 usage_metadata 是 None，成本统计就是空的。"""
    from minitap.mobile_use.services.llm import get_llamacpp_llm

    client = get_llamacpp_llm()
    assert client.stream_usage is True


def test_temperature_is_passed_through():
    from minitap.mobile_use.services.llm import get_llamacpp_llm

    client = get_llamacpp_llm(temperature=0)
    assert client.temperature == 0


# ---------------------------------------------------------------------------
# 2. 渠道下发（manager → runner → mobile-use）
# ---------------------------------------------------------------------------


def test_channel_override_wins_over_settings():
    """manager 下发的端点/密钥必须盖过设备本地的 settings。"""
    from minitap.mobile_use.services.llm import ProviderOverride, get_llamacpp_llm

    client = get_llamacpp_llm(
        override=ProviderOverride(
            api_key=SecretStr("channel-key"),
            base_url="http://10.0.0.5:9931/v1",
            strict=True,
        )
    )
    assert client.openai_api_base == "http://10.0.0.5:9931/v1"
    assert client.openai_api_key.get_secret_value() == "channel-key"


def test_channel_without_api_key_is_allowed():
    """本地 server 默认不鉴权 —— 渠道没给密钥不能报错。

    这条与 openai / minimax / deepseek 的「缺 key 就报错」**相反**，
    别让后人照抄那边的断言。
    """
    from minitap.mobile_use.config import LLM
    from minitap.mobile_use.services.llm import ProviderOverride, get_llamacpp_llm

    node = LLM(provider="llamacpp", model="unsloth/Qwen3.5-0.8B-GGUF:Q6_K")
    node.validate_provider("Executor")  # 不抛异常即通过

    client = get_llamacpp_llm(
        override=ProviderOverride(base_url="http://127.0.0.1:9931/v1", strict=True)
    )
    # OpenAI SDK 不接受空 api_key，占位即可（server 不校验这个头）。
    assert client.openai_api_key.get_secret_value() == "sk-local"


def test_llamacpp_is_keyless_for_the_channel_path():
    """manager 渠道路径不能因为「没有 api_key」把这条渠道堵死。"""
    from minitap.mobile_use.config import provider_requires_api_key
    from minitap.mobile_use.sdk.utils import llm_config_for_single_model

    assert provider_requires_api_key("llamacpp") is False
    config = llm_config_for_single_model(
        provider="llamacpp",
        model="unsloth/Qwen3.5-0.8B-GGUF:Q6_K",
        base_url="http://127.0.0.1:9931/v1",
    )
    assert config.executor.provider == "llamacpp"
    assert config.executor.allow_env_fallback is False


# ---------------------------------------------------------------------------
# 3. 「思考模式」开关（manager web 端下发的那个）
# ---------------------------------------------------------------------------


def test_thinking_switch_from_the_channel():
    """渠道下发 thinking=True → 请求里开思考，且换一个更宽的 token 上限。

    思考链和正文**共用** max_tokens 预算，所以开思考时上限必须更宽 ——
    实测「思考 ON + 256 token」的结果是 JSON 没写完。
    """
    from minitap.mobile_use.services.llm import ProviderOverride, get_llamacpp_llm

    client = get_llamacpp_llm(
        override=ProviderOverride(thinking=True, strict=True),
    )
    assert client.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
    assert client.max_tokens == 8192


def test_thinking_switch_off_uses_the_small_budget():
    from minitap.mobile_use.services.llm import ProviderOverride, get_llamacpp_llm

    client = get_llamacpp_llm(override=ProviderOverride(thinking=False, strict=True))
    assert client.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert client.max_tokens == 1024


def test_thinking_switch_survives_the_channel_config():
    """开关要能从渠道一路走到 LLM 配置里（否则管理端改了没反应）。"""
    from minitap.mobile_use.sdk.utils import llm_config_for_single_model

    config = llm_config_for_single_model(
        provider="llamacpp",
        model="unsloth/Qwen3.5-0.8B-GGUF:Q6_K",
        base_url="http://127.0.0.1:9931/v1",
        thinking=True,
    )
    assert config.executor.thinking is True
    assert config.executor.fallback.thinking is True


def test_thinking_defaults_to_none_when_the_channel_does_not_set_it():
    """None = 没指定 = 由 provider 自己的默认决定（llamacpp 默认关）。"""
    from minitap.mobile_use.sdk.utils import llm_config_for_single_model

    config = llm_config_for_single_model(
        provider="llamacpp",
        model="unsloth/Qwen3.5-0.8B-GGUF:Q6_K",
        base_url="http://127.0.0.1:9931/v1",
    )
    assert config.executor.thinking is None


# ---------------------------------------------------------------------------
# 4. 分发：get_llm / get_structured_llm
# ---------------------------------------------------------------------------


def _ctx(llm_config):
    """get_llm 只读 ctx.llm_config 和 ctx.trace_id。"""
    return SimpleNamespace(llm_config=llm_config, trace_id="t-1")


def test_get_llm_dispatches_to_the_llamacpp_factory():
    from minitap.mobile_use.sdk.utils import llm_config_for_single_model
    from minitap.mobile_use.services.llm import LlamaCppChatOpenAI, get_llm

    config = llm_config_for_single_model(
        provider="llamacpp",
        model="unsloth/Qwen3.5-0.8B-GGUF:Q6_K",
        base_url="http://127.0.0.1:9931/v1",
    )
    client = get_llm(ctx=_ctx(config), name="executor")
    assert isinstance(client, LlamaCppChatOpenAI)
    assert client.openai_api_base == "http://127.0.0.1:9931/v1"


def test_get_structured_llm_uses_json_schema_for_llamacpp():
    """llamacpp 走原生 json_schema —— 0.8B 在工具绑定那条路上会不调工具。"""
    from minitap.mobile_use.agents.cortex.types import CortexOutput
    from minitap.mobile_use.sdk.utils import llm_config_for_single_model
    from minitap.mobile_use.services.llm import get_structured_llm

    config = llm_config_for_single_model(
        provider="llamacpp",
        model="unsloth/Qwen3.5-0.8B-GGUF:Q6_K",
        base_url="http://127.0.0.1:9931/v1",
    )
    runnable = get_structured_llm(ctx=_ctx(config), name="cortex", schema=CortexOutput)
    # json_schema 主路径 + 工具备路径
    assert hasattr(runnable, "_native")
    assert hasattr(runnable, "_tool")


def test_get_structured_llm_keeps_the_tool_path_for_other_providers():
    """回归护栏：minimax / deepseek 的行为必须与改动前**逐字一致**。

    改动前 executor / reporter 直接 new ``_ToolCallStructuredRunnable``；
    改动后它们走 ``get_structured_llm``，对非 llamacpp 的 provider 必须
    产出同一个东西（同样把 schema 绑成工具、同样 force_tool_choice=True）。
    """
    from minitap.mobile_use.agents.cortex.types import CortexOutput
    from minitap.mobile_use.sdk.utils import llm_config_for_single_model
    from minitap.mobile_use.services.llm import _ToolCallStructuredRunnable, get_structured_llm

    config = llm_config_for_single_model(
        provider="minimax",
        model="MiniMax-M3.1-Flash-Preview",
        api_key="sk-x",
    )
    runnable = get_structured_llm(ctx=_ctx(config), name="cortex", schema=CortexOutput)
    assert isinstance(runnable, _ToolCallStructuredRunnable)
    assert [t["function"]["name"] for t in runnable._bound.kwargs["tools"]] == ["CortexOutput"]
    # 强制 tool_choice（改动前就是 True）
    tool_choice = runnable._bound.kwargs["tool_choice"]
    assert tool_choice["function"]["name"] == "CortexOutput"


# ---------------------------------------------------------------------------
# 5. 结构化输出的两条路径
# ---------------------------------------------------------------------------


class _Subgoal(BaseModel):
    goal: str = Field(description="the subgoal")
    reasoning: str = Field(description="why")


class _PlannerOutput(BaseModel):
    subgoals: list[_Subgoal] = Field(description="all subgoals")


def test_llamacpp_primary_path_sends_response_format():
    """主路径是原生 ``response_format=json_schema``。

    llama.cpp 用 GBNF 硬约束输出 —— 这是这个 provider 可靠的原因。
    """
    from minitap.mobile_use.agents.planner.types import PlannerOutput
    from minitap.mobile_use.services.llm import get_llamacpp_llm

    captured: dict = {}
    original = lc_base.BaseChatOpenAI._get_request_payload

    class _Stop(Exception):
        pass

    def spy(self, input_, **kwargs):
        captured.update(original(self, input_, **kwargs))
        raise _Stop

    lc_base.BaseChatOpenAI._get_request_payload = spy
    try:
        get_llamacpp_llm().with_structured_output(PlannerOutput).invoke("hi")
    except Exception:
        # 主路径被 _Stop 打断后会去试备路径，备路径同样被 _Stop 打断 ——
        # 这里只关心**发出去的请求体**，异常本身不是被测对象。
        pass
    finally:
        lc_base.BaseChatOpenAI._get_request_payload = original

    assert captured["response_format"] is PlannerOutput


def test_llamacpp_fallback_binds_the_schema_as_a_tool():
    """备路径把 schema 当工具绑上去，供主路径解析失败时再试一次。"""
    from minitap.mobile_use.services.llm import get_llamacpp_llm

    runnable = get_llamacpp_llm().with_structured_output(_PlannerOutput)
    tool_path = runnable._tool
    assert [t["function"]["name"] for t in tool_path._bound.kwargs["tools"]] == ["_PlannerOutput"]
    # 不强制 tool_choice：0.8B 经常不调工具，强制的失败率更高。
    assert "tool_choice" not in tool_path._bound.kwargs


def test_llamacpp_falls_back_to_the_tool_path():
    """主路径炸了必须自动走备路径，而不是把异常直接抛给 agent。"""
    from minitap.mobile_use.services.llm import get_llamacpp_llm

    runnable = get_llamacpp_llm().with_structured_output(_PlannerOutput)

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


def test_llamacpp_reports_both_paths_when_both_fail():
    """两条都失败时，报错必须带上**两条**的原因。"""
    from minitap.mobile_use.services.llm import get_llamacpp_llm

    runnable = get_llamacpp_llm().with_structured_output(_PlannerOutput)

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


# ---------------------------------------------------------------------------
# 6. 真实端点（本地 llama-server）
# ---------------------------------------------------------------------------


def _server_is_up(host: str = "127.0.0.1", port: int = 9931) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


@pytest.mark.integration
def test_llamacpp_structured_output():
    """真连本地 server，断言解析出**对象**（不是 200）。

    失败模式是「返回了不合 schema 的内容」或「content 恒为空」——
    只断言 200 什么都测不到。
    """
    from minitap.mobile_use.services.llm import get_llamacpp_llm

    if not _server_is_up():
        pytest.skip("no llama-server listening on 127.0.0.1:9931")

    model = os.environ.get("LLAMACPP_MODEL", "unsloth/Qwen3.5-0.8B-GGUF:Q6_K")
    client = get_llamacpp_llm(model_name=model, temperature=0)
    structured = client.with_structured_output(_PlannerOutput)
    result = asyncio.run(structured.ainvoke("Split 'buy milk' into subgoals."))

    assert isinstance(result, _PlannerOutput)
    assert result.subgoals
