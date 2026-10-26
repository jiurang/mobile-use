"""渠道目录（channel catalog）的契约测试。

这个目录是「渠道改造」的核心：**加一个供应商 = 加一条数据**，不再是加一个工厂
函数。所以它需要被测住三件事：

1. **目录是唯一的真相** —— ``LLMProvider`` 字面量、密钥要求、端点要求全都从它
   推导，不存在第二份手抄的清单。
2. **渠道的旋钮真的生效** —— 尤其是「要不要思考」：以前渠道下发的 thinking 被
   DeepSeek 完全无视（只读 .env），于是每个 agent 节点都 400。
3. **目录能被导出给 manager 侧** —— 添加渠道的表单靠它给出「选哪个渠道 / 选哪个
   模型 / 默认地址是什么」。

写这些用例时踩过的坑：

* 测「渠道下发」必须走**渠道路径**（``allow_env_fallback=False`` → strict），
  直接调工厂走的是 .env 那条老路，两条路的 thinking 处理不一样。
* 断言「有没有发某个请求体字段」要看**值**，不要看 key 在不在 —— 协议层统一之后
  空字段也会出现（值是 None），「不发」和「发 None」在请求体上一回事。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from minitap.mobile_use.config import (
    ENDPOINTLESS_PROVIDERS,
    KEYLESS_PROVIDERS,
    LLM,
    LLMProvider,
    provider_requires_api_key,
    provider_supports_endpoint,
)
from minitap.mobile_use.sdk.utils import llm_config_for_single_model
from minitap.mobile_use.services.channels import (
    CHANNELS,
    catalog_payload,
    channel_ids,
    default_base_url,
    get_channel,
    require_channel,
)
from minitap.mobile_use.services.llm import (
    ProviderOverride,
    build_openai_protocol_client,
    get_deepseek_llm,
    get_llm,
    get_llamacpp_llm,
    get_minimax_llm,
)


def _ctx(llm_config):
    return SimpleNamespace(llm_config=llm_config, trace_id="t-1")


def _channel_override(provider: str, model: str, **kwargs) -> ProviderOverride:
    """复刻 runner-host 的 ``_to_llm_spec`` → ``ProviderOverride`` 那条链。

    密钥必须自带：渠道路径是 strict 的（不许回退到设备本地的 .env）。
    """
    llm = LLM(
        provider=provider,
        model=model,
        api_key=kwargs.pop("api_key", "channel-key"),
        allow_env_fallback=False,
        **kwargs,
    )
    return ProviderOverride(
        api_key=llm.api_key,
        base_url=llm.base_url,
        thinking=llm.thinking,
        reasoning_effort=llm.reasoning_effort,
        strict=not llm.allow_env_fallback,
    )


# ---------------------------------------------------------------------------
# 1. 目录是唯一的真相
# ---------------------------------------------------------------------------


def test_provider_literal_is_derived_from_the_catalog():
    """``LLMProvider`` 必须和目录一模一样。

    以前它是手抄的一份字面量：加一个渠道要同时改目录和它，漏改的后果是
    「渠道在 manager 侧能存，到设备上却因为 provider 非法起不来」。
    """
    assert set(LLMProvider.__args__) == set(channel_ids())


def test_keyless_and_endpointless_sets_are_derived_from_the_catalog():
    assert KEYLESS_PROVIDERS == {cid for cid, spec in CHANNELS.items() if not spec.requires_api_key}
    assert ENDPOINTLESS_PROVIDERS == {
        cid for cid, spec in CHANNELS.items() if not spec.supports_endpoint
    }
    # 三个已知的「不要密钥」渠道：服务账号凭据 / 端点不鉴权。
    assert KEYLESS_PROVIDERS == {"azure", "vertexai", "llamacpp"}
    assert ENDPOINTLESS_PROVIDERS == {"google", "vertexai"}


def test_provider_helpers_read_the_catalog():
    for cid in channel_ids():
        spec = require_channel(cid)
        assert provider_requires_api_key(cid) is spec.requires_api_key
        assert provider_supports_endpoint(cid) is spec.supports_endpoint


def test_every_channel_declares_a_usable_protocol_and_client():
    """每条渠道都要能落到一个客户端实现上 —— 不能有「查得到、用不了」的条目。"""
    from minitap.mobile_use.services.llm import _OPENAI_CLIENT_KINDS

    for cid, spec in CHANNELS.items():
        assert spec.protocol in {"openai", "anthropic", "google", "vertexai", "azure"}
        if spec.protocol == "openai":
            assert spec.client_kind in _OPENAI_CLIENT_KINDS, cid
        # 有默认端点的渠道，目录里必须写着它；没有的（google/vertexai/azure/
        # custom）必须显式声明「必须自己给」。
        if spec.base_url is None and spec.protocol in {"openai", "anthropic"}:
            assert spec.requires_base_url, cid


def test_every_channel_client_kind_is_a_declared_kind():
    """🔴 渠道用的 ``client_kind`` 必须在 ``ClientKind`` 里**声明过**。

    踩过的坑（2026-10-24）：目录里 deepseek 渠道写着 ``client_kind="deepseek"``，
    而 ``ClientKind`` 字面量里已经没有这个值了。**运行时照样跑** ——
    ``ChannelSpec`` 是普通 dataclass，不做校验 —— 于是错得很安静；
    但类型检查直接报错（``Literal['deepseek'] is not assignable to ClientKind``），
    而且「目录和代码对不上」正是这套改造要消灭的东西。

    ⇒ 这条守住「目录里写的每个值都有人声明过、每个声明过的值都有人用」。
    """
    from typing import get_args

    from minitap.mobile_use.services.channels import ClientKind
    from minitap.mobile_use.services.llm import _OPENAI_CLIENT_KINDS

    declared = set(get_args(ClientKind))
    for cid, spec in CHANNELS.items():
        if spec.protocol == "openai":
            assert spec.client_kind in declared, f"{cid}: {spec.client_kind!r} 不在 ClientKind 里"
    # 声明过的每个 kind 都得真的能落到一个客户端类上。
    assert declared == set(_OPENAI_CLIENT_KINDS)


def test_thinking_aware_channels_share_one_client_class():
    """DeepSeek 和 MiMo 行为**完全相同** ⇒ 共用一个客户端类。

    差别只在渠道目录的 ``thinking_default_on`` 里（DeepSeek 默认开思考；
    MiMo 实测服务端默认也开思考 —— 2026-10-24 实测 reasoning_tokens 非零）。
    两个名字指同一份行为 = 两个所有者，迟早走样。
    """
    from minitap.mobile_use.services.llm import _OPENAI_CLIENT_KINDS

    assert CHANNELS["deepseek"].client_kind == "thinking_aware"
    assert CHANNELS["mimo-token-plan"].client_kind == "thinking_aware"
    assert _OPENAI_CLIENT_KINDS["thinking_aware"] == "ThinkingAwareChatOpenAI"
    assert "deepseek" not in _OPENAI_CLIENT_KINDS


def test_unknown_channel_is_reported_with_the_known_list():
    with pytest.raises(ValueError, match="unknown LLM channel"):
        require_channel("deepseesk")


def test_no_channel_carries_a_dead_field():
    """目录里不能有「代码从不读」的字段。

    留一个没人读的字段，目录就会和代码对不上 —— 而对不上的症状是
    「目录里写着 False，代码却按 True 跑」，最难查。
    """
    from dataclasses import fields

    from minitap.mobile_use.services.llm import (
        _OPENAI_CLIENT_KINDS,
        build_openai_protocol_client,
    )

    read_fields = {
        "id",
        "label",
        "protocol",
        "base_url",
        "models",
        "api_key_env",
        "base_url_env",
        "requires_api_key",
        "supports_endpoint",
        "thinking_style",
        "thinking_default_on",
        "thinking_env",
        "thinking_fallback",
        "session_query_param",
        "effort_style",
        "effort_levels",
        "effort_default",
        "temperature_range",
        "structured_output",
        "max_tokens_policy",
        "client_kind",
        "requires_base_url",
        "note",
        "request_timeout",
        "max_retries",
        "retry_rules",
        "stream_usage",
    }
    declared = {f.name for f in fields(CHANNELS["openai"])}
    assert declared == read_fields, f"目录里多/少了字段：{declared ^ read_fields}"
    # client_kind 必须真的能被客户端工厂用上。
    for cid, spec in CHANNELS.items():
        if spec.protocol == "openai":
            assert spec.client_kind in _OPENAI_CLIENT_KINDS, cid
    assert build_openai_protocol_client is not None


# ---------------------------------------------------------------------------
# 2. 默认端点：渠道不给就用目录里的那份
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cid", ["deepseek", "minimax", "siliconflow", "openai", "mimo-token-plan"])
def test_channel_without_url_uses_the_catalog_default(cid):
    """渠道没给端点 → 用目录里的默认端点，且**不读** .env。

    这是「默认配置好的 URL」这层价值的落点：用户不用手抄地址。
    """
    cfg = llm_config_for_single_model(cid, "some-model", api_key="k")
    client = get_llm(_ctx(cfg), name="cortex")
    assert client.openai_api_base == default_base_url(cid)


def test_custom_channel_without_a_url_is_rejected():
    """「自定义（OpenAI 兼容）」没有默认地址 —— 不给端点必须报错。

    放行的症状是「请求打到了 OpenAI 官方端点」，而错误信息完全指不到真正的原因。
    """
    with pytest.raises(ValueError, match="base_url"):
        build_openai_protocol_client(
            require_channel("custom"), "some-model", 1, ProviderOverride(api_key=SecretStr("k"))
        )


# ---------------------------------------------------------------------------
# 3. 渠道的「要不要思考」必须真的生效
# ---------------------------------------------------------------------------


def test_deepseek_channel_thinking_off_reaches_the_request_body():
    """🔴 渠道把思考关掉 → 请求体里必须真的有 ``thinking.disabled``。

    以前这里只读 ``settings.DEEPSEEK_THINKING``，**完全无视渠道下发的值**。
    症状：manager 里明明选了「直答」，模型还在思考 → 而 DeepSeek 思考模式下
    强制 tool_choice 是硬 400 → 每个 agent 节点全挂。
    """
    client = get_deepseek_llm(
        model_name="deepseek-flash",
        override=_channel_override("deepseek", "deepseek-flash", thinking=False),
    )
    assert client.extra_body == {"thinking": {"type": "disabled"}}


def test_deepseek_channel_thinking_on_reaches_the_request_body():
    client = get_deepseek_llm(
        model_name="deepseek-flash",
        override=_channel_override("deepseek", "deepseek-flash", thinking=True),
    )
    assert client.extra_body == {"thinking": {"type": "enabled"}}


def test_deepseek_channel_without_thinking_sends_nothing():
    """没指定 → 不发这个字段 → 走 DeepSeek 自己的默认（推理模型默认开思考）。"""
    client = get_deepseek_llm(
        model_name="deepseek-flash",
        override=_channel_override("deepseek", "deepseek-flash"),
    )
    assert not client.extra_body


def test_minimax_channel_thinking_off_reaches_the_request_body():
    client = get_minimax_llm(
        model_name="MiniMax-M3",
        override=_channel_override("minimax", "MiniMax-M3", thinking=False),
    )
    # MiniMax 还会带 output_config.effort（思考深度），所以只看 thinking 那一项。
    assert client.extra_body["thinking"] == {"type": "disabled"}


def test_llamacpp_channel_thinking_switch_reaches_the_request_body():
    """llama.cpp 的形状不一样：``chat_template_kwargs.enable_thinking``。"""
    client = get_llamacpp_llm(
        model_name="m",
        override=_channel_override("llamacpp", "m", thinking=True),
    )
    assert client.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


def test_llamacpp_channel_defaults_to_thinking_off():
    """没指定 → 关思考。

    开着时 content 恒为空，结构化输出必然失败（实测：思考 ON + max_tokens=256
    → JSON 没写完；不限长 → 直接挂死）。
    """
    client = get_llamacpp_llm(
        model_name="m",
        override=_channel_override("llamacpp", "m"),
    )
    assert client.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_channel_thinking_is_not_overridden_by_the_local_env(monkeypatch):
    """渠道说了就算 —— 设备本地的 .env 不能改写它。

    渠道是 manager 明确指定的东西，.env 是设备上残留的旧配置。后者赢的话，
    症状是「我在界面上关了思考，它还在思考」，而且不报错。
    """
    from minitap.mobile_use.config import settings

    monkeypatch.setattr(settings, "DEEPSEEK_THINKING", True, raising=False)
    client = get_deepseek_llm(
        model_name="deepseek-flash",
        override=_channel_override("deepseek", "deepseek-flash", thinking=False),
    )
    assert client.extra_body == {"thinking": {"type": "disabled"}}


# ---------------------------------------------------------------------------
# 3b. 思考强度（DeepSeek 的 reasoning_effort）必须真的生效
# ---------------------------------------------------------------------------


def test_deepseek_channel_reasoning_effort_reaches_the_request_body():
    """渠道把强度设成 max → 请求体里必须真的有 ``reasoning_effort: max``。

    只断言客户端的字段不够：字段声明了却没进请求体，症状和没配置一样。
    """
    from langchain_core.messages import HumanMessage

    client = get_deepseek_llm(
        model_name="deepseek-flash",
        override=_channel_override("deepseek", "deepseek-flash", reasoning_effort="MAX"),
    )
    assert client.reasoning_effort == "max"
    payload = client._get_request_payload([HumanMessage(content="hi")])
    assert payload["reasoning_effort"] == "max"


def test_deepseek_channel_without_effort_sends_nothing():
    """没指定 → 不发 → 走服务端默认档位（DeepSeek 是 high）。"""
    client = get_deepseek_llm(
        model_name="deepseek-flash",
        override=_channel_override("deepseek", "deepseek-flash"),
    )
    assert client.reasoning_effort is None


def test_reasoning_effort_is_dropped_when_thinking_is_off():
    """思考都关了，强度没有意义 —— 不能发一对矛盾参数。"""
    client = get_deepseek_llm(
        model_name="deepseek-flash",
        override=_channel_override(
            "deepseek", "deepseek-flash", thinking=False, reasoning_effort="max"
        ),
    )
    assert client.reasoning_effort is None


def test_unknown_reasoning_effort_fails_loudly():
    """写错档位必须报错。

    静默回落的症状是「选了低档，花的还是高档的 token」，没有任何地方能看出来。
    """
    from minitap.mobile_use.services.llm import _resolve_reasoning_effort

    with pytest.raises(ValueError, match="reasoning_effort"):
        _resolve_reasoning_effort(get_channel("deepseek"), None, "xhigh")


def test_reasoning_effort_is_ignored_by_channels_without_the_knob():
    """目录里没有这个旋钮的渠道收到它也**不发**（不是报错）。"""
    client = get_minimax_llm(
        model_name="MiniMax-M3",
        override=_channel_override("minimax", "MiniMax-M3", reasoning_effort="max"),
    )
    assert client.reasoning_effort is None


# ---------------------------------------------------------------------------
# 4. 「能不能强制 tool_choice」由这次请求的思考状态决定
# ---------------------------------------------------------------------------


def test_deepseek_does_not_force_tool_choice_while_thinking():
    """思考开着 → 不能强制 tool_choice（DeepSeek 直接 400）。

    2026-10-24 实测：deepseek-flash + 强制 tool_choice = HTTP 400
    「Thinking mode does not support this tool_choice」。
    """
    client = get_deepseek_llm(
        model_name="deepseek-flash",
        override=_channel_override("deepseek", "deepseek-flash"),
    )
    assert client.should_force_tool_choice() is False


def test_deepseek_forces_tool_choice_once_thinking_is_off():
    client = get_deepseek_llm(
        model_name="deepseek-flash",
        override=_channel_override("deepseek", "deepseek-flash", thinking=False),
    )
    assert client.should_force_tool_choice() is True


def test_executor_path_does_not_force_tool_choice_for_a_thinking_deepseek():
    """🔴 executor / reporter 走的是 ``get_structured_llm``，不是 ``with_structured_output``。

    以前那条路恒为 ``force_tool_choice=True`` —— 于是 DeepSeek 思考模式下
    每个 executor 调用都 400。这是「deepseek 一直在请求失败」的直接原因。

    渠道没指定思考 → 走 DeepSeek 的默认（推理模型默认**开**思考）→ 不能强制。
    """
    cfg = llm_config_for_single_model("deepseek", "deepseek-flash", api_key="k")
    runner = get_structured_llm_for(cfg)
    assert "tool_choice" not in runner._bound.kwargs


def get_structured_llm_for(cfg):
    """借 executor 那条路造一个结构化输出 runner（不跑真请求）。"""
    from minitap.mobile_use.agents.executor.types import ExecutorOutput
    from minitap.mobile_use.services.llm import get_structured_llm

    return get_structured_llm(ctx=_ctx(cfg), name="executor", schema=ExecutorOutput)


def test_executor_path_forces_tool_choice_once_thinking_is_off():
    """思考关掉之后，强制是安全的 —— 该强制就强制，模型更听话。"""
    cfg = llm_config_for_single_model("deepseek", "deepseek-flash", api_key="k", thinking=False)
    runner = get_structured_llm_for(cfg)
    assert "tool_choice" in runner._bound.kwargs


def test_mimo_does_not_force_tool_choice_when_thinking_is_not_requested():
    """🔴 MiMo 服务端**默认开着思考** ⇒ 没显式关掉时不能强制 tool_choice。

    2026-10-24 实测（token-plan-cn.xiaomimimo.com，mimo-v2.6-pro）：不下发
    ``thinking`` 时 usage 里 ``reasoning_tokens`` = 13~17、message 里有
    ``reasoning_content`` ⇒ 服务端默认就是开思考。

    以前这里断言「可以强制」，依据是「MiMo 的思考是按需开启的」—— 与实测相反。
    后果：框架在思考**实际开着**的时候强制 tool_choice ⇒ 预算一紧就
    ``finish_reason=length`` + **没有 tool_calls**（静默失败，不是 400）。

    ⇒ 与 DeepSeek 同规则：没说 = 按开着算 ⇒ 不强制。
    """
    cfg = llm_config_for_single_model("mimo-token-plan", "mimo-v2.6-pro", api_key="k")
    runner = get_structured_llm_for(cfg)
    assert "tool_choice" not in runner._bound.kwargs


def test_mimo_forces_tool_choice_once_thinking_is_off():
    """显式关掉思考后强制是安全的 —— 而且更省。

    实测：thinking=disabled ⇒ ``reasoning_tokens`` = 0、``completion_tokens``
    从 ~35 降到 17，工具调用 3/3 命中。
    """
    cfg = llm_config_for_single_model(
        "mimo-token-plan", "mimo-v2.6-pro", api_key="k", thinking=False
    )
    runner = get_structured_llm_for(cfg)
    assert "tool_choice" in runner._bound.kwargs


def test_mimo_thinking_switch_reaches_the_request_body():
    """思考开关必须真的进请求体（形状是 ``thinking.type``）。"""
    from minitap.mobile_use.services.llm import build_openai_protocol_client

    spec = require_channel("mimo-token-plan")
    client = build_openai_protocol_client(
        spec,
        "mimo-v2.6-pro",
        1,
        _channel_override("mimo-token-plan", "mimo-v2.6-pro", thinking=False),
    )
    assert client.extra_body == {"thinking": {"type": "disabled"}}


def test_mimo_declares_a_thinking_knob():
    """渠道必须给出思考开关的环境变量名，否则用户没地方调。

    以前 mimo 这条没写 ``thinking_env`` ⇒ ``MIMO_THINKING`` 没人读 ⇒
    「在 .env 里把思考关掉」这件事根本做不到（而且不报错）。
    """
    spec = require_channel("mimo-token-plan")
    assert spec.thinking_env == "MIMO_THINKING"
    assert spec.thinking_default_on is True, "服务端默认开着思考（2026-10-24 实测）"


def test_every_channel_env_var_is_declared_on_settings():
    """🔴 渠道目录里引用的每个环境变量名，``Settings`` 都必须声明过。

    这是「配置根本没接上」那一类静默失败的**唯一防线**。

    ``Settings.model_config`` 是 ``{"extra": "ignore"}`` ⇒ 没声明的变量名会被
    **静默丢弃**：``.env`` 里明明写了 ``MIMO_API_KEY=tp-…``，``settings`` 上却
    没有这个属性 ⇒ ``ProviderOverride.key_or()`` 回退成 ``None`` ⇒ 客户端拿
    占位 key ``"sk-local"`` 去请求 ⇒ **401 "Invalid API Key"**，
    而错误信息完全指不到真正的原因。

    2026-10-24 实测踩过：mimo 渠道配好了、key 也对，但 ``Settings`` 里漏了
    ``MIMO_API_KEY`` ⇒ 整条渠道 100% 401，且不写代码根本看不出来。
    """
    from minitap.mobile_use.config import Settings

    declared = set(Settings.model_fields)
    missing: list[str] = []
    for cid, spec in CHANNELS.items():
        for attr in (spec.api_key_env, spec.thinking_env, spec.base_url_env):
            if attr and attr not in declared:
                missing.append(f"{cid}: {attr}")
    assert not missing, (
        "渠道引用的环境变量没在 Settings 里声明（会被 extra=ignore 静默丢弃）: "
        + ", ".join(missing)
    )


def test_mimo_does_not_force_tool_choice_when_thinking_is_on():
    """MiMo 官方文档：开 thinking 后 tool_calls 输出不完整 ⇒ 不强制。"""
    cfg = llm_config_for_single_model(
        "mimo-token-plan", "mimo-v2.6-pro", api_key="k", thinking=True
    )
    runner = get_structured_llm_for(cfg)
    assert "tool_choice" not in runner._bound.kwargs


# ---------------------------------------------------------------------------
# 5. 小米 MiMo Token Plan 这条渠道本身
# ---------------------------------------------------------------------------


def test_mimo_token_plan_endpoint_and_models():
    """端点和模型名必须是**官方文档里的那一份**。

    2026-10-24 官方文档（mimo.mi.com/docs/zh-CN/tokenplan/Token Plan/quick-access）：
      OpenAI 兼容协议 中国集群 https://token-plan-cn.xiaomimimo.com/v1
      Anthropic 兼容协议 中国集群 https://token-plan-cn.xiaomimimo.com/anthropic
      个人版 key 格式 tp-xxxxx
    模型：mimo-v2.6-pro / mimo-v2.6-flash（另有 asr / tts 三款是语音）
    ⚠️ mimo-v2.5-pro、mimo-v2.5 已于 2026-10-21 下线，不能再列。
    """
    spec = get_channel("mimo-token-plan")
    assert spec is not None
    assert spec.protocol == "openai"
    assert spec.base_url == "https://token-plan-cn.xiaomimimo.com/v1"
    assert "mimo-v2.6-pro" in spec.models
    assert "mimo-v2.6-flash" in spec.models
    assert not any(m.startswith("mimo-v2.5") for m in spec.models), "v2.5 系列已下线"
    assert spec.requires_api_key is True


def test_mimo_token_plan_carries_the_terms_of_service_warning():
    """Token Plan 的 key 按小米条款只允许用于编程工具。

    这条限制必须跟着渠道走到界面上：用在自动化脚本/应用后端属于违反条款，
    平台可以暂停服务、封禁 key —— 而用户看到 401/封号时根本想不到是条款。
    """
    spec = get_channel("mimo-token-plan")
    assert spec.note is not None
    assert "编程工具" in spec.note


# ---------------------------------------------------------------------------
# 6. 导给 manager 侧的载荷
# ---------------------------------------------------------------------------


def test_catalog_payload_is_json_serialisable_and_has_what_the_form_needs():
    payload = catalog_payload()
    json.dumps(payload)  # 必须能序列化 —— manager 侧要把它当 JSON 存/传

    ids = [item["id"] for item in payload]
    assert ids == list(channel_ids())

    for item in payload:
        assert item["protocol"] in {"openai", "anthropic", "google", "vertexai", "azure"}
        assert isinstance(item["models"], list)
        assert isinstance(item["requiresApiKey"], bool)
        assert isinstance(item["supportsEndpoint"], bool)
        # 「默认地址」是这层改造的核心收益：添加渠道时不用手抄 URL。
        if item["id"] in {"deepseek", "minimax", "siliconflow", "openai", "mimo-token-plan"}:
            assert item["baseUrl"], item["id"]


def test_catalog_carries_the_reasoning_effort_levels_for_deepseek():
    """思考强度是**界面要渲染的选项**，所以必须跟着目录一起导出。

    这些值来自官方 ``GET /models``（2026-10-26）：low / high / max，默认 high。
    没这个旋钮的渠道给空列表 —— 前端据此决定要不要渲染那个下拉框。
    """
    by_id = {item["id"]: item for item in catalog_payload()}
    deepseek = by_id["deepseek"]
    assert deepseek["reasoningEffortLevels"] == ["low", "high", "max"]
    assert deepseek["reasoningEffortDefault"] == "high"

    assert by_id["minimax"]["reasoningEffortLevels"] == []
    assert by_id["minimax"]["reasoningEffortDefault"] is None


def test_catalog_model_lists_are_not_empty_for_the_channels_we_ship():
    """这几个渠道是要给人**选**模型的：模型名写错 → 请求打到一个不存在的模型。

    所以它们的推荐列表不能空。
    """
    for cid in ("deepseek", "minimax", "siliconflow", "openai", "anthropic", "mimo-token-plan"):
        assert get_channel(cid).models, cid


def test_catalog_models_match_the_live_platform_model_lists():
    """推荐列表里的模型名必须是**平台真的有的**那些。

    2026-10-24 实测：
      * ``GET https://api.deepseek.com/models`` → deepseek-flash / deepseek-v4-pro
      * ``GET https://api.minimaxi.com/v1/models`` → MiniMax-M3.1-Flash / MiniMax-M3 …
    这条断言挡不住「平台后来下架了某个模型」，但能挡住「手抄错了名字」。
    """
    deepseek = set(get_channel("deepseek").models)
    assert deepseek == {"deepseek-flash", "deepseek-v4-pro"}

    minimax = set(get_channel("minimax").models)
    assert "MiniMax-M3" in minimax
    assert "MiniMax-M3.1-Flash" in minimax
