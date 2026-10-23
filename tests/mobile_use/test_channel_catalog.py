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


def test_unknown_channel_is_reported_with_the_known_list():
    with pytest.raises(ValueError, match="unknown LLM channel"):
        require_channel("deepseesk")


# ---------------------------------------------------------------------------
# 2. 默认端点：渠道不给就用目录里的那份
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cid", ["deepseek", "minimax", "siliconflow", "openai"])
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


# ---------------------------------------------------------------------------
# 5. 导给 manager 侧的载荷
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
        if item["id"] in {"deepseek", "minimax", "siliconflow", "openai"}:
            assert item["baseUrl"], item["id"]


def test_catalog_model_lists_are_not_empty_for_the_channels_we_ship():
    """这几个渠道是要给人**选**模型的：模型名写错 → 请求打到一个不存在的模型。

    所以它们的推荐列表不能空。
    """
    for cid in ("deepseek", "minimax", "siliconflow", "openai", "anthropic"):
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
