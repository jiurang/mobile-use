"""渠道下发(manager → runner → mobile-use)这条路径的回归测试。

写这些用例时踩过的坑,留给下一个人:

* 「没下发渠道时行为不变」是最重要的不变量,所以每个工厂都要有 settings 兜底用例,
  而不是只测新路径。
* hopper 的调用点硬编码 temperature=0(:mod:`minitap.mobile_use.agents.hopper`),
  渠道温度必须能盖掉它 —— 这正是本文件第 4 组用例存在的理由。
* Google / Vertex 的客户端**没有**端点参数。下发了就必须报出来,静默丢弃会变成
  「配置改了没反应」。

**渠道不许回退到设备本地的 .env。** manager 已经明确指定了用哪个模型、哪个端点、
哪把密钥,这时 mobile-use 再从 .env 里补一把,症状是「我明明选了 M3.1,跑起来是 .env
里那个模型」—— 而且它不报错。所以:

* 渠道缺密钥 → 构造配置时直接报错;
* 渠道没给端点 → 用 provider 自带的默认端点,但**不读** .env 里的
  ``<PROVIDER>_BASE_URL``;
* 「没下发渠道」的老路(``llm-config.override.jsonc`` + .env)行为一字不变。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from minitap.mobile_use.config import LLM, LLMConfig, settings
from minitap.mobile_use.sdk.utils import llm_config_for_single_model
from minitap.mobile_use.services.llm import (
    ProviderOverride,
    get_google_llm,
    get_llm,
    get_minimax_llm,
)


@pytest.fixture
def no_settings_minimax(monkeypatch):
    """把 settings 里 MiniMax 的东西全清空。

    清空是故意的:只有这样,「下发的 key 生效」才不是被 env 巧合满足的。
    """
    monkeypatch.setattr(settings, "MINIMAX_API_KEY", None, raising=False)
    monkeypatch.setattr(settings, "MINIMAX_BASE_URL", None, raising=False)


@pytest.fixture
def settings_minimax(monkeypatch):
    monkeypatch.setattr(settings, "MINIMAX_API_KEY", SecretStr("env-key"), raising=False)
    monkeypatch.setattr(settings, "MINIMAX_BASE_URL", "https://env.example.com/v1", raising=False)


def _ctx(llm_config):
    """get_llm 只读 ctx.llm_config 和 ctx.trace_id。

    造一个真的 MobileUseContext 需要塞一整套 device 上下文,跟这里要测的东西
    无关,还会让用例变慢。
    """
    return SimpleNamespace(llm_config=llm_config, trace_id="t-1")


# ---------------------------------------------------------------------------
# 1. LLM 这个数据模型本身
# ---------------------------------------------------------------------------


def test_llm_carries_the_three_channel_fields():
    node = LLM(provider="minimax", model="MiniMax-M3.1-Flash-Preview", base_url="http://x/v1")
    assert node.base_url == "http://x/v1"
    assert node.api_key is None
    assert node.temperature is None


def test_llm_str_does_not_leak_the_key():
    node = LLM(provider="minimax", model="M3.1", api_key="sk-super-secret")
    assert "sk-super-secret" not in str(node)


def test_validate_provider_accepts_inline_key_when_env_is_empty(no_settings_minimax):
    """settings 里没有 key,但配置带了 —— 必须放行。

    这条是「渠道下发」能成立的前提:key 由 manager 管理,设备上不一定有 env。
    """
    node = LLM(provider="minimax", model="M3.1", api_key="sk-from-manager")
    node.validate_provider("Cortex")  # 不抛异常即通过


def test_validate_provider_still_rejects_a_missing_key(no_settings_minimax):
    """两处都没有 key —— 仍然必须报错,不能因为新增了渠道就放松校验。"""
    node = LLM(provider="minimax", model="M3.1")
    with pytest.raises(Exception, match="MINIMAX_API_KEY"):
        node.validate_provider("Cortex")


# ---------------------------------------------------------------------------
# 2. 单渠道 → 完整 LLMConfig
# ---------------------------------------------------------------------------


def test_single_model_config_covers_every_role_and_fallback():
    cfg = llm_config_for_single_model("minimax", "M3.1", api_key="k", base_url="http://x/v1")

    nodes = [
        cfg.planner,
        cfg.orchestrator,
        cfg.contextor,
        cfg.cortex,
        cfg.executor,
        cfg.utils.outputter,
        cfg.utils.hopper,
        cfg.utils.video_analyzer,
    ]
    assert len(nodes) == 8
    for node in nodes:
        assert node.provider == "minimax"
        assert node.model == "M3.1"
        assert node.api_key is not None
        assert node.base_url == "http://x/v1"
        # fallback 也得是同一条渠道,否则主渠道失败时会落回设备上的默认模型
        assert node.fallback.model == "M3.1"
        assert node.fallback.api_key is not None


def test_single_model_config_omits_unset_optional_fields():
    """没传的字段必须真的不存在,而不是塞成 None。

    `LLMWithFallback.model_validate` 会走 validate_provider,塞 None 和不塞的
    区别在于:不塞时温度退回各调用点的硬编码值。
    """
    cfg = llm_config_for_single_model("minimax", "M3.1", api_key="k")
    assert cfg.cortex.temperature is None
    assert cfg.cortex.base_url is None
    assert cfg.cortex.api_key is not None


def test_channel_without_key_is_rejected():
    """渠道缺密钥必须报错，而不是回退到设备本地的 .env。

    这条是「manager 下发的模型参数不被 .env 顶掉」的底线：回退的症状是
    「我明明选了 M3.1，跑起来是 .env 里那个模型」，而且它不报错。
    """
    with pytest.raises(ValueError, match="api_key"):
        llm_config_for_single_model("minimax", "M3.1")


def test_channel_config_is_marked_no_env_fallback():
    """渠道产出的配置必须标上「不许从 .env 补齐」。

    标记丢了，strict 就不会打开，兜底会悄悄回来 —— 而这种回归没有任何症状，
    只能靠这个断言挡住。
    """
    cfg = llm_config_for_single_model("minimax", "M3.1", api_key="k")
    assert cfg.cortex.allow_env_fallback is False
    assert cfg.cortex.fallback.allow_env_fallback is False


def test_keyless_provider_may_omit_the_key():
    """azure / vertexai 走服务账号凭据，没有密钥是合法的。"""
    cfg = llm_config_for_single_model("vertexai", "gemini-2.5-pro")
    assert cfg.cortex.api_key is None
    assert cfg.cortex.allow_env_fallback is False


# ---------------------------------------------------------------------------
# 3. 下发优先、settings 兜底
# ---------------------------------------------------------------------------


def test_channel_credentials_win_over_settings(settings_minimax):
    cfg = llm_config_for_single_model(
        "minimax", "M3.1", api_key="channel-key", base_url="http://channel/v1"
    )
    client = get_llm(_ctx(cfg), name="cortex")

    assert client.openai_api_key is not None
    assert client.openai_api_key.get_secret_value() == "channel-key"
    assert client.openai_api_base == "http://channel/v1"


def test_file_config_still_falls_back_to_settings(settings_minimax):
    """没下发渠道的部署必须一字不变 —— 否则这是破坏性改动。

    「没下发渠道」的老路是配置文件（``llm-config.override.jsonc``，里面只有
    provider/model）+ .env 里的密钥。渠道那条路（:func:`llm_config_for_single_model`）
    不再回落，见 :func:`test_channel_without_key_is_rejected`。
    """
    def role(model: str = "M3.1"):
        return {
            "provider": "minimax",
            "model": model,
            "fallback": {"provider": "minimax", "model": "M3"},
        }

    cfg = LLMConfig.model_validate(
        {
            "planner": role(),
            "orchestrator": role(),
            "contextor": role(),
            "cortex": role(),
            "executor": role(),
            "utils": {"outputter": role(), "hopper": role()},
        }
    )
    client = get_llm(_ctx(cfg), name="cortex")

    assert client.openai_api_key is not None
    assert client.openai_api_key.get_secret_value() == "env-key"
    assert client.openai_api_base == "https://env.example.com/v1"


def test_channel_does_not_read_the_endpoint_from_env(settings_minimax):
    """渠道没给端点时，用 provider 自带的默认端点，**不读** .env 里的那个。

    读 .env 意味着「manager 说用 minimax，端点却是设备本地 .env 里那个」——
    国内 key 发到国际端点是 401，而错误信息完全指不到真正的原因。
    """
    cfg = llm_config_for_single_model("minimax", "M3.1", api_key="channel-key")
    client = get_llm(_ctx(cfg), name="cortex")

    assert client.openai_api_base == "https://api.minimax.io/v1"


def test_channel_key_works_with_empty_settings(no_settings_minimax):
    """settings 全空时,只有下发这一条路能让它跑起来。"""
    cfg = llm_config_for_single_model("minimax", "M3.1", api_key="channel-key")
    dispatched = get_llm(_ctx(cfg), name="executor")
    assert dispatched.openai_api_key is not None
    assert dispatched.openai_api_key.get_secret_value() == "channel-key"


def test_fallback_channel_is_used_when_requested(settings_minimax):
    cfg = llm_config_for_single_model(
        "minimax", "M3.1", api_key="channel-key", base_url="http://channel/v1"
    )
    client = get_llm(_ctx(cfg), name="cortex", use_fallback=True)
    assert client.openai_api_base == "http://channel/v1"


# ---------------------------------------------------------------------------
# 4. 温度
# ---------------------------------------------------------------------------


def test_channel_temperature_overrides_call_site_value(settings_minimax):
    """hopper 的调用点传 temperature=0,渠道给 0.7 时必须是 0.7。"""
    cfg = llm_config_for_single_model("minimax", "M3.1", api_key="k", temperature=0.7)
    client = get_llm(_ctx(cfg), name="hopper", is_utils=True, temperature=0)
    assert client.temperature == pytest.approx(0.7)


def test_call_site_temperature_kept_when_channel_silent(settings_minimax):
    cfg = llm_config_for_single_model("minimax", "M3.1", api_key="k")
    client = get_llm(_ctx(cfg), name="hopper", is_utils=True, temperature=0)
    assert client.temperature == pytest.approx(0)


def test_minimax_still_clamps_temperature(no_settings_minimax):
    """渠道传来的温度也要走 MiniMax 的 0..1 钳制,不能绕过。

    钳制在 `get_minimax_llm` 里,不在 get_llm,所以必须直接打工厂。
    """
    client = get_minimax_llm(
        "M3.1",
        5.0,
        override=ProviderOverride(api_key=SecretStr("k")),
    )
    assert client.temperature == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. 下发了但用不上 —— 必须报出来
# ---------------------------------------------------------------------------


def test_unsupported_base_url_is_reported_not_dropped(monkeypatch, caplog):
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", SecretStr("g-key"), raising=False)
    with caplog.at_level("WARNING", logger="minitap.mobile_use.services.llm"):
        get_google_llm("gemini-2.5-pro", 0.7, override=ProviderOverride(base_url="http://x"))

    assert any("base_url" in r.message for r in caplog.records)
