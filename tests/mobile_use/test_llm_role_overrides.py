"""按角色覆盖模型（渠道 × 角色）这条路径的回归测试。

写这些用例时盯着的三件事：

* **没覆盖的角色必须还是基础渠道。** 漏一个角色的症状是「我给 cortex 换了模型，
  它没换，但 executor 莫名其妙也换了」——两个方向都难查。
* **覆盖必须跨供应商生效。** 现实里最常见的是 cortex 走视觉更强的 Gemini、
  其余走 MiniMax。只给模型名不给供应商，那个角色会静默落回基础渠道的供应商。
* **角色名写错要报错。** 静默忽略等于「我指定了但没生效」，而且没有任何日志。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from minitap.mobile_use.config import settings
from minitap.mobile_use.sdk.utils import (
    ALL_ROLE_NAMES,
    llm_config_for_role_overrides,
    llm_config_for_single_model,
)
from minitap.mobile_use.services.llm import get_llm


@pytest.fixture
def no_settings_minimax(monkeypatch):
    """把 settings 里 MiniMax 的东西全清空。

    清空是故意的：只有这样，「下发的 key 生效」才不是被 env 巧合满足的。
    """
    monkeypatch.setattr(settings, "MINIMAX_API_KEY", None, raising=False)
    monkeypatch.setattr(settings, "MINIMAX_BASE_URL", None, raising=False)


def _ctx(llm_config):
    return SimpleNamespace(llm_config=llm_config, trace_id="t-1")


BASE = {
    "provider": "minimax",
    "model": "MiniMax-M3.1-Flash-Preview",
    "base_url": "https://api.minimax.io/v1",
    "api_key": "base-key",
}


# ---------------------------------------------------------------------------
# 1. 基础渠道仍然铺满全部角色
# ---------------------------------------------------------------------------


def test_base_channel_still_covers_every_role():
    """单渠道的语义一个字都没变：10 个角色全用基础渠道。"""
    cfg = llm_config_for_role_overrides(BASE)

    nodes = [
        cfg.planner,
        cfg.orchestrator,
        cfg.contextor,
        cfg.cortex,
        cfg.executor,
        cfg.reporter,
        cfg.vision,
        cfg.utils.outputter,
        cfg.utils.hopper,
        cfg.utils.video_analyzer,
    ]
    assert len(nodes) == 10
    for node in nodes:
        assert node is not None
        assert node.provider == "minimax"
        assert node.model == "MiniMax-M3.1-Flash-Preview"
        assert node.api_key is not None
        assert node.base_url == "https://api.minimax.io/v1"
        # fallback 也得是同一条渠道，否则主渠道失败时会落回设备上的默认模型
        assert node.fallback.model == "MiniMax-M3.1-Flash-Preview"
        assert node.fallback.api_key is not None


def test_single_model_helper_is_the_same_as_base_without_overrides():
    """两个入口必须给出同一份配置，否则「换了个入口行为就变了」是最难查的一类。"""
    single = llm_config_for_single_model(
        "minimax",
        "MiniMax-M3.1-Flash-Preview",
        base_url="https://api.minimax.io/v1",
        api_key="base-key",
    )
    based = llm_config_for_role_overrides(BASE)

    assert single.model_dump() == based.model_dump()


def test_all_role_names_matches_the_config_shape():
    """角色清单和 LLMConfig 的形状必须对得上：少一个就有一个角色落回默认模型。

    2026-10-21：新增 `reporter`（Executor 自主执行改造的汇报角色，
    见 doc/executor-autonomy-plan.md §2.1.1）。
    2026-10-28：新增 `vision`（`Target.vision` 的视觉定位小调用）——
    manager 下发渠道时要能单独指定它，否则 grounding 模型换不了。
    """
    assert set(ALL_ROLE_NAMES) == {
        "planner",
        "orchestrator",
        "contextor",
        "cortex",
        "executor",
        "reporter",
        "vision",
        "outputter",
        "hopper",
        "video_analyzer",
    }
    assert len(ALL_ROLE_NAMES) == 10


# ---------------------------------------------------------------------------
# 2. 覆盖：只改点名的角色
# ---------------------------------------------------------------------------


def test_override_replaces_only_the_named_role():
    cfg = llm_config_for_role_overrides(
        BASE,
        {"cortex": {"provider": "minimax", "model": "MiniMax-M2.7", "api_key": "cortex-key"}},
    )

    assert cfg.cortex.model == "MiniMax-M2.7"
    assert cfg.cortex.api_key is not None
    assert cfg.cortex.api_key.get_secret_value() == "cortex-key"
    # 其余角色原样不动
    assert cfg.executor.model == "MiniMax-M3.1-Flash-Preview"
    assert cfg.planner.model == "MiniMax-M3.1-Flash-Preview"
    assert cfg.utils.hopper.model == "MiniMax-M3.1-Flash-Preview"


def test_override_carries_its_own_provider_and_endpoint():
    """跨供应商覆盖：cortex 走另一个 provider 时，端点和密钥都是它自己的。"""
    cfg = llm_config_for_role_overrides(
        BASE,
        {
            "cortex": {
                "provider": "openai",
                "model": "gpt-5.2",
                "base_url": "http://gw.internal/v1",
                "api_key": "other-key",
            }
        },
    )

    assert cfg.cortex.provider == "openai"
    assert cfg.cortex.model == "gpt-5.2"
    assert cfg.cortex.base_url == "http://gw.internal/v1"
    assert cfg.cortex.api_key is not None
    assert cfg.cortex.api_key.get_secret_value() == "other-key"
    # fallback 必须跟着覆盖走，不能还指着基础渠道
    assert cfg.cortex.fallback.model == "gpt-5.2"
    assert cfg.cortex.fallback.api_key is not None
    assert cfg.cortex.fallback.api_key.get_secret_value() == "other-key"


def test_override_without_key_is_rejected():
    """覆盖角色不自带密钥时必须报错，不能沿用 settings。

    回落 .env 的症状是「我给 cortex 选了 M2.7，它跑起来用的是 .env 里那个模型
    的密钥」—— 而 .env 里那把密钥的模型可能根本不是 M2.7。
    """
    monkey = pytest.MonkeyPatch()
    monkey.setattr(settings, "MINIMAX_API_KEY", SecretStr("env-key"), raising=False)
    try:
        with pytest.raises(ValueError, match="api_key"):
            llm_config_for_role_overrides(
                BASE,
                {"cortex": {"provider": "minimax", "model": "MiniMax-M2.7"}},
            )
    finally:
        monkey.undo()


def test_override_without_key_does_not_silently_use_the_base_key():
    """最坏的一种情况：覆盖角色缺密钥，却悄悄用上了基础渠道的密钥。

    这时「我给 cortex 换了模型」在密钥层面根本没换过 —— 症状和「换了但没生效」
    长得一模一样，所以必须在这里挡死。
    """
    with pytest.raises(ValueError, match="api_key"):
        llm_config_for_role_overrides(
            BASE,
            {"cortex": {"provider": "minimax", "model": "MiniMax-M2.7"}},
        )


def test_unknown_role_name_is_rejected():
    """角色名写错必须报错，不能静默忽略。"""
    with pytest.raises(ValueError, match="unknown LLM role"):
        llm_config_for_role_overrides(BASE, {"cortexx": {"provider": "minimax", "model": "M2.7"}})


def test_spec_without_provider_or_model_is_rejected():
    """半截的接入参数必须报错，而不是拼出一个「有模型名没供应商」的怪东西。"""
    with pytest.raises(ValueError, match="provider and model"):
        llm_config_for_role_overrides(BASE, {"cortex": {"model": "MiniMax-M2.7"}})


# ---------------------------------------------------------------------------
# 3. 运行期真的按角色取到不同的模型
# ---------------------------------------------------------------------------


def test_get_llm_returns_a_different_model_per_role(no_settings_minimax):
    """这是整个功能的存在理由：同一个 Agent 里，不同角色拿到不同的模型。"""
    cfg = llm_config_for_role_overrides(
        BASE,
        {"cortex": {"provider": "minimax", "model": "MiniMax-M2.7", "api_key": "cortex-key"}},
    )

    cortex = get_llm(_ctx(cfg), name="cortex")
    executor = get_llm(_ctx(cfg), name="executor")

    assert cortex.model_name == "MiniMax-M2.7"
    assert executor.model_name == "MiniMax-M3.1-Flash-Preview"


def test_get_llm_uses_the_overridden_key_for_that_role_only(no_settings_minimax):
    """覆盖角色的密钥也只作用于那个角色。"""
    cfg = llm_config_for_role_overrides(
        BASE,
        {"cortex": {"provider": "minimax", "model": "MiniMax-M2.7", "api_key": "cortex-key"}},
    )

    cortex = get_llm(_ctx(cfg), name="cortex")
    executor = get_llm(_ctx(cfg), name="executor")

    assert cortex.openai_api_key is not None
    assert cortex.openai_api_key.get_secret_value() == "cortex-key"
    assert executor.openai_api_key is not None
    assert executor.openai_api_key.get_secret_value() == "base-key"


def test_override_temperature_applies_to_that_role_only(no_settings_minimax):
    """温度是角色级的：hopper 的调用点写死 0，覆盖它时该生效；没覆盖的不动。"""
    cfg = llm_config_for_role_overrides(
        BASE,
        {
            "hopper": {
                "provider": "minimax",
                "model": "MiniMax-M3.1-Flash-Preview",
                "api_key": "hopper-key",
                "temperature": 0.7,
            },
        },
    )

    hopper = get_llm(_ctx(cfg), name="hopper", is_utils=True, temperature=0)
    cortex = get_llm(_ctx(cfg), name="cortex", temperature=1)

    assert hopper.temperature == pytest.approx(0.7)
    # 没覆盖的 cortex 仍然用调用点的温度
    assert cortex.temperature == pytest.approx(1)


def test_fallback_of_an_overridden_role_points_at_itself(no_settings_minimax):
    """use_fallback 拿到的是覆盖角色的接入参数，不是基础渠道的。"""
    cfg = llm_config_for_role_overrides(
        BASE,
        {
            "cortex": {
                "provider": "minimax",
                "model": "MiniMax-M2.7",
                "base_url": "http://fallback.example.com/v1",
                "api_key": "cortex-key",
            }
        },
    )

    client = get_llm(_ctx(cfg), name="cortex", use_fallback=True)
    assert client.openai_api_base == "http://fallback.example.com/v1"
    assert client.openai_api_key is not None
    assert client.openai_api_key.get_secret_value() == "cortex-key"
