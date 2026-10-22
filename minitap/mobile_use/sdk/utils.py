import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from minitap.mobile_use.config import (
    LLMConfig,
    LLMConfigUtils,
    LLMWithFallback,
    deep_merge_llm_config,
    get_default_llm_config,
    provider_requires_api_key,
)
from minitap.mobile_use.utils.file import load_jsonc
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

# 角色名清单留在 mobile-use 这一侧维护：runner 不该知道 mobile-use 有哪些角色，
# 更不该漏掉其中一个 —— 漏掉的那个会静默落回默认模型，是最难查的一类故障。
AGENT_ROLE_NAMES: tuple[str, ...] = (
    "planner",
    "orchestrator",
    "contextor",
    "cortex",
    "executor",
    "reporter",
)
UTILS_ROLE_NAMES: tuple[str, ...] = (
    "outputter",
    "hopper",
    "video_analyzer",
)
ALL_ROLE_NAMES: tuple[str, ...] = AGENT_ROLE_NAMES + UTILS_ROLE_NAMES


def load_llm_config_override(path: Path) -> LLMConfig:
    default_config = get_default_llm_config()

    override_config_dict = {}
    if os.path.exists(path):
        logger.info(f"Loading custom LLM config from {path.resolve()}...")
        with open(path, encoding="utf-8") as f:
            override_config_dict = load_jsonc(f)
    else:
        logger.warning("Custom LLM config not found - using the default config")

    try:
        return deep_merge_llm_config(default_config, override_config_dict)
    except ValidationError as e:
        logger.error(f"Invalid LLM config: {e}")
        logger.info("Falling back to default config")
        return default_config


def _llm_entry(spec: Mapping[str, Any]) -> dict[str, Any]:
    """把一份接入参数收敛成 LLM 的入参：没给的项**不出现**，而不是塞成 None。

    「不出现」和「None」在这里是两种语义：前者表示「沿用设备本地的既有配置 /
    各调用点的硬编码值」，后者是同义的显式写法。两种都对，但保持老行为
    （:func:`llm_config_for_single_model` 一直是不出现）能少一类「行为悄悄变了」。

    **这条路径是「渠道」路径，密钥不是可选项。** 这里的入参来自 manager
    运行期下发的渠道：manager 已经明确指定了用哪个模型、哪个端点、哪把密钥。
    缺密钥时如果放行，mobile-use 会从设备本地的 .env 里补一把 —— 症状是
    「我明明选了 M3.1，跑起来是 .env 里那个模型」，而且**它不报错**。
    所以缺密钥在这里直接拒绝，报清楚缺的是哪一项。

    端点（base_url）不同：不给就用该 provider 自带的默认端点（代码里的常量），
    但同样不会去读 .env —— 那由 :attr:`LLM.allow_env_fallback` 保证。
    """
    provider = spec.get("provider")
    model = spec.get("model")
    if not provider or not model:
        raise ValueError(f"an LLM spec needs both provider and model, got: {dict(spec)!r}")
    api_key = spec.get("api_key")
    if not api_key and provider_requires_api_key(provider):
        raise ValueError(
            f"渠道缺少 api_key（provider={provider}, model={model}）："
            f"渠道必须自带密钥，不允许回退到设备本地的 .env。"
            f"请在 manager 的渠道配置里补上 API Key。"
        )
    entry: dict[str, Any] = {"provider": provider, "model": model}
    for key in ("base_url", "api_key", "temperature", "thinking"):
        value = spec.get(key)
        if value is not None:
            entry[key] = value
    # 运行期下发的渠道：不再从设备本地的环境补齐缺失项。见 LLM.allow_env_fallback。
    entry["allow_env_fallback"] = False
    return entry


def _llm_node(spec: Mapping[str, Any]) -> LLMWithFallback:
    """一个角色用的接入节点，fallback 指向它自己。

    fallback 必须是**同一个**接入点：主渠道失败时落回设备上的默认模型，
    症状是「我明明选了 M3.1，跑起来不是」——那种问题查起来要花很久。
    """
    entry = _llm_entry(spec)
    return LLMWithFallback.model_validate({**entry, "fallback": dict(entry)})


def llm_config_for_single_model(
    provider: str,
    model: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    thinking: bool | None = None,
) -> LLMConfig:
    """把「一条渠道」铺成一份完整的 LLMConfig —— 每个角色都用它。

    存在的意义:runner(PC 或 apk 里跑的那份)拿到 manager 下发的渠道参数后,不该
    自己知道 mobile-use 有 planner/cortex/executor/... 这 9 个角色名,更不该漏掉
    其中一个 —— 漏掉的那个会静默落回默认模型,是最难查的一类故障。所以角色清单留在
    这一侧维护。

    没有下发渠道的调用方不要碰这个函数,继续用 :func:`initialize_llm_config`。

    Args:
        provider: LLMProvider 的取值,如 "minimax"。
        model: 模型名。
        base_url: 端点。None 表示用该 provider 自带的默认端点（**不读 .env**）。
        api_key: 密钥。**provider 需要密钥时必填**（azure / vertexai 除外）——
            None 不再表示「沿用 settings」：渠道下发时缺密钥会静默落到设备本地的
            .env，症状是「我明明选了 A，跑起来是 B」且不报错。
        temperature: 温度。None 表示沿用各 agent 调用点的硬编码值。
        thinking: 「要不要思考」。None 表示沿用该 provider 自己的默认。
            llama.cpp 和 minimax 消费它（MiniMax 那边只有 MiniMax-M3 真能关，
            M3.1 / M2.x 官方强制思考、关不掉）。

    Raises:
        ValueError: provider/model 缺失，或 provider 需要密钥而 api_key 为空。
    """
    return llm_config_for_role_overrides(
        {
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "temperature": temperature,
            "thinking": thinking,
        }
    )


def llm_config_for_role_overrides(
    base: Mapping[str, Any],
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> LLMConfig:
    """把「一条基础渠道 + 按角色的覆盖」铺成一份完整的 LLMConfig。

    和 :func:`llm_config_for_single_model` 的区别只有一处：**每个角色可以指向
    不同的模型**。基础渠道填满全部 9 个角色，``overrides`` 里点名的角色改用自己的
    接入参数，其余角色照旧。

    为什么覆盖必须自带**完整**的接入参数（provider + model + 可选的 key/端点/温度），
    而不是「同一个渠道换个模型名」：角色之间经常要跨供应商 —— 视觉密集的 cortex 走
    Gemini、其余走 MiniMax 是最常见的一种。只给模型名不给供应商，那个角色就会静默
    落回基础渠道的供应商，症状是「以为换了模型，其实在同一个模型上跑」。

    **这条路径产出的配置不回退到设备本地的 .env。** 密钥缺失在这里就报错，
    端点缺失用 provider 自带的默认端点；两者都不会去读 .env。原因见
    :func:`_llm_entry` 和 :attr:`~minitap.mobile_use.config.LLM.allow_env_fallback`。

    Args:
        base: 基础接入参数，键为 provider / model / base_url / api_key / temperature。
        overrides: 角色名 → 该角色专用的接入参数。键必须是 :data:`ALL_ROLE_NAMES`
            里的名字；写错会直接报错，而不是静默忽略 —— 静默忽略的症状正是
            「我给 cortex 指定了模型，它还在用默认的」。

    Returns:
        一份 9 个角色全部有值的 LLMConfig。

    Raises:
        ValueError: 角色名不认识，或某个接入参数缺 provider / model / 必需的密钥。
    """
    role_map = dict(overrides or {})
    unknown = sorted(set(role_map) - set(ALL_ROLE_NAMES))
    if unknown:
        raise ValueError(
            "unknown LLM role(s): "
            + ", ".join(unknown)
            + "; known roles are: "
            + ", ".join(ALL_ROLE_NAMES)
        )

    def spec_for(role: str) -> Mapping[str, Any]:
        return role_map.get(role, base)

    return LLMConfig(
        planner=_llm_node(spec_for("planner")),
        orchestrator=_llm_node(spec_for("orchestrator")),
        contextor=_llm_node(spec_for("contextor")),
        cortex=_llm_node(spec_for("cortex")),
        executor=_llm_node(spec_for("executor")),
        reporter=_llm_node(spec_for("reporter")),
        utils=LLMConfigUtils(
            outputter=_llm_node(spec_for("outputter")),
            hopper=_llm_node(spec_for("hopper")),
            video_analyzer=_llm_node(spec_for("video_analyzer")),
        ),
    )
