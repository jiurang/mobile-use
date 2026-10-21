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
    """
    provider = spec.get("provider")
    model = spec.get("model")
    if not provider or not model:
        raise ValueError(f"an LLM spec needs both provider and model, got: {dict(spec)!r}")
    entry: dict[str, Any] = {"provider": provider, "model": model}
    for key in ("base_url", "api_key", "temperature"):
        value = spec.get(key)
        if value is not None:
            entry[key] = value
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
) -> LLMConfig:
    """把「一条渠道」铺成一份完整的 LLMConfig —— 每个角色都用它。

    存在的意义:runner(PC 或 apk 里跑的那份)拿到 manager 下发的渠道参数后,不该
    自己知道 mobile-use 有 planner/cortex/executor/... 这 8 个角色名,更不该漏掉
    其中一个 —— 漏掉的那个会静默落回默认模型,是最难查的一类故障。所以角色清单留在
    这一侧维护。

    没有下发渠道的调用方不要碰这个函数,继续用 :func:`initialize_llm_config`。

    Args:
        provider: LLMProvider 的取值,如 "minimax"。
        model: 模型名。
        base_url: 端点。None 表示沿用 settings / 官方默认。
        api_key: 密钥。None 表示沿用 settings。
        temperature: 温度。None 表示沿用各 agent 调用点的硬编码值。
    """
    return llm_config_for_role_overrides(
        {
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "temperature": temperature,
        }
    )


def llm_config_for_role_overrides(
    base: Mapping[str, Any],
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> LLMConfig:
    """把「一条基础渠道 + 按角色的覆盖」铺成一份完整的 LLMConfig。

    和 :func:`llm_config_for_single_model` 的区别只有一处：**每个角色可以指向
    不同的模型**。基础渠道填满全部 8 个角色，``overrides`` 里点名的角色改用自己的
    接入参数，其余角色照旧。

    为什么覆盖必须自带**完整**的接入参数（provider + model + 可选的 key/端点/温度），
    而不是「同一个渠道换个模型名」：角色之间经常要跨供应商 —— 视觉密集的 cortex 走
    Gemini、其余走 MiniMax 是最常见的一种。只给模型名不给供应商，那个角色就会静默
    落回基础渠道的供应商，症状是「以为换了模型，其实在同一个模型上跑」。

    Args:
        base: 基础接入参数，键为 provider / model / base_url / api_key / temperature。
        overrides: 角色名 → 该角色专用的接入参数。键必须是 :data:`ALL_ROLE_NAMES`
            里的名字；写错会直接报错，而不是静默忽略 —— 静默忽略的症状正是
            「我给 cortex 指定了模型，它还在用默认的」。

    Returns:
        一份 8 个角色全部有值的 LLMConfig。
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
        utils=LLMConfigUtils(
            outputter=_llm_node(spec_for("outputter")),
            hopper=_llm_node(spec_for("hopper")),
            video_analyzer=_llm_node(spec_for("video_analyzer")),
        ),
    )
