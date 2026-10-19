import os
from pathlib import Path

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
    entry: dict[str, object] = {"provider": provider, "model": model}
    if base_url is not None:
        entry["base_url"] = base_url
    if api_key is not None:
        entry["api_key"] = api_key
    if temperature is not None:
        entry["temperature"] = temperature

    node = LLMWithFallback.model_validate({**entry, "fallback": dict(entry)})
    return LLMConfig(
        planner=node,
        orchestrator=node,
        contextor=node,
        cortex=node,
        executor=node,
        utils=LLMConfigUtils(
            outputter=node,
            hopper=node,
            video_analyzer=node,
        ),
    )
