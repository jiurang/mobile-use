"""OCR 引擎注册表 —— 「自选 OCR 模型」的落点。

想加一个新引擎（比如云端 OCR、MaaFramework 桥接、别的 PaddleOCR 权重）时：

    from minitap.mobile_use.services.ocr import OcrConfig, register_ocr_engine

    def factory(config: OcrConfig) -> MyEngine:
        return MyEngine(config)

    register_ocr_engine("my_engine", factory)

调用方不需要改。

引擎实例是**按配置缓存**的：onnxruntime 的 InferenceSession 加载两个模型
要几百毫秒，每轮重建会把 OCR 的开销翻好几倍。缓存 key 包含了模型目录和
会影响推理行为的参数。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from minitap.mobile_use.services.ocr.types import (
    OcrConfig,
    OcrEngine,
    resolve_device,
    resolve_rec_batch,
)
from minitap.mobile_use.utils.logger import get_logger

if TYPE_CHECKING:
    from minitap.mobile_use.context import MobileUseContext

logger = get_logger(__name__)

OcrEngineFactory = Callable[[OcrConfig], OcrEngine]

_REGISTRY: dict[str, OcrEngineFactory] = {}
_CACHE: dict[tuple, OcrEngine] = {}


def register_ocr_engine(name: str, factory: OcrEngineFactory) -> None:
    """注册一个 OCR 引擎。重复注册同名会覆盖（便于测试打桩）。"""
    _REGISTRY[name] = factory


def available_ocr_engines() -> list[str]:
    return sorted(_REGISTRY)


def _cache_key(config: OcrConfig) -> tuple:
    # 注意用的是 `resolve_device` / `resolve_rec_batch` 的**解析结果**，
    # 不是原始字段：走 MOBILE_USE_OCR_DEVICE 环境变量时 `config.device` 是 None，
    # `cpu` 和 `cuda` 两种环境设置会算出
    # **同一个 key**，于是第二次直接拿到上一次的引擎实例 —— 设了 cpu 却仍在
    # CUDA 上跑，且没有任何报错。实测踩过。
    #
    # key 还要覆盖全部会影响这个实例行为的字段。漏一个就等于「改配置没反应
    # 但也不报错」，是最难查的一类。
    return (
        config.engine,
        str(config.model_dir) if config.model_dir else None,
        resolve_device(config),
        config.device_id,
        config.det_limit_side_len,
        config.det_thresh,
        config.det_box_thresh,
        config.det_unclip_ratio,
        config.rec_min_score,
        resolve_rec_batch(config),
        config.max_texts,
    )


def create_ocr_engine(config: OcrConfig | None = None) -> OcrEngine:
    """按配置造一个引擎实例（带缓存）。"""
    config = config or OcrConfig()

    factory = _REGISTRY.get(config.engine)
    if factory is None:
        known = ", ".join(available_ocr_engines()) or "(空)"
        raise ValueError(f"未知的 OCR 引擎 '{config.engine}'。已注册：{known}")

    key = _cache_key(config)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    engine = factory(config)
    _CACHE[key] = engine
    return engine


def get_ocr_engine(ctx: MobileUseContext) -> OcrEngine | None:
    """按 context 里的感知配置取引擎。

    引擎加载失败（依赖没装、模型文件不在）时**只警告并返回 None**：
    OCR 是增强感知，不该把整个 agent 搞挂 —— 层级树和截图还在。
    """
    config = ctx.perception.ocr_config
    try:
        return create_ocr_engine(config)
    except Exception as e:
        logger.warning(f"OCR 不可用，本轮跳过 OCR 感知：{type(e).__name__}: {e}")
        return None


def clear_ocr_engine_cache() -> None:
    """清空实例缓存。主要给测试用。"""
    _CACHE.clear()


def _register_builtin_engines() -> None:
    """内置引擎的注册要延迟到真正使用时 —— 顶层 import 会把重量级依赖拖进来。"""

    def paddle_factory(config: OcrConfig) -> OcrEngine:
        from minitap.mobile_use.services.ocr.paddle_onnx import PaddleOcrOnnxEngine

        return PaddleOcrOnnxEngine(config)

    _REGISTRY.setdefault("paddle_onnx", paddle_factory)


_register_builtin_engines()
