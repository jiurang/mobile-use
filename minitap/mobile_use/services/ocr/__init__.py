"""OCR 感知：可插拔的 OCR 引擎。

内置引擎：`paddle_onnx`（百度 PaddleOCR 的 det + rec + 字典）。
换一个模型只需要换 `OcrConfig.model_dir`，换一个实现只需要 `register_ocr_engine`。
"""

from minitap.mobile_use.services.ocr.registry import (
    available_ocr_engines,
    clear_ocr_engine_cache,
    create_ocr_engine,
    get_ocr_engine,
    register_ocr_engine,
)
from minitap.mobile_use.services.ocr.types import OcrConfig, OcrEngine, OcrText

__all__ = [
    "OcrConfig",
    "OcrEngine",
    "OcrText",
    "available_ocr_engines",
    "clear_ocr_engine_cache",
    "create_ocr_engine",
    "get_ocr_engine",
    "register_ocr_engine",
]
