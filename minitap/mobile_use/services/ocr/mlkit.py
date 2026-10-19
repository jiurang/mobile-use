"""ML Kit 文字识别引擎（Android）。

为什么不是 PaddleOCR
-------------------
PyPI 上 OCR 运行时的 wheel 平台 tag 全都只有 manylinux / macosx / win，**没有
android**：onnxruntime、opencv-python、numpy、pyclipper 全都不提供。而
onnxruntime 自己的构建脚本 ``tools/ci_build/build.py`` 在 ``if args.android:
... continue`` 处把整个 wheel 阶段跳过了，官方根本不产出 Android Python wheel ——
想在手机上跑 PaddleOCR 只能手工用 NDK 交叉编译 pybind11 binding。

ML Kit 是 Google 维护的 Android 原生 OCR，AAR 直接链进 APK，零 NDK 交叉编译。
本模块是 ``MlKitOcr.java`` 的 Python 侧适配层，实现 :class:`OcrEngine` 协议，
通过 ``registry.register_ocr_engine`` 注册为 ``mlkit``。

与 PaddleOCR 的行为差异
--------------------
1. **置信度语义不同。** ML Kit 的 ``Text.Element.getConfidence()`` 在部分情况下返回
   ``0``（表示"没给"而不是"非常不可信"）。这里把 ``0`` 当作**未知**处理、回填成
   ``1.0``，否则 ``config.rec_min_score``（默认 0.6）会把这些结果全滤成空数组 ——
   而且滤掉的是本来正确的文字。用非 0 的分数则照常参与 ``rec_min_score`` 和
   ``max_texts`` 的按分截断，与 PC 端 paddle_onnx 的行为对齐。
2. **没有旋转多边形框。** ML Kit 给的是 ``Element.getBoundingBox()``（轴对齐矩形）
   和 4 个 ``cornerPoints``；这里用 ``boundingBox`` 填 ``OcrText.bounds``，
   因为 ``OcrText`` 本来就只存轴对齐的 :class:`ElementBounds`。

本模块在 PC 上不可用：``from java import jclass`` 只存在于 Chaquopy 运行时。
构造时抛 :class:`OcrDependenciesMissing`，由 ``registry.get_ocr_engine``
降级成 warning 并跳过本轮 OCR 感知 —— 这正是 ``services/ocr/registry.py``
docstring 承诺的「加新引擎不影响调用方」。
"""

from __future__ import annotations

import json
from typing import Any

from minitap.mobile_use.services.ocr.paddle_onnx import OcrDependenciesMissing
from minitap.mobile_use.services.ocr.types import OcrConfig, OcrText
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds

logger = get_logger(__name__)

#: 一次识别的上限。正常真机 100-400ms；Java 侧也有同样的兜底超时。
DEFAULT_TIMEOUT_MS = 10_000

#: ML Kit 拿不到分数时的占位值。见模块 docstring 的差异 1。
_UNKNOWN_SCORE = 1.0


class MlKitOcrEngine:
    """ML Kit 引擎。构造即验证运行环境是否可用。"""

    name = "mlkit"

    def __init__(self, config: OcrConfig | None = None):
        self._config = config or OcrConfig()
        self._bridge = self._load_bridge()

    @staticmethod
    def _load_bridge() -> Any:
        try:
            from java import jclass
        except ImportError as exc:
            raise OcrDependenciesMissing(
                "mlkit 引擎只能在 runner-apk（Chaquopy）里用："
                f"java 桥不可用（{exc}）。PC 上请用 paddle_onnx 引擎。"
            ) from exc
        try:
            return jclass("com.mobilemanager.runnerapk.MlKitOcr")
        except Exception as exc:  # noqa: BLE001 - jclass 的失败形态不统一
            raise OcrDependenciesMissing(
                "找不到 com.mobilemanager.runnerapk.MlKitOcr —— "
                f"APK 可能是旧的，没重新编译（{type(exc).__name__}: {exc}）"
            ) from exc

    def recognize(self, image_base64: str) -> list[OcrText]:
        """识别一张 base64 截图里的全部文字。"""
        raw = self._bridge.recognize(image_base64, DEFAULT_TIMEOUT_MS)
        payload = json.loads(raw)

        if isinstance(payload, dict):
            # Java 侧所有异常都收敛成 {"error": "..."}，绝不把 Java stack
            # trace 抛回 Python（跨语言边界的异常对调用方没有诊断价值）。
            raise RuntimeError(f"ML Kit OCR 失败：{payload.get('error', 'unknown')}")

        results: list[OcrText] = []
        unknown_scores = 0
        for item in payload:
            text = (item.get("text") or "").strip()
            if not text:
                continue
            raw_score = float(item.get("score") or 0.0)
            if raw_score > 0.0:
                score = raw_score
            else:
                # 0 在 ML Kit 里表示"没给分数"，不是"完全不可信"。
                unknown_scores += 1
                score = _UNKNOWN_SCORE
            if score < self._config.rec_min_score:
                continue
            w = max(1, int(item.get("w", 0)))
            h = max(1, int(item.get("h", 0)))
            results.append(
                OcrText(
                    text=text,
                    score=score,
                    bounds=ElementBounds(
                        x=int(item.get("x", 0)),
                        y=int(item.get("y", 0)),
                        width=w,
                        height=h,
                    ),
                )
            )

        if unknown_scores:
            logger.debug(
                f"ML Kit returned no confidence for {unknown_scores}/{len(payload)} texts; "
                "treated them as 1.0 so rec_min_score does not discard correct text"
            )

        # 按分数截断，与 paddle_onnx 一致；同分时保持 ML Kit 的阅读顺序稳定。
        if self._config.max_texts and len(results) > self._config.max_texts:
            logger.warning(
                f"ML Kit OCR found {len(results)} texts, keeping the top "
                f"{self._config.max_texts} by score"
            )
            results.sort(key=lambda t: -t.score)
            results = results[: self._config.max_texts]

        results.sort(key=lambda t: (t.bounds.y, t.bounds.x))
        logger.debug(f"ML Kit OCR: {len(results)} texts")
        return results


def register() -> None:
    """注册到全局 OCR 引擎表。由 runner_bootstrap 在启动时调用。"""
    from minitap.mobile_use.services.ocr.registry import register_ocr_engine

    register_ocr_engine(MlKitOcrEngine.name, MlKitOcrEngine)
