"""OCR 感知的公共类型。

设计参照 MaaFramework 的 `Ocr` 任务：模型是可插拔的（`engine` + 模型目录），
结果是「文本 + 框 + 置信度」的扁平列表，由上层渲染成带索引的块喂给 Cortex。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds

# device 名 -> onnxruntime provider 名。放在这里而不是 paddle_onnx，
# 因为 registry 的实例缓存 key 也要用它 —— 环境变量路径下 `config.device`
# 是 None，不解析就拿来当 key 的话，`MOBILE_USE_OCR_DEVICE=cpu` 和
# `=cuda` 会算出**同一个 key**，第二次直接拿到上一次的缓存实例。
DEVICE_PROVIDERS: dict[str, str] = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
}

# rec 的批处理方式。必须显式枚举 —— 之前是裸 `str`，`"SERIAL"` 这种大小写
# 不同的写法会掉进 else 分支被当成 pad，**静默改变识别结果**。
REC_BATCH_MODES: tuple[str, ...] = ("serial", "pad", "resize")

logger = get_logger(__name__)


def _normalize(raw: str | None, allowed: tuple[str, ...], fallback: str, field: str) -> str:
    """小写 + 去空格 + 校验；不认识就 warn 后回落，而不是静默当别的方式用。

    静默回落在这里特别危险：批处理方式变了，**识别结果会跟着变**（标点、
    长句都可能读出不同文本），而用户从头到尾看不到任何提示。
    """
    value = (raw or "").strip().lower()
    if not value:
        return fallback
    if value in allowed:
        return value
    logger.warning(f"未知的 OCR {field}={raw!r}，按 {fallback!r} 处理。可选：{', '.join(allowed)}")
    return fallback


def resolve_engine(config: OcrConfig) -> str:
    """按 配置 -> 环境变量 -> paddle_onnx 的顺序定下引擎名。

    和 :func:`resolve_device` / :func:`resolve_rec_batch` 同一套优先级。
    runner-apk 就是靠 ``MOBILE_USE_OCR_ENGINE=mlkit`` 切到 ML Kit 的 ——
    感知配置（``runner_host.models.PerceptionConfig``）只有三个 bool 开关，
    契约里没有"引擎"这个概念，所以环境变量是唯一的注入点，不改协议就能换引擎。

    这里**不校验**取值是否合法：合法值由运行时注册表决定（PC 上没有 ``mlkit``，
    校验会误报），未知名由 ``registry.create_ocr_engine`` 报错并列出已注册项，
    那条错误信息比这里硬编码的列表准确。
    """
    import os

    raw = config.engine
    if raw is None:
        raw = os.getenv("MOBILE_USE_OCR_ENGINE")
    return (raw or "").strip() or "paddle_onnx"


def resolve_device(config: OcrConfig) -> str:
    """按 配置 -> 环境变量 -> auto 的顺序定下后端名。

    和 `resolve_model_dir` 同一套优先级，方便不改代码切后端
    （`MOBILE_USE_OCR_DEVICE=cuda`）。返回值一定是 `DEVICE_PROVIDERS` 里的
    key 或 `"auto"`。

    **谁要用「最终生效的后端」都必须过这个函数**，不要直接读 `config.device` ——
    那在走环境变量时是 `None`。
    """
    import os

    device = config.device
    if device is None:
        device = os.getenv("MOBILE_USE_OCR_DEVICE")
    # 先 strip 再判空 —— 反过来的话 `MOBILE_USE_OCR_DEVICE="   "` 这种
    # 全空格的设置会 truthy 通过 `or`，strip 完变成空串而不是回落到 auto。
    return (device or "").strip().lower() or "auto"


def resolve_rec_batch(config: OcrConfig) -> str:
    """按 配置 -> 环境变量 -> pad 的顺序定下批处理方式，保证是合法枚举值。"""
    import os

    raw = config.rec_batch
    if raw is None:
        raw = os.getenv("MOBILE_USE_OCR_REC_BATCH")
    return _normalize(raw, REC_BATCH_MODES, "pad", "rec_batch")


class OcrText(BaseModel):
    """一条 OCR 识别结果。

    `bounds` 复用 UI 层级树的 `ElementBounds`，这样 OCR 结果能直接填进
    `Target.bounds`，不需要在工具层做坐标形状转换。
    """

    text: str
    score: float = Field(description="识别置信度，0~1")
    bounds: ElementBounds


class OcrConfig(BaseModel):
    """OCR 引擎配置。

    `engine` + `model_dir` 就是「自选 OCR 模型」的落点：换引擎或换权重目录
    即可，不需要改调用方。
    """

    engine: str | None = Field(
        default=None,
        description=(
            "引擎名，对应 registry 里注册的 key（见 available_ocr_engines）。"
            "None 时读 MOBILE_USE_OCR_ENGINE 环境变量，再退回 paddle_onnx —— "
            "走 resolve_engine() 拿最终生效值，不要直接读这个字段。"
        ),
    )
    model_dir: Path | None = Field(
        default=None,
        description="模型目录。None 时用 MOBILE_USE_OCR_MODEL_DIR，再退回仓库内 minitap/model/ocr",
    )
    device: str | None = Field(
        default=None,
        description=(
            "推理后端：auto / cuda / dml / cpu。None 时读 MOBILE_USE_OCR_DEVICE，"
            "再退回 auto。auto 只探 CUDA，不探 DML —— DirectML 实测比 CPU 慢"
            "（见 paddle_onnx 模块 docstring），且要额外装包，不该被 auto 悄悄选中。"
        ),
    )
    device_id: int = Field(
        default=0,
        description="GPU 序号，多卡机器上指定用哪张",
    )

    # det（DB）参数，取值与 PaddleOCR 默认一致
    det_limit_side_len: int = Field(
        default=1280,
        description=(
            "检测输入的长边上限。真机实测 1220x2712 下：736→22 框、960→25 框、"
            "1280→28 框、2688→29 框，差异不大；1280 是延迟与召回的折中。"
        ),
    )
    det_thresh: float = Field(default=0.3, description="二值化阈值")
    det_box_thresh: float = Field(default=0.6, description="文本框置信度下限")
    det_unclip_ratio: float = Field(default=1.5, description="DB 框外扩比例")

    rec_min_score: float = Field(
        default=0.5,
        description="单条文本的识别置信度下限，低于此值丢弃",
    )
    rec_batch: str | None = Field(
        default=None,
        description=(
            "rec 的批处理方式：serial / pad / resize。None 时读 "
            "MOBILE_USE_OCR_REC_BATCH，再退回 pad。"
            "改造前是逐框 run()，一屏 26 框就是 26 次调用，**这才是瓶颈**。"
            "pad（默认）每行保持与 serial 完全一致的几何，右侧补黑到组内最大宽度，"
            "CPU 约 1.5x、CUDA 约 5.5x；resize 把所有行拉到统一宽度，矮框被额外"
            "横向拉长、长句更容易读错；serial 与改造前逐字一致、最慢。"
            "GPU 下不批量化反而比 CPU 慢 4 倍（每次 run 固定开销 ~40ms vs CPU 6.2ms）。"
        ),
    )
    max_texts: int = Field(
        default=80,
        description="最多保留多少条文本，防止满屏文字把 token 撑爆",
    )


@runtime_checkable
class OcrEngine(Protocol):
    """OCR 引擎协议。

    任何实现了这两个成员的类都可以注册进来当作 OCR 模型用。
    """

    name: str

    def recognize(self, image_base64: str) -> list[OcrText]:
        """识别一张截图里的全部文本。

        Args:
            image_base64: 截图的 base64，可带 `data:image/...;base64,` 前缀。

        Returns:
            按阅读顺序（先上后下、先左后右）排列的文本列表。
        """
        raise NotImplementedError
