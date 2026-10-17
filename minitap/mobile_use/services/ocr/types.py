"""OCR 感知的公共类型。

设计参照 MaaFramework 的 `Ocr` 任务：模型是可插拔的（`engine` + 模型目录），
结果是「文本 + 框 + 置信度」的扁平列表，由上层渲染成带索引的块喂给 Cortex。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from minitap.mobile_use.utils.ui_hierarchy import ElementBounds


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

    engine: str = Field(
        default="paddle_onnx",
        description="引擎名，对应 registry 里注册的 key（见 available_ocr_engines）",
    )
    model_dir: Path | None = Field(
        default=None,
        description="模型目录。None 时用 MOBILE_USE_OCR_MODEL_DIR，再退回仓库内 minitap/model/ocr",
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
