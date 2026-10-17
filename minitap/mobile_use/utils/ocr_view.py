"""把 OCR 结果渲染成 Cortex 读取的紧凑索引块。

和 `ui_hierarchy_view.py` 是同一套思路：原始 OCR 输出是几十条
`{text, score, bounds}`，直接 dump 进 prompt 是纯浪费。这里按阅读顺序编号、
每行只留模型真正要用的三个字段。

实测成本（Redmi Turbo 4, 1220x2712, 抖音视频屏）：
    23 条文本 -> 576 tokens；同期 UI 层级树裁剪块 4070 tokens。
也就是说 OCR 感知大约是层级树之外的 +14% token，换来的是视频/自绘 UI
上那 7 条层级树完全没有的文本。

必须守住的不变量
----------------
1. **索引是给人看的**，不是工具契约。`Target` 里没有 index 字段，模型要用
   OCR 结果得把 `text` 和 `bounds` 抄进 target。
2. **框必须是原图坐标**。OCR 跑在原分辨率截图上，`x,y,WxH` 可以直接填进
   `Target.bounds`。一旦换成压缩图坐标，点位会整体偏移。
3. **置信度要透出**。低于阈值的已经被引擎丢掉了，剩下的里 `0.67` 和 `0.99`
   的可信度不一样，模型该看得见差别。
"""

from __future__ import annotations

import json

from minitap.mobile_use.services.ocr.types import OcrText
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

_HEADER = 'Text on screen (OCR), one per line: [i] "text" (x,y,WxH) conf\n'

# 单条文本超过这个长度就截断。一整屏评论那种长文本会把 token 撑爆，
# 而模型点它靠的是坐标和开头几个字。
MAX_TEXT_CHARS = 80


def _truncate(text: str) -> str:
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return text[: MAX_TEXT_CHARS - 1] + "…"


def render_ocr_results(results: list[OcrText] | None) -> str:
    """渲染 OCR 文本块；没有结果时返回空串（调用方据此不发消息）。"""
    if not results:
        return ""

    lines = []
    for index, item in enumerate(results, 1):
        b = item.bounds
        lines.append(
            f"[{index}] {json.dumps(_truncate(item.text), ensure_ascii=False)} "
            f"({b.x},{b.y},{b.width}x{b.height}) {item.score:.2f}"
        )

    logger.debug(f"OCR view: {len(results)} texts rendered")
    return f"{_HEADER}\n" + "\n".join(lines)
