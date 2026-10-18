"""把 OCR 结果渲染成 Cortex 读取的紧凑索引块，并负责把索引还原回 OCR 结果。

和 `ui_hierarchy_view.py` 是同一套思路：原始 OCR 输出是几十条
`{text, score, bounds}`，直接 dump 进 prompt 是纯浪费。这里按阅读顺序编号、
每行只留模型真正要用的三个字段。

**渲染和还原必须同源，所以两者都住在这个模块。** `render_ocr_results` 打的
`[i]` 和 `resolve_ocr_index` 吃的 `index` 是同一个契约的两端：前者把
`results[i-1]` 印成 `[i]`，后者原样还原。任何一边改了编号方式，另一边立刻错。

必须守住的不变量
----------------
1. **索引就是工具参数。** `Target.ocr_index` 直接吃这个 `[i]`，
   1-based（模型抄 `[3]` 就写 `ocr_index=3`，不做任何算术）。
   越界不静默 —— 返回 None，调用方必须把「这个索引不存在」告诉模型。
2. **不去重、不重排。** 识别到什么就渲染什么，编号就是原始阅读顺序里的位置。
   曾经按「层级树里已经出现过」剔过重复项，结果编号整体前移：模型手里的
   `[3]` 和原始第 3 条对不上，而它无从得知。这是给提示层做优化时最容易
   顺手破坏工具契约的地方 —— 省下的那点 token 不值这个风险。
3. **框必须是原图坐标**。OCR 跑在原分辨率截图上，`x,y,WxH` 可以直接填进
   `Target.bounds`。一旦换成压缩图坐标，点位会整体偏移。
4. **置信度要透出**。低于阈值的已经被引擎丢掉了，剩下的里 `0.67` 和 `0.99`
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
    """渲染 OCR 文本块；没有结果时返回空串（调用方据此不发消息）。

    编号从 1 开始、连续、不跳号，且与 `results` 的下标严格对应
    （`[i]` 就是 `results[i-1]`）。这条契约由 `resolve_ocr_index` 兑现。

    Args:
        results: OCR 引擎的输出，已按阅读顺序排列。
    """
    if not results:
        return ""

    logger.debug(f"OCR view: rendering {len(results)} texts")
    lines = [
        f"[{index}] {json.dumps(_truncate(item.text), ensure_ascii=False)} "
        f"({item.bounds.x},{item.bounds.y},{item.bounds.width}x{item.bounds.height}) "
        f"{item.score:.2f}"
        for index, item in enumerate(results, 1)
    ]
    return f"{_HEADER}\n" + "\n".join(lines)


def resolve_ocr_index(
    results: list[OcrText] | None, index: int | None
) -> OcrText | None:
    """把 `Target.ocr_index` 还原成 OCR 结果。**1-based**，与渲染出的 `[i]` 一致。

    越界或没有结果时返回 None —— 调用方要据此报错，不能退化成「点别的地方」。

    Args:
        results: 本屏 OCR 结果，必须和渲染时用的是同一份。
        index: 1-based 行号，也就是提示里印出来的那个 `[i]`。
    """
    if index is None:
        return None
    if not results:
        logger.warning(f"Cannot resolve ocr_index={index}: no OCR results on this screen")
        return None
    if index < 1 or index > len(results):
        logger.warning(
            f"ocr_index={index} out of range; the OCR block had {len(results)} line(s) "
            f"(valid 1..{len(results)})"
        )
        return None
    return results[index - 1]
