"""把 OCR 结果渲染成 Cortex 读取的紧凑索引块。

和 `ui_hierarchy_view.py` 是同一套思路：原始 OCR 输出是几十条
`{text, score, bounds}`，直接 dump 进 prompt 是纯浪费。这里按阅读顺序编号、
每行只留模型真正要用的三个字段。

实测成本（Redmi Turbo 4, 1220x2712, 抖音视频屏）：
    23 条文本 -> 576 tokens；同期 UI 层级树裁剪块 4070 tokens。
也就是说 OCR 感知大约是层级树之外的 +14% token，换来的是视频/自绘 UI
上那 7 条层级树完全没有的文本。

2026-10-17 在抖音搜索页实测：25 条 OCR 文本里 **19 条是层级树里已经有的**
（"6.8万"、"《侠盗猎车手：圣安地列斯》"、"全屏观看" …），只有 6 条是树里没有的，
而那 6 条全是误识别噪声（"搜素"、"TtheFo"、"Sanandmrag"）。也就是说这一屏上
OCR 的 642 tokens 换来的增量信息接近于零。加上 `already_rendered` 去重之后，
这一屏 OCR 只剩 6 条 / 约 130 tokens。

必须守住的不变量
----------------
1. **索引是给人看的**，不是工具契约。`Target` 里没有 index 字段，模型要用
   OCR 结果得把 `text` 和 `bounds` 抄进 target。
2. **框必须是原图坐标**。OCR 跑在原分辨率截图上，`x,y,WxH` 可以直接填进
   `Target.bounds`。一旦换成压缩图坐标，点位会整体偏移。
3. **置信度要透出**。低于阈值的已经被引擎丢掉了，剩下的里 `0.67` 和 `0.99`
   的可信度不一样，模型该看得见差别。
4. **去重只能对着「已渲染的树」做**。`already_rendered` 传进来的必须是
   `ui_hierarchy_view.render_hierarchy_and_labels()` 的第二返回值，也就是
   真正进了 prompt 的那批 label —— 不是原始 dump 的全集。树的过滤和这里的
   去重是先后作用的两道筛子：某条文字所在的节点如果被树过滤掉了，它就已经
   不在 prompt 里了，这时再拿「原始树里有它」当理由从 OCR 里剔掉，这条
   文字就两边都没了。只做完全一致匹配（归一化去空白+小写后相等），不做
   子串包含 —— 子串匹配会把 "512" 和 "6.8万" 之类误判成同一条，砍掉真信息。
"""

from __future__ import annotations

import json

from minitap.mobile_use.services.ocr.types import OcrText
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.ui_hierarchy_view import _normalize_label

logger = get_logger(__name__)

_HEADER = 'Text on screen (OCR), one per line: [i] "text" (x,y,WxH) conf\n'

# 单条文本超过这个长度就截断。一整屏评论那种长文本会把 token 撑爆，
# 而模型点它靠的是坐标和开头几个字。
MAX_TEXT_CHARS = 80


def _truncate(text: str) -> str:
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return text[: MAX_TEXT_CHARS - 1] + "…"


def render_ocr_results(
    results: list[OcrText] | None,
    already_rendered: set[str] | None = None,
) -> str:
    """渲染 OCR 文本块；没有结果时返回空串（调用方据此不发消息）。

    Args:
        results: OCR 引擎的输出。
        already_rendered: 已经出现在 UI 层级树渲染块里的 label（**归一化后**，
            见 `ui_hierarchy_view._normalize_label`）。这些文字在 prompt 里
            已经出现过一遍，再抄一遍只是重复烧 token —— 抖音搜索页实测 25 条
            里有 19 条属于这种情况。命中就整条丢掉。
    """
    if not results:
        return ""

    kept: list[OcrText] = []
    dropped = 0
    for item in results:
        if already_rendered and item.text and _normalize_label(item.text) in already_rendered:
            dropped += 1
            continue
        kept.append(item)

    if dropped:
        logger.debug(f"OCR view: dropped {dropped} texts already present in the UI hierarchy")

    if not kept:
        return ""

    lines = []
    for index, item in enumerate(kept, 1):
        b = item.bounds
        lines.append(
            f"[{index}] {json.dumps(_truncate(item.text), ensure_ascii=False)} "
            f"({b.x},{b.y},{b.width}x{b.height}) {item.score:.2f}"
        )

    logger.debug(f"OCR view: {len(kept)}/{len(results)} texts rendered")
    return f"{_HEADER}\n" + "\n".join(lines)

