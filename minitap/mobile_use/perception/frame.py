"""感知帧：一轮 Contextor 抓到的「这一屏」—— 截图、元素表、OCR、视觉主体的唯一载体。

为什么要有帧
------------
改造前，同一屏有四份互不相认的表示：Contextor 抓的截图只给 OCR 用完就丢、
元素表留在 state、视觉主体由工具另截一张图后自己算指纹、失效判定又回退到
元素表指纹。于是「编号只对当屏有效」这句话没有可验证的锚点：视觉模型看的是
A 图，编号却按 B 屏的元素表登记。

帧把四样东西绑成一个对象：

* ``screen_elements`` / ``ocr_results`` / ``screenshot_base64`` 来自**同一次**
  ``get_screen_data``，不再有第二次抓图；
* ``epoch`` 由文字通道内容派生，是这一帧的身份（同屏同值，屏变即变）；
* ``vision_elements`` 只属于这一帧。新帧 ``epoch`` 相同才允许继承（屏幕没变，
  扫描结果还成立），不同就清空。

三条不变量（改这块之前先读）
----------------------------
1. **一帧一图**：视觉扫描永远读 ``frame.screenshot_base64``，不自己截屏。
2. **索引只对帧成立**：``index`` / ``ocr_index`` / ``vision_index`` 都以帧上的
   数据为解析依据；``vision_index`` 的反解和渲染同在本模块，编号契约只有一份。
3. **帧回答它被展示的那一屏**：模型基于帧 A 决策，工具就扫帧 A。Contextor 抓帧
   到工具调用之间屏幕若变化，由下一帧的 ``epoch`` 负责作废，而不是让工具私自
   换一张更「新」的图 —— 那会让元素表编号和视觉编号来自两屏。

帧为什么不进 state
------------------
截图是 1~3MB 的 base64。state 是图的状态通道，每步都要流动；把图塞进去会让
每次状态更新都背着它，也容易被 trace / 录制路径顺手序列化。帧是**一次运行的
感知上下文**，和 ``ctx.task_started_at`` 同类，放在 ``MobileUseContext`` 上。
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time

from pydantic import BaseModel, Field

from minitap.mobile_use.services.ocr.types import OcrText
from minitap.mobile_use.utils.screen_elements import (
    ScreenElement,
    element_table_fingerprint,
    is_noise_text,
)
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds

#: 没有文字通道可用来派生身份时的前缀。这种帧每轮都是新身份，视觉列表不继承
#: —— 此时视觉是唯一通道，必须每轮重扫，否则模型会一直看着上几屏的旧列表。
_EMPTY_EPOCH_PREFIX = "empty"

#: 空帧身份里的进程内序号。``time.time_ns()`` 在 Windows 上实测有同值返回，
#: 光靠时间戳不足以保证「每轮唯一」。
_EMPTY_EPOCH_SEQ = itertools.count(1)


class VisionElement(BaseModel):
    """缓存进帧、并渲染给 Executor 的一行视觉主体。

    编号是 **1-based**，和工具回执 / 提示词里的 ``[i]`` 严格一致；``label`` 就是
    模型的 description；``bounds`` 已换算成像素。
    """

    index: int = Field(description="1-based index shown to the model")
    label: str = Field(description="Short description of the subject")
    bounds: ElementBounds


class ScreenFrame(BaseModel):
    """这一轮 Contextor 抓到的屏幕：所有索引都在这份对象上成立。"""

    epoch: str = Field(description="Content-derived identity of this screen")
    screenshot_base64: str = Field(description="Raw-resolution screenshot of this frame")
    width: int
    height: int
    screen_elements: list[ScreenElement] = Field(default_factory=list)
    ocr_results: list[OcrText] | None = None
    screen_state: str | None = None
    foreground_app: str | None = None
    captured_at: float = Field(default_factory=time.time)
    vision_elements: list[VisionElement] = Field(default_factory=list)


def frame_epoch(
    elements: list[ScreenElement] | None,
    ocr_results: list[OcrText] | None = None,
) -> str:
    """这一帧的内容身份：有元素表用元素表，没有就退回 OCR。

    时钟 / 点赞数这类纯数字噪音按 ``#noise#`` 归一（见 ``is_noise_text``），
    所以它们在跳不会把一次刚做完的视觉扫描误判成过期；位置或文字一变化，
    epoch 立刻不同。

    两条文字通道都拿不到任何行时返回**每轮唯一**的 ``empty@...``：此时没有
    任何内容可判定「是不是同一屏」，而视觉又是唯一通道，只能要求每轮重扫。
    """
    if elements:
        return element_table_fingerprint(elements)

    if ocr_results:
        parts: list[str] = []
        for item in ocr_results:
            text = (item.text or "").strip()
            normalized = "#noise#" if is_noise_text(text) else text
            bounds = item.bounds
            parts.append(f"{normalized}|{bounds.x},{bounds.y},{bounds.width}x{bounds.height}")
        return hashlib.sha1("\n".join(sorted(parts)).encode("utf-8")).hexdigest()

    return f"{_EMPTY_EPOCH_PREFIX}@{time.time_ns()}-{next(_EMPTY_EPOCH_SEQ)}"


def get_frame(ctx) -> ScreenFrame | None:
    """当前帧；``ctx`` 可能是测试替身，所以走 ``getattr``。"""
    return getattr(ctx, "screen_frame", None)


def publish_frame(ctx, frame: ScreenFrame) -> ScreenFrame:
    """发布本轮新帧，并决定视觉列表继承还是作废。

    继承条件只有一个：``epoch`` 相同。屏幕内容没变，扫描结果就还指向同一批
    主体；变了则整列作废 —— 编号、描述、框来自同一次扫描，绝不能拆开使用。
    """
    previous = get_frame(ctx)
    if previous is not None and previous.epoch == frame.epoch and previous.vision_elements:
        frame.vision_elements = list(previous.vision_elements)
    ctx.screen_frame = frame  # type: ignore[attr-defined]
    return frame


def attach_vision(ctx, elements: list[VisionElement]) -> bool:
    """把一次扫描的结果挂到当前帧上；没有帧时返回 False（调用方必须明说原因）。"""
    frame = get_frame(ctx)
    if frame is None:
        return False
    frame.vision_elements = elements
    return True


def clear_vision(ctx) -> None:
    """作废当前帧的视觉列表（扫描失败时用：旧编号可能指向另一屏）。"""
    frame = get_frame(ctx)
    if frame is not None:
        frame.vision_elements = []


def render_vision_rows(elements: list[VisionElement]) -> str:
    """只渲染主体行（不带表头）。空表返回空串。

    **编号契约的唯一来源**：这里印出的 ``[i]`` 和 :func:`resolve_vision` 吃的
    下标是同一条契约的两端。
    """
    lines: list[str] = []
    for element in elements:
        lines.append(
            f"[{element.index}] {json.dumps(element.label, ensure_ascii=False)} "
            f"({element.bounds.x},{element.bounds.y},"
            f"{element.bounds.width}x{element.bounds.height})"
        )
    return "\n".join(lines)


def resolve_vision(
    frame: ScreenFrame, index: int | None
) -> tuple[VisionElement | None, str | None]:
    """把 ``Target.vision_index`` 还原成帧上的视觉主体行（1-based）。

    失败一定返回**给模型看的原因**，不能静默退化成「点别的地方」：

    * 这一帧还没扫过（或扫描失败、屏变了导致列表被作废）；
    * 编号越界。
    """
    elements = frame.vision_elements or []
    if not elements:
        return None, (
            f"vision_index={index} cannot be used: there are no visual subjects on "
            "this screen - call scan_visual_subjects first"
        )
    if index is None or index < 1 or index > len(elements):
        return None, (
            f"vision_index={index} is out of range; the visual list had {len(elements)} "
            f"item(s), valid range is 1..{len(elements)}"
        )
    return elements[index - 1], None
