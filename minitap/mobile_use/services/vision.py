"""视觉感知：由 **vision 列出当屏全部视觉主体**，Executor 只按编号选取。

它是什么位置
------------
**不是定位工具，也不接受 Executor 写的目标描述。**

Executor 的文字通道只有元素表 + OCR。两条都叫不出名字的东西（纯图标按钮、
图片里的元素、WebView / 画布、自绘控件）以前只能靠模型猜坐标，或者干脆做不了。
现在补的是一条**按需触发的感知通道**：Executor 在决策时发现文字通道不足以确认
屏上有什么，就调用 ``scan_visual_subjects`` 工具；工具拿**当前感知帧**的截图，
问一次视觉模型「这屏上都有哪些视觉主体」，把**全部主体**编号后交给帧。

关键在方向：**识别什么由 vision 决定，不由 Executor 决定**。Executor 只决定
「什么时候需要看」，看到列表后再按 ``vision_index`` 选一个。让看不见图的
Executor 先描述目标、再让 vision 去找，等于让它猜屏上有什么 —— 真机实测
（2026-10-28，Redmi Turbo 4 / MiniMax-M3.1）里，模糊描述会让视觉模型
高置信度地给错框，置信度门限挡不住。

图仍然不进 Executor 的 prompt
-----------------------------
图只进**一次独立的小调用**。Executor 的 system prompt 和局部历史一个字节都不动，
只有模型真的调用扫描工具时才产生开销。

一帧一图
--------
本模块只接受 ``ScreenFrame``（见 ``perception/frame.py``），**不自己截屏**：
元素表、OCR 和视觉主体必须来自同一次抓取，否则两套编号会来自两屏。

坐标为什么用 0-1000 归一化
--------------------------
视觉模型不知道设备分辨率，也不该让它算像素。归一化坐标是 GUI grounding 的通用
口径；这里乘上 ``ctx.device.device_width/height`` 换算成像素框。

失败一律**降级**（返回原因，不抛）：视觉是补充通道，拿不到就如实告诉模型
「改用元素表 / OCR，别猜」，不能把整个动作步带崩。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.perception.frame import ScreenFrame, VisionElement
from minitap.mobile_use.prompt_language import PromptLanguage, load_prompt_template, pick
from minitap.mobile_use.services.llm import get_structured_llm, invoke_llm_with_timeout_message
from minitap.mobile_use.utils.image_compress import compress_screenshot_base64
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds

logger = get_logger(__name__)

#: 视觉扫描的系统提示词按语言分目录，英文是基准（``load_prompt_template``
#: 会把路径里的 ``en`` 段换成 ``zh``）。
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

#: 发给视觉模型的 JPEG 质量，与 OCR / 旧截图通道一致（q50 实测可读）。
SCREENSHOT_JPEG_QUALITY = 50

#: 归一化坐标的上限（0-1000 口径）。
NORMALIZED_MAX = 1000

#: 一次扫描最多返回多少个主体。
#:
#: 20 是「一屏值得操作/值得知道的东西」的经验上限：再多的行只会稀释注意力，
#: 也让工具结果变长。模型仍然可能返回更多，这里按顺序截断。
MAX_SUBJECTS = 20


class VisionSubject(BaseModel):
    """视觉模型看到的一个主体。坐标是 **0-1000 归一化**，不是像素。"""

    description: str = Field(
        description=(
            "A short, concrete description of the subject: what it is and where it is "
            "(e.g. 'gear icon, top right'). 3-15 words."
        )
    )
    x1: int = Field(default=0, description="Left edge, normalized 0-1000.")
    y1: int = Field(default=0, description="Top edge, normalized 0-1000.")
    x2: int = Field(default=0, description="Right edge, normalized 0-1000.")
    y2: int = Field(default=0, description="Bottom edge, normalized 0-1000.")


class VisionScanOutput(BaseModel):
    """视觉模型对「这一屏有哪些视觉主体」的完整回答。"""

    subjects: list[VisionSubject] = Field(
        default_factory=list,
        description="Every distinct visual subject on the screen, most important first.",
    )


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def bounds_from_normalized_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    screen_width: int,
    screen_height: int,
) -> ElementBounds | None:
    """把 0-1000 归一化框换算成像素框；框不合法（空框 / 反框）返回 ``None``。

    越界的坐标先夹到 0-1000 再乘，保证算出来的框一定落在屏幕内。
    """
    nx1, nx2 = _clamp(x1, 0, NORMALIZED_MAX), _clamp(x2, 0, NORMALIZED_MAX)
    ny1, ny2 = _clamp(y1, 0, NORMALIZED_MAX), _clamp(y2, 0, NORMALIZED_MAX)
    if nx2 <= nx1 or ny2 <= ny1:
        return None

    left = _clamp(int(nx1 / NORMALIZED_MAX * screen_width), 0, screen_width - 1)
    top = _clamp(int(ny1 / NORMALIZED_MAX * screen_height), 0, screen_height - 1)
    right = _clamp(int(round(nx2 / NORMALIZED_MAX * screen_width)), left + 1, screen_width)
    bottom = _clamp(int(round(ny2 / NORMALIZED_MAX * screen_height)), top + 1, screen_height)
    return ElementBounds(x=left, y=top, width=right - left, height=bottom - top)


def scan_system_prompt(ctx: MobileUseContext) -> str:
    """视觉扫描的系统提示词（``services/prompts/<lang>/vision_scan.md``）。

    它只服务「列清单」这一件事：定义主体粒度、排除项、框的紧度与尺度、
    描述口径和排序上限。刻意不写 Executor 的决策规则，也不写示例与自检 ——
    这是一次独立的小调用，指令越短，模型越容易照着做。

    ``max_subjects`` 由常量传入，避免提示词里的数字和 ``MAX_SUBJECTS`` 漂移。
    """
    return load_prompt_template(_PROMPTS_DIR / "en" / "vision_scan.md", ctx.prompt_language).render(
        max_subjects=MAX_SUBJECTS
    )


def render_scan_receipt(count: int, *, language: PromptLanguage) -> str:
    """工具回执：只报数量，**不再重复整张表**。

    完整编号表由 Executor 每轮从帧渲染（``prompt.render_vision_subjects_message``）。
    两处都列一遍时，缓存刚失效的那一轮会出现「历史里还有旧编号、当轮没有表」的
    割裂；表只留一处，编号就永远只有一个来源。
    """
    if count <= 0:
        return pick(
            "Visual scan finished: no visual subjects were found on this screen. "
            "Do not invent any. Use the element table or OCR instead.",
            "视觉扫描完成：这一屏没有找到视觉主体。不要编造。改用元素表或 OCR。",
            language=language,
        )
    return pick(
        f"Visual scan finished: {count} subject(s) are numbered in the vision table "
        'for this screen. Use target {"vision_index": <i>} to act on one; do not add '
        "bounds of your own. If the screen changes, the table is dropped - scan again.",
        f"视觉扫描完成：这一屏的 {count} 个视觉主体已列在视觉表中，编号从 1 开始。"
        '要操作其中一项，target 写 {"vision_index": <i>}，不要自己给 bounds。'
        "屏幕变了这张表会被作废，届时重新扫描。",
        language=language,
    )


async def scan_screen_subjects(
    ctx: MobileUseContext,
    screenshot_base64: str,
    *,
    screen_width: int | None = None,
    screen_height: int | None = None,
) -> tuple[list[VisionElement], str]:
    """对一张截图跑一次视觉扫描，返回 ``(主体列表, 失败原因)``。

    成功时原因为空串；失败时列表为空、原因是**给模型看的一句话**。
    任何异常都降级成原因，不抛。

    归一化框换算用的分辨率默认取 ``ctx.device``；扫帧时显式传帧自己的
    ``width/height`` —— 帧是那次抓屏的事实来源，横竖屏切换时不能拿旧尺寸算框。
    """
    width = screen_width or ctx.device.device_width
    height = screen_height or ctx.device.device_height

    try:
        # Pillow 的解码 / 缩放 / 编码是纯 CPU 活（真机 100~300ms），丢线程里。
        image_b64 = await asyncio.to_thread(
            compress_screenshot_base64, screenshot_base64, quality=SCREENSHOT_JPEG_QUALITY
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[vision] Screenshot compression failed: {type(e).__name__}: {e}")
        return [], f"the current screenshot could not be prepared ({type(e).__name__})"

    messages = [
        SystemMessage(content=scan_system_prompt(ctx)),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": pick(
                        "List every visual subject on this screen.",
                        "列出这一屏上所有的视觉主体。",
                        language=ctx.prompt_language,
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ]
        ),
    ]

    runner = get_structured_llm(
        ctx=ctx,
        name="vision",
        schema=VisionScanOutput,
        temperature=0,
    )
    try:
        response = await invoke_llm_with_timeout_message(
            runner.ainvoke(messages),
            agent_name="vision",
            task_id=ctx.trace_id,
        )
    except Exception as e:  # noqa: BLE001
        # 含「这条渠道的模型看不了图」：照实说，让模型退回元素表索引。
        logger.warning(f"[vision] Scan call failed: {type(e).__name__}: {str(e)[:200]}")
        return [], (f"the vision model could not be used ({type(e).__name__}: {str(e)[:200]})")

    if not isinstance(response, VisionScanOutput):
        return [], "the vision model returned an unreadable answer"

    elements: list[VisionElement] = []
    for subject in response.subjects:
        description = (subject.description or "").strip()
        if not description:
            continue
        bounds = bounds_from_normalized_box(
            subject.x1,
            subject.y1,
            subject.x2,
            subject.y2,
            width,
            height,
        )
        if bounds is None:
            continue
        elements.append(VisionElement(index=len(elements) + 1, label=description, bounds=bounds))
        if len(elements) >= MAX_SUBJECTS:
            break

    logger.info(f"[vision] Scan found {len(elements)} visual subject(s)")
    return elements, ""


async def scan_frame(
    ctx: MobileUseContext, frame: ScreenFrame | None
) -> tuple[list[VisionElement], str]:
    """扫**当前感知帧**：直接用帧里的截图，不另抓一张。

    为什么不重新截屏：模型是基于这一帧的元素表决定「需要看一眼」的。若此处
    另抓一张新图，视觉编号来自新屏、元素表编号来自旧屏，屏一动两套编号就指向
    不同的东西。帧变了由 Contextor 下一轮负责作废，工具不越权。

    息屏 / 没有帧 / 帧里没有图时都返回**给模型看的原因**，不抛。
    """
    if frame is None:
        return [], "the current screen has not been read yet - its frame is unavailable"
    if frame.screen_state == "off":
        return [], "the screen is off, so a screenshot would be black - call wake_screen first"
    if not frame.screenshot_base64:
        return [], "the current screen frame has no screenshot"

    return await scan_screen_subjects(
        ctx,
        frame.screenshot_base64,
        screen_width=frame.width,
        screen_height=frame.height,
    )
