"""视觉感知：由 **vision 列出当屏全部视觉主体**，Executor 只按编号选取。

它是什么位置
------------
**不是定位工具，也不接受 Executor 写的目标描述。**

Executor 的文字通道只有元素表 + OCR。两条都叫不出名字的东西（纯图标按钮、
图片里的元素、WebView / 画布、自绘控件）以前只能靠模型猜坐标，或者干脆做不了。
现在补的是一条**按需触发的感知通道**：Executor 在决策时发现文字通道不足以确认
屏上有什么，就调用 ``scan_visual_subjects`` 工具；工具截一张当屏图，问一次视觉
模型「这屏上都有哪些视觉主体」，把**全部主体**编号后返回。

关键在方向：**识别什么由 vision 决定，不由 Executor 决定**。Executor 只决定
「什么时候需要看」，看到列表后再按 ``vision_index`` 选一个。让看不见图的
Executor 先描述目标、再让 vision 去找，等于让它猜屏上有什么 —— 真机实测
（2026-10-28，Redmi Turbo 4 / MiniMax-M3.1）里，模糊描述会让视觉模型
高置信度地给错框，置信度门限挡不住。

图仍然不进 Executor 的 prompt
-----------------------------
图只进**一次独立的小调用**。Executor 的 system prompt 和局部历史一个字节都不动，
只有模型真的调用扫描工具时才产生开销。

坐标为什么用 0-1000 归一化
--------------------------
视觉模型不知道设备分辨率，也不该让它算像素。归一化坐标是 GUI grounding 的通用
口径；这里乘上 ``ctx.device.device_width/height`` 换算成像素框。

失败一律**降级**（返回原因，不抛）：视觉是补充通道，拿不到就如实告诉模型
「改用元素表 / OCR，别猜」，不能把整个动作步带崩。
"""

from __future__ import annotations

import asyncio
import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.controllers.unified_controller import UnifiedMobileController
from minitap.mobile_use.prompt_language import PromptLanguage, pick
from minitap.mobile_use.services.llm import get_structured_llm, invoke_llm_with_timeout_message
from minitap.mobile_use.utils.image_compress import compress_screenshot_base64
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds

logger = get_logger(__name__)

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


class VisionElement(BaseModel):
    """缓存进 state、并渲染给 Executor 的一行视觉主体。

    编号是 **1-based**，和工具结果 / 提示词里的 ``[i]`` 严格一致；``label`` 就是
    模型的 description；``bounds`` 已换算成像素。
    """

    index: int = Field(description="1-based index shown to the model")
    label: str = Field(description="Short description of the subject")
    bounds: ElementBounds


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


async def capture_screen_base64(ctx: MobileUseContext) -> str:
    """抓一张当屏截图，返回**原始分辨率**的 base64（不带 data URI 前缀）。

    两条路，快的那条优先：

    * ``ctx.ui_adb_client.get_screenshot_base64()`` —— uiautomator 只截图，
      不动层级树（真机实测 166ms）；
    * 控制器那条 ``take_screenshot()``（iOS / 云设备 / 快路不可用）。

    快路失败**不抛**，直接回落通用路 —— 视觉是补充通道。
    """
    client = getattr(ctx, "ui_adb_client", None)
    getter = getattr(client, "get_screenshot_base64", None)
    if getter is not None:
        try:
            image = await asyncio.to_thread(getter)
            if image:
                return image
            logger.warning("[vision] Fast capture returned nothing; falling back")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[vision] Fast capture failed ({e}); falling back")

    return await UnifiedMobileController(ctx).take_screenshot()


def scan_system_prompt(ctx: MobileUseContext) -> str:
    """视觉扫描的系统提示词。

    只讲三件事：什么算视觉主体、坐标口径、看不见怎么办。刻意保持短 ——
    这是一次独立的小调用，不是 Executor 的决策提示词。
    """
    return pick(
        "You inventory the visual subjects in a phone screenshot. A visual subject is "
        "anything whose meaning is carried by pixels rather than readable text: icons, "
        "photos, illustrations, avatars, video thumbnails, canvas or custom-drawn "
        "graphics, and buttons with no label.\n"
        "List every distinct visual subject you can see. Skip plain text, paragraphs, "
        "clocks, battery indicators and decorative separators.\n"
        "For each subject return:\n"
        "- description: 3-15 words saying what it is and where it is (e.g. 'gear icon, "
        "top right', 'photo of a red-roofed house in the middle of the feed').\n"
        "- bounding box x1,y1,x2,y2 normalized to 0-1000 (0,0 is the top-left corner, "
        "1000,1000 is the bottom-right).\n"
        "Return at most 20 subjects, most prominent or actionable first. Never invent a "
        "subject that is not visibly there; if there are none, return an empty list.",
        "你在清点一张手机截图里的视觉主体。视觉主体指含义由像素而不是可读文字承载的"
        "东西：图标、照片、插画、头像、视频封面、画布或自绘图形，以及没有文字标签"
        "的按钮。\n"
        "请列出所有彼此可区分的视觉主体。纯文字、段落、时钟、电量、装饰分隔线请跳过。\n"
        "每个主体返回两项：\n"
        "- description：3~15 个字，说清「是什么 + 在哪」（例如「齿轮图标，右上角」"
        "「红屋顶房子的照片，信息流中间」）。\n"
        "- 边界框 x1,y1,x2,y2，归一化到 0-1000（左上角是 0,0，右下角是 1000,1000）。\n"
        "最多 20 项，按显眼或可操作程度排序。看不见的东西一律不要编造；如果"
        "一个都没有，返回空列表。",
        language=ctx.prompt_language,
    )


def render_vision_rows(elements: list[VisionElement]) -> str:
    """只渲染主体行（不带表头）。空表返回空串。

    **编号契约的唯一来源**：这里印出的 ``[i]`` 和 ``resolve_vision_index`` 吃的
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


def resolve_vision_index(
    elements: list[VisionElement] | None, index: int | None
) -> VisionElement | None:
    """把 ``Target.vision_index`` 还原成主体行。**1-based**，越界返回 None。

    越界或没有缓存时返回 None —— 调用方必须把「这个编号不存在」告诉模型，
    不能退化成「点别的地方」。
    """
    if index is None or not elements:
        return None
    if index < 1 or index > len(elements):
        logger.warning(
            f"vision_index={index} out of range; the visual list had {len(elements)} "
            f"item(s) (valid 1..{len(elements)})"
        )
        return None
    return elements[index - 1]


def render_vision_scan_result(
    elements: list[VisionElement], *, language: PromptLanguage
) -> str:
    """工具结果的正文：表头 + 编号行 + 怎么用。

    这是 Executor **下一轮**看到的原文，必须同时说清三件事：编号从 1 开始、
    只对当屏有效、动作时写 ``vision_index``。
    """
    rows = render_vision_rows(elements)
    if not rows:
        return pick(
            "Visual scan finished: no visual subjects were found on this screen. "
            "Do not invent any. Use the element table or OCR instead.",
            "视觉扫描完成：这一屏没有找到视觉主体。不要编造。改用元素表或 OCR。",
            language=language,
        )
    header = pick(
        "Visual subjects on the current screen (1-based, valid only for this screen):",
        "当前屏幕的视觉主体（1-based，只对当屏有效）：",
        language=language,
    )
    usage = pick(
        'To act on one, use target {"vision_index": <i>}. Do not add bounds of your own. '
        "If the screen changes before you act, scan again.",
        '要操作其中一项，target 写 {"vision_index": <i>}，不要自己给 bounds。'
        "动作之前屏幕变了，就重新扫描。",
        language=language,
    )
    return f"{header}\n{rows}\n\n{usage}"


async def scan_screen_subjects(
    ctx: MobileUseContext, screenshot_base64: str
) -> tuple[list[VisionElement], str]:
    """对一张截图跑一次视觉扫描，返回 ``(主体列表, 失败原因)``。

    成功时原因为空串；失败时列表为空、原因是**给模型看的一句话**。
    任何异常都降级成原因，不抛。
    """
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
        return [], (
            f"the vision model could not be used ({type(e).__name__}: {str(e)[:200]})"
        )

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
            ctx.device.device_width,
            ctx.device.device_height,
        )
        if bounds is None:
            continue
        elements.append(
            VisionElement(index=len(elements) + 1, label=description, bounds=bounds)
        )
        if len(elements) >= MAX_SUBJECTS:
            break

    logger.info(f"[vision] Scan found {len(elements)} visual subject(s)")
    return elements, ""


async def scan_current_screen(ctx: MobileUseContext) -> tuple[list[VisionElement], str]:
    """扫一次**当前**屏幕：读电源状态 → 抓图 → 调视觉模型。

    息屏时截出来是一张全黑图，问视觉模型只是白花一次调用，所以先挡掉。
    """
    controller = UnifiedMobileController(ctx)

    try:
        power_state = await controller.get_screen_power_state()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[vision] Could not read the screen power state: {e}")
        power_state = "unknown"
    if power_state == "off":
        return [], "the screen is off, so a screenshot would be black - call wake_screen first"

    try:
        raw = await capture_screen_base64(ctx)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[vision] Screenshot failed: {type(e).__name__}: {e}")
        return [], f"the screenshot could not be taken ({type(e).__name__}: {e})"
    if not raw:
        return [], "the screenshot came back empty"

    return await scan_screen_subjects(ctx, raw)
