"""视觉定位：按一句话描述，在当屏截图上找到一个目标，给出它的像素框。

它是什么位置
------------
**不是第三条感知通道，也不把图交给 Executor 的 LLM。**

Executor 的感知仍然只有两条文字通道（元素表 + OCR）。两条都叫不出名字的东西
（纯图标按钮、图片里的元素、WebView / 画布）以前只能靠模型猜坐标，或者干脆做不了。
这里补的是**寻址**，不是**描述**：模型在 ``Target.vision`` 写一句话，框架现截一张
图，问一次视觉模型「它在哪」，再把框填回 ``Target.bounds``，后续工具照常执行。

为什么不复用旧的 ``take_screenshot``
------------------------------------
那条是「把图发给 Executor 的 LLM」：每张约 1,200 input token，而且图进局部历史会
顶掉前缀缓存（备份实测缓存率 82% → 41%）。这里图只进**一次独立的小调用**，
Executor 的 system prompt 和局部历史一个字节都不动 —— 只有模型主动用 ``vision``
时才产生开销。

坐标为什么用 0-1000 归一化
--------------------------
视觉模型不知道设备分辨率，也不该让它算像素。归一化坐标是 GUI grounding 的通用
口径；这里乘上 ``ctx.device.device_width/height`` 换算成像素框。

调用失败一律**降级**（返回原因，不抛）：视觉定位是补充通道，拿不到就如实告诉
模型「改用 index / 换个描述」，不能把整个动作步带崩。
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.controllers.unified_controller import UnifiedMobileController
from minitap.mobile_use.prompt_language import pick
from minitap.mobile_use.services.llm import get_structured_llm, invoke_llm_with_timeout_message
from minitap.mobile_use.utils.image_compress import compress_screenshot_base64
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds

logger = get_logger(__name__)

#: 发给视觉模型的 JPEG 质量，与 OCR / 旧截图通道一致（q50 实测可读）。
SCREENSHOT_JPEG_QUALITY = 50

#: 归一化坐标的上限（0-1000 口径）。
NORMALIZED_MAX = 1000

#: 采纳一次视觉定位所需的最低置信度（0-100）。
#:
#: 真机实测（2026-10-28，Redmi Turbo 4 / MiniMax-M3.1）：屏上根本不存在的目标
#: （「一只戴墨镜的蓝色鲸鱼」）也会被强行给一个小框。视觉框会跳过 IoU 校验，
#: 所以这条门限是「不要拿相似的东西顶替」的唯一硬保险。
MIN_CONFIDENCE = 60


class VisionLocationOutput(BaseModel):
    """视觉模型的回答：目标在不在、在哪儿。

    坐标是 **0-1000 归一化**的边界框，不是像素：模型不知道设备分辨率，
    让它算像素只会把框算歪。``confidence`` 是给框架做采纳门限用的 —— 模型
    对「相似但不是」的判断只有它自己能表达。
    """

    found: bool = Field(description="True when the described target is visible on the screen.")
    confidence: int = Field(
        default=0,
        description=(
            "0-100: how sure you are that this box is the described target, not just "
            "something similar. Below 60 the box is discarded."
        ),
    )
    x1: int = Field(default=0, description="Left edge of the box, normalized 0-1000.")
    y1: int = Field(default=0, description="Top edge of the box, normalized 0-1000.")
    x2: int = Field(default=0, description="Right edge of the box, normalized 0-1000.")
    y2: int = Field(default=0, description="Bottom edge of the box, normalized 0-1000.")


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

    快路失败**不抛**，直接回落通用路 —— 视觉定位是补充通道。
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


def _system_prompt(ctx: MobileUseContext) -> str:
    """视觉定位的系统提示词。只讲三件事：找什么、用什么坐标、看不见怎么办。

    ⚠️ 刻意保持极短：这是**一次独立的小调用**，不是 Executor 的决策提示词。
    """
    return pick(
        "You locate one target in a phone screenshot. The user describes it in a few words. "
        "Return its bounding box in normalized coordinates (x1,y1,x2,y2 on a 0-1000 scale) "
        "and a confidence 0-100. If the target is not clearly visible, set found=false - "
        "never substitute a similar-looking object, and never guess.",
        "你在手机截图里定位一个目标。用户会用一句话描述它。"
        "用 0-1000 归一化坐标返回它的边界框 x1,y1,x2,y2，并给出 0-100 的置信度。"
        "看不见、或只有相似的东西，就 found=false —— 不要拿相似的东西顶替，也不要猜。",
        language=ctx.prompt_language,
    )


async def locate_target_by_vision(
    ctx: MobileUseContext, description: str
) -> tuple[ElementBounds | None, str]:
    """按描述在当屏截图上定位目标。

    Returns:
        ``(bounds, 失败原因)``。成功时原因为空串；失败时 bounds 为 ``None``，
        原因是**给模型看的一句话**（说明为什么没找到、下一步该用什么）。
    """
    controller = UnifiedMobileController(ctx)

    # 息屏时截出来是一张全黑图，问视觉模型只是白花一次调用。
    try:
        power_state = await controller.get_screen_power_state()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[vision] Could not read the screen power state: {e}")
        power_state = "unknown"
    if power_state == "off":
        return None, "the screen is off, so the screenshot would be black - call wake_screen first"

    try:
        raw = await capture_screen_base64(ctx)
        # Pillow 的解码 / 缩放 / 编码是纯 CPU 活（真机 100~300ms），丢线程里。
        image_b64 = await asyncio.to_thread(
            compress_screenshot_base64, raw, quality=SCREENSHOT_JPEG_QUALITY
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[vision] Screenshot failed: {type(e).__name__}: {e}")
        return None, f"could not take a screenshot: {type(e).__name__}: {e}"

    messages = [
        SystemMessage(content=_system_prompt(ctx)),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": pick(
                        f"Target: {description}",
                        f"目标：{description}",
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
        schema=VisionLocationOutput,
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
        logger.warning(f"[vision] Vision call failed: {type(e).__name__}: {str(e)[:200]}")
        return None, (
            f"the vision model could not be used ({type(e).__name__}: {str(e)[:200]}) - "
            "use an element-table index or text instead"
        )

    if not isinstance(response, VisionLocationOutput) or not response.found:
        return None, f"nothing matching {description!r} is visible on the screen"

    if response.confidence < MIN_CONFIDENCE:
        logger.info(
            f"[vision] Rejected {description!r}: confidence {response.confidence}/100 "
            f"(threshold {MIN_CONFIDENCE})"
        )
        return None, (
            f"not sure that {description!r} is on the screen "
            f"(confidence {response.confidence}/100) - describe it more specifically "
            "or use an element-table index"
        )

    bounds = bounds_from_normalized_box(
        response.x1,
        response.y1,
        response.x2,
        response.y2,
        ctx.device.device_width,
        ctx.device.device_height,
    )
    if bounds is None:
        return None, "the vision model returned no usable box"

    logger.info(
        f"[vision] Located {description!r} at {bounds} (confidence {response.confidence}/100)"
    )
    return bounds, ""


def _target_with_vision_bounds(description: str, bounds: ElementBounds) -> dict[str, Any]:
    """视觉框写进 target —— 它就是**唯一**的定位依据。

    模型写 ``vision`` 的前提就是「元素表和 OCR 都叫不出这个目标」。留着
    ``index`` / ``resource_id`` / ``text`` 中任何一条，工具都会在视觉框之前先用它
    （``index`` 会被 ``resolve_target`` 用旧行覆盖；``resource_id`` 会按身份在当屏
    重新定位），等于悄悄否掉这次视觉结果。``vision`` 本身留着：它是「这个框来自
    视觉」的出处（坐标校验靠它放行，失败账本靠它认身份）。
    """
    return {
        "vision": description,
        "bounds": {
            "x": bounds.x,
            "y": bounds.y,
            "width": bounds.width,
            "height": bounds.height,
        },
    }


async def resolve_vision_targets(
    ctx: MobileUseContext, actions: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str]:
    """把动作里的 ``target.vision`` 解析成当屏坐标。

    调用点：Executor 派发**之前**（一轮只有一个动作，所以每步最多一次视觉调用）。

    Returns:
        ``(actions, 失败原因)``。成功时原因为空串；失败时原样返回输入的动作，
        由调用方决定「不派发 + 回灌原因」。
    """
    resolved: list[dict[str, Any]] = []
    for action in actions:
        target = action.get("target")
        if not isinstance(target, dict) or not target.get("vision"):
            resolved.append(action)
            continue

        description = target["vision"]
        if not isinstance(description, str) or not description.strip():
            return actions, "target.vision must be a non-empty description"

        bounds, reason = await locate_target_by_vision(ctx, description.strip())
        if bounds is None:
            return actions, reason

        resolved.append(
            {
                **action,
                "target": _target_with_vision_bounds(description.strip(), bounds),
            }
        )
    return resolved, ""
