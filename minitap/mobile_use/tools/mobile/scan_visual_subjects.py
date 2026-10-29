"""按需视觉扫描工具：让 Executor 在「看不清」时先看一眼，而不是猜。

为什么是工具，而不是常驻感知
----------------------------
元素表和 OCR 是文字通道，它们对图标、图片、画布、自绘控件天然失明。常驻扫描
每步都要付一次截图 + 一次视觉调用，而绝大多数步骤根本用不上；按需扫描把成本
挪到真正需要的时刻。

但**识别什么不能由 Executor 决定**：它看不见图，让它写「找右上角的齿轮」等于
让它猜屏上有什么。所以这个工具不接受任何目标描述，只回答一个问题：
「这一屏有哪些视觉主体？」—— 全部列出来、编号，挂到**当前感知帧**上。

一帧一图
--------
扫描读 ``ctx.screen_frame.screenshot_base64``，**不自己截屏**。元素表编号和
视觉编号必须来自同一帧，否则屏一动两套编号就指向不同的东西。

结果挂回帧（``frame.vision_elements``）而不是写进 state：下一步模型写
``Target.vision_index`` 时，工具侧必须能还原到**同一个**框；帧是唯一持有者，
Contextor 换帧时会在同处决定保留还是作废。
"""

from __future__ import annotations

import time
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from minitap.mobile_use.constants import EXECUTOR_MESSAGES_KEY
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.perception.frame import attach_vision, clear_vision, get_frame
from minitap.mobile_use.prompt_language import pick
from minitap.mobile_use.services.vision import render_scan_receipt, scan_frame
from minitap.mobile_use.tools.tool_wrapper import ToolWrapper
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


def get_scan_visual_subjects_tool(ctx: MobileUseContext):
    @tool
    async def scan_visual_subjects(
        agent_thought: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[State, InjectedState],
    ) -> Command:
        """List every visual subject on the current screen (icons, images, canvas,
        controls with no readable text) and number them.

        Use this when the element table and the OCR block do not let you be certain
        what is on screen. It takes no target description: it returns ALL visual
        subjects, which are then numbered in the vision table; you pick one by its
        1-based number with target {"vision_index": <i>} in your next action.

        The list belongs to the screen frame you were shown. If the screen changes,
        the table is dropped and you must call this again.
        """
        started = time.perf_counter()
        frame = get_frame(ctx)
        elements, reason = await scan_frame(ctx, frame)
        succeeded = not reason

        if reason:
            content = pick(
                f"Visual scan failed: {reason}. Do not guess what is on screen: "
                "use the element table or OCR, or fix the screen first.",
                f"视觉扫描失败：{reason}。不要猜屏上有什么：改用元素表或 OCR，或先把屏幕处理好。",
                language=ctx.prompt_language,
            )
            # 旧列表只对旧帧有效；扫描失败说明这一帧的视觉信息拿不到，作废。
            clear_vision(ctx)
            outcome = scan_visual_subjects_wrapper.on_failure_fn(reason)
        else:
            content = render_scan_receipt(len(elements), language=ctx.prompt_language)
            if not attach_vision(ctx, elements):
                logger.warning("[vision] Scan succeeded but there is no current frame")
            outcome = scan_visual_subjects_wrapper.on_success_fn(len(elements))

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            f"[vision] scan_visual_subjects: {len(elements)} subject(s), "
            f"success={succeeded}, {elapsed_ms:.0f}ms"
        )

        tool_message = ToolMessage(
            tool_call_id=tool_call_id,
            content=content,
            additional_kwargs={} if not reason else {"error": reason},
            status="success" if not reason else "error",
        )
        return Command(
            update=await state.asanitize_update(
                ctx=ctx,
                update={
                    "agents_thoughts": [agent_thought, outcome],
                    EXECUTOR_MESSAGES_KEY: [tool_message],
                },
                agent="executor",
            ),
        )

    return scan_visual_subjects


scan_visual_subjects_wrapper = ToolWrapper(
    tool_fn_getter=get_scan_visual_subjects_tool,
    on_success_fn=lambda count: f"Visual scan found {count} visual subject(s).",
    on_failure_fn=lambda reason: f"Visual scan failed: {reason}.",
)
