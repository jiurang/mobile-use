"""按需视觉扫描工具：让 Executor 在「看不清」时先看一眼，而不是猜。

为什么是工具，而不是常驻感知
----------------------------
元素表和 OCR 是文字通道，它们对图标、图片、画布、自绘控件天然失明。常驻扫描
每步都要付一次截图 + 一次视觉调用，而绝大多数步骤根本用不上；按需扫描把成本
挪到真正需要的时刻。

但**识别什么不能由 Executor 决定**：它看不见图，让它写「找右上角的齿轮」等于
让它猜屏上有什么。所以这个工具不接受任何目标描述，只回答一个问题：
「这一屏有哪些视觉主体？」—— 全部列出来、编号、交给模型挑。

结果同时缓存进 state（``vision_elements``）：下一步模型写
``Target.vision_index`` 时，工具侧必须能还原到**同一个**框，只靠工具回执里的
文字是不够的。
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
from minitap.mobile_use.prompt_language import pick
from minitap.mobile_use.services.vision import render_vision_scan_result, scan_current_screen
from minitap.mobile_use.tools.tool_wrapper import ToolWrapper
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.screen_elements import scan_cache_fingerprint

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
        subjects, and you pick one by its 1-based number with
        target {"vision_index": <i>} in your next action.

        The list is valid only for the screen it was produced on. If the screen
        changes before you act, call this again.
        """
        started = time.perf_counter()
        elements, reason = await scan_current_screen(ctx)
        succeeded = not reason

        if succeeded:
            content = render_vision_scan_result(elements, language=ctx.prompt_language)
            outcome = scan_visual_subjects_wrapper.on_success_fn(len(elements))
            update = {
                "vision_elements": elements,
                # 记下扫描时那一屏的指纹：Contextor 每步比对，屏变了就清掉。
                "vision_scan_fingerprint": scan_cache_fingerprint(
                    state.latest_screen_elements, state.latest_ocr_results
                ),
            }
        else:
            content = pick(
                f"Visual scan failed: {reason}. Do not guess what is on screen: "
                "use the element table or OCR, or fix the screen first.",
                f"视觉扫描失败：{reason}。不要猜屏上有什么：改用元素表或 OCR，"
                "或先把屏幕处理好。",
                language=ctx.prompt_language,
            )
            outcome = scan_visual_subjects_wrapper.on_failure_fn(reason)
            # 失败就把旧缓存清掉：旧编号对应的可能是另一屏，继续留着只会点错。
            update = {"vision_elements": [], "vision_scan_fingerprint": None}

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            f"[vision] scan_visual_subjects: {len(elements)} subject(s), "
            f"success={succeeded}, {elapsed_ms:.0f}ms"
        )

        tool_message = ToolMessage(
            tool_call_id=tool_call_id,
            content=content,
            additional_kwargs={} if succeeded else {"error": reason},
            status="success" if succeeded else "error",
        )
        return Command(
            update=await state.asanitize_update(
                ctx=ctx,
                update={
                    "agents_thoughts": [agent_thought, outcome],
                    EXECUTOR_MESSAGES_KEY: [tool_message],
                    **update,
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
