from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel

from minitap.mobile_use.constants import EXECUTOR_MESSAGES_KEY
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.controllers.controller_factory import create_device_controller
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.tools.tool_wrapper import ToolWrapper
from minitap.mobile_use.tools.types import Target
from minitap.mobile_use.tools.utils import (
    check_failure_breaker,
    focus_element_if_needed,
    move_cursor_to_end_if_bounds,
    record_target_failure,
    record_target_failure_fact,
    resolve_target,
)
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.screen_fingerprint import build_context, format_context
from minitap.mobile_use.utils.ui_hierarchy import find_element_by_resource_id, get_element_text

logger = get_logger(__name__)


class InputResult(BaseModel):
    """Result of an input operation from the controller layer."""

    ok: bool
    error: str | None = None


async def _controller_input_text(ctx: MobileUseContext, text: str) -> InputResult:
    """
    Thin wrapper to normalize the controller result.
    """
    controller = create_device_controller(ctx)
    success = await controller.input_text(text)
    if success:
        return InputResult(ok=True)
    return InputResult(ok=False, error="Failed to type text")


def get_focus_and_input_text_tool(ctx: MobileUseContext) -> BaseTool:
    @tool
    async def focus_and_input_text(
        agent_thought: str,
        text: str,
        target: Target,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[State, InjectedState],
    ):
        """
        Focus a text field and type text into it.

        - Ensure the corresponding element is focused (tap if necessary).
        - If bounds are available, tap near the end to place the cursor at the end.
        - Type the provided `text` using the controller.

        Args:
            agent_thought: The thought of the agent.
            text: The text to type.
            target: The target of the text input (if available).
        """
        # 把 index / ocr_index 解析成具体的元素。索引优先于模型自己抄的字段 ——
        # 它指向的是模型当时看到的那一行，id 和坐标都是代码给的。
        target, index_error = resolve_target(state=state, target=target)
        if index_error:
            logger.warning(f"Input target could not be resolved: {index_error}")

        # 熔断：同一个目标连着失败过就别再试了（见 tap 里的同一段说明）。
        breaker = check_failure_breaker(state=state, target=target)
        if breaker:
            logger.warning(f"Input blocked by failure breaker: {breaker}")
            error_message = breaker
            tool_message = ToolMessage(
                tool_call_id=tool_call_id,
                content=focus_and_input_text_wrapper.on_failure_fn(text, error_message),
                additional_kwargs={"error": error_message},
                status="error",
            )
            return Command(
                update=await state.asanitize_update(
                    ctx=ctx,
                    update={
                        "agents_thoughts": [agent_thought, error_message],
                        EXECUTOR_MESSAGES_KEY: [tool_message],
                        "failed_targets": record_target_failure(
                            state=state, target=target, reason=breaker[:200]
                        ),
                        "executor_failures": record_target_failure_fact(
                            state=state, target=target, reason=breaker[:200]
                        ),
                    },
                    agent="executor",
                ),
            )

        focus_method = await focus_element_if_needed(ctx=ctx, target=target)
        if not focus_method:
            error_message = "Failed to focus the text input element before typing."
            if index_error:
                error_message = f"{error_message} ({index_error})"
            tool_message = ToolMessage(
                tool_call_id=tool_call_id,
                content=focus_and_input_text_wrapper.on_failure_fn(text, error_message),
                additional_kwargs={"error": error_message},
                status="error",
            )
            return Command(
                update=await state.asanitize_update(
                    ctx=ctx,
                    update={
                        "agents_thoughts": [agent_thought, error_message],
                        EXECUTOR_MESSAGES_KEY: [tool_message],
                        "failed_targets": record_target_failure(
                            state=state, target=target, reason=error_message[:200]
                        ),
                        "executor_failures": record_target_failure_fact(
                            state=state, target=target, reason=error_message[:200]
                        ),
                    },
                    agent="executor",
                ),
            )

        await move_cursor_to_end_if_bounds(ctx=ctx, state=state, target=target)

        result = await _controller_input_text(ctx=ctx, text=text)
        status: Literal["success", "error"] = "success" if result.ok else "error"

        text_input_content = ""
        if status == "success" and target.resource_id:
            controller = create_device_controller(ctx)
            screen_data = await controller.get_screen_data()
            state.latest_ui_hierarchy = screen_data.elements
            element = find_element_by_resource_id(
                ui_hierarchy=state.latest_ui_hierarchy,
                resource_id=target.resource_id,
                index=target.resource_id_index,
            )
            if element:
                text_input_content = get_element_text(element)

        agent_outcome = (
            focus_and_input_text_wrapper.on_success_fn(
                text_to_type=text,
                text_from_resource_id=text_input_content,
                target_resource_id=target.resource_id,
                focus_method=focus_method,
            )
            if result.ok
            else focus_and_input_text_wrapper.on_failure_fn(text_to_type=text, error=result.error)
        )
        # 🔴 附上**上下文指纹**（~40 字符）—— 和 tap 同一个理由：
        # app 改版后 id 会变，**邻居**不会。详见 ``utils/screen_fingerprint.py``。
        if result.ok and target.resource_id:
            context = build_context(state.latest_screen_elements, target.resource_id)
            ctx_text = format_context(context)
            if ctx_text:
                agent_outcome = f"{agent_outcome} {ctx_text}"

        tool_message = ToolMessage(
            tool_call_id=tool_call_id,
            content=agent_outcome,
            additional_kwargs={"error": result.error} if not result.ok else {},
            status=status,
        )

        return Command(
            update=await state.asanitize_update(
                ctx=ctx,
                update={
                    "agents_thoughts": [agent_thought, agent_outcome],
                    EXECUTOR_MESSAGES_KEY: [tool_message],
                    "failed_targets": (
                        []
                        if result.ok
                        else record_target_failure(
                            state=state,
                            target=target,
                            reason=str(result.error or "input failed")[:200],
                        )
                    ),
                    # 跨子目标的失败事实（只增不减）：成功时**不清**。
                    "executor_failures": (
                        state.executor_failures
                        if result.ok
                        else record_target_failure_fact(
                            state=state,
                            target=target,
                            reason=str(result.error or "input failed")[:200],
                        )
                    ),
                },
                agent="executor",
            ),
        )

    return focus_and_input_text


def _on_input_success(text_to_type, text_from_resource_id, target_resource_id, focus_method):
    """Success message handler for input text operations."""
    if focus_method == "resource_id":
        return (
            f"Typed {repr(text_to_type)}\n"
            f"Here is the whole content of input with id {repr(target_resource_id)}: "
            f"{repr(text_from_resource_id)}"
        )
    else:
        return (
            f"Typed {repr(text_to_type)} using {focus_method}."
            + " Should now verify before moving forward."
        )


focus_and_input_text_wrapper = ToolWrapper(
    tool_fn_getter=get_focus_and_input_text_tool,
    on_success_fn=_on_input_success,
    on_failure_fn=lambda text, error: f"Failed to input text {repr(text)}. Reason: {error}",
)
