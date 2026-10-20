from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from minitap.mobile_use.constants import EXECUTOR_MESSAGES_KEY
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.controllers.unified_controller import UnifiedMobileController
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.tools.bounds_validation import coordinates_look_like_hierarchy_derived
from minitap.mobile_use.tools.live_target import resolve_live_bounds_by_id
from minitap.mobile_use.tools.tool_wrapper import ToolWrapper
from minitap.mobile_use.tools.types import Target
from minitap.mobile_use.tools.utils import (
    check_failure_breaker,
    record_target_failure,
    resolve_target,
    target_label,
)
from minitap.mobile_use.utils.failed_targets import FailedTarget
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


def get_long_press_on_tool(ctx: MobileUseContext) -> BaseTool:
    @tool
    async def long_press_on(
        agent_thought: str,
        target: Target,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[State, InjectedState],
        duration_ms: int = 1000,
    ):
        """
        Long presses on a UI element identified by the 'target' object.

        The 'target' object allows specifying an element by its resource_id
        (with an optional index), its bounds, or its text content (with an optional index).
        The tool uses a fallback strategy, trying the locators in that order.

        Args:
            target: The UI element to long press on (bounds, resource_id, or text).
            duration_ms: Duration of the long press in milliseconds. Choose based on interaction:
                        - 500-800ms: Quick long press (e.g., selecting text, haptic feedback)
                        - 1000ms (default): Standard long press (most common use case)
                        - 1500-2000ms: Extended long press (e.g., context menus, special actions)
                        - 2500ms+: Very long press (e.g., accessibility, advanced gestures)
        """
        error_obj: dict | None = {
            "error": "No valid selector provided or all selectors failed."
        }  # Default to failure
        latest_selector_info: str | None = None
        blocked = False

        controller = UnifiedMobileController(ctx)

        # 把 index / ocr_index 解析成具体的元素。索引优先于模型自己抄的
        # 坐标 —— 它指向的是模型当时看到的那一行，不会抄错。
        # 索引用不了**不等于立刻失败**：模型自己给的 bounds/id/text 仍然可以试，
        # 但原因要留着，最后一起报给模型。
        target, index_error = resolve_target(state=state, target=target)
        if index_error:
            logger.warning(f"Long press target could not be resolved: {index_error}")

        # 熔断：同一个目标已经连着失败过，就别再点它了（见 tap 里的同一段说明）。
        breaker = check_failure_breaker(state=state, target=target)
        if breaker:
            logger.warning(f"Long press blocked by failure breaker: {breaker}")
            latest_selector_info = target_label(target)
            error_obj = {"error": breaker}
            blocked = True

        # 1. 有 resource_id → 按**身份**在当屏重新定位它，再点它的当前位置。
        #
        #    和 tap 同一套思路：门限当初防的是「模型从缩放截图上目测坐标」，
        #    那个前提已经不存在（截图通道是关的、模型只报索引）。按身份找
        #    没有门限就没有误伤，屏变了也点的是同一个元素的新位置。
        if error_obj is not None and not blocked and target.resource_id:
            live_bounds, why = await resolve_live_bounds_by_id(
                ctx=ctx,
                resource_id=target.resource_id,
                index=target.resource_id_index or 0,
                hint=target.bounds,
            )
            if live_bounds is None:
                logger.warning(f"Target is not on the current screen: {why}")
                latest_selector_info = f"resource_id={target.resource_id}"
                error_obj = {"error": why}
            else:
                center_point = live_bounds.get_center()
                try:
                    logger.info(
                        "Attempting to long press using resource_id: "
                        f"'{target.resource_id}' at {center_point.x},{center_point.y}"
                    )
                    latest_selector_info = (
                        f"resource_id='{target.resource_id}' ({center_point.x},{center_point.y})"
                    )
                    result = await controller.tap_at(
                        x=center_point.x,
                        y=center_point.y,
                        long_press=True,
                        long_press_duration=duration_ms,
                    )
                    if result.error is None:  # Success
                        error_obj = None
                    else:
                        logger.warning(
                            f"Long press with resource_id '{target.resource_id}' failed. "
                            f"Error: {result.error}"
                        )
                        error_obj = {"error": result.error}
                except Exception as e:
                    logger.warning(
                        f"Exception during long press with resource_id '{target.resource_id}': {e}"
                    )
                    error_obj = {"error": str(e)}

        # 2. 没有 resource_id（OCR 行）但有坐标 → 位置校验后点。
        #    OCR 文本不在层级树里，没有 id 可查，只能靠几何判据。
        if error_obj is not None and not blocked and target.bounds and not target.resource_id:
            # 与 tap 同一套校验：坐标若不像是从层级树抄来的，就别去点一个不知道在哪的地方，
            # 直接让位给 resource_id / text 两级兜底。
            derived_ok, derived_reason = await coordinates_look_like_hierarchy_derived(
                ctx=ctx, target=target
            )
            if not derived_ok:
                logger.warning(f"Long press coordinates rejected: {derived_reason}")
                reason = derived_reason
                if target.index is not None:
                    # 索引指向「模型看到的那一行」；校验不过说明当屏已经变了。
                    # 不说清楚，模型只会把同一个索引原样再报一遍。
                    reason = (
                        f"{derived_reason} (this target came from screen element "
                        f"index {target.index}; the screen may have changed since "
                        f"it was read — re-read the screen and use a fresh index)"
                    )
                latest_selector_info = f"coordinates='{target.bounds}' rejected"
                error_obj = {"error": reason}
            else:
                try:
                    center_point = target.bounds.get_center()
                    logger.info(
                        "Attempting to long press using coordinates: "
                        f"{center_point.x},{center_point.y}"
                    )
                    latest_selector_info = f"coordinates='{target.bounds}'"
                    result = await controller.tap_at(
                        x=center_point.x,
                        y=center_point.y,
                        long_press=True,
                        long_press_duration=duration_ms,
                    )
                    if result.error is None:  # Success
                        error_obj = None
                    else:
                        logger.warning(
                            f"Long press with coordinates '{target.bounds}' failed. "
                            f"Error: {result.error}"
                        )
                        error_obj = {"error": result.error}
                except Exception as e:
                    logger.warning(
                        f"Exception during long press with coordinates '{target.bounds}': {e}"
                    )
                    error_obj = {"error": str(e)}

        # 3. If the element wasn't found by id and there's no usable geometry,
        #    try with text (last resort)
        if error_obj is not None and not blocked and target.text:
            try:
                logger.info(
                    f"Attempting to long press using text: '{target.text}' "
                    f"at index {target.text_index}"
                )
                latest_selector_info = f"text='{target.text}' (index={target.text_index})"
                result = await controller.tap_element(
                    text=target.text,
                    index=target.text_index or 0,
                    long_press=True,
                    long_press_duration=duration_ms,
                )
                if result.error is None:  # Success
                    error_obj = None
                else:
                    logger.warning(
                        f"Long press with text '{target.text}' failed. Error: {result.error}"
                    )
                    error_obj = {"error": result.error}
            except Exception as e:
                logger.warning(f"Exception during long press with text '{target.text}': {e}")
                error_obj = {"error": str(e)}

        has_failed = error_obj is not None
        final_selector_info = latest_selector_info if latest_selector_info else "N/A"
        if has_failed and index_error:
            # 索引用不了这件事要单独说清楚，不能被别的错误盖掉 ——
            # 模型需要知道是「第 42 行不存在」而不是「点空了」。
            error_obj = {**error_obj, "index_error": index_error}
        agent_outcome = (
            long_press_on_wrapper.on_failure_fn(final_selector_info)
            if has_failed
            else long_press_on_wrapper.on_success_fn(final_selector_info)
        )

        # 成功清账、失败记账：让「这个目标已经不在屏上了」进到下一轮上下文。
        failed_targets: list[FailedTarget] = (
            []
            if not has_failed
            else record_target_failure(
                state=state,
                target=target,
                reason=str((error_obj or {}).get("error", ""))[:200],
            )
        )

        tool_message = ToolMessage(
            tool_call_id=tool_call_id,
            content=agent_outcome,
            additional_kwargs=error_obj if has_failed else {},
            status="error" if has_failed else "success",
        )
        return Command(
            update=await state.asanitize_update(
                ctx=ctx,
                update={
                    "agents_thoughts": [agent_thought, agent_outcome],
                    EXECUTOR_MESSAGES_KEY: [tool_message],
                    "failed_targets": failed_targets,
                },
                agent="executor",
            ),
        )

    return long_press_on


long_press_on_wrapper = ToolWrapper(
    tool_fn_getter=get_long_press_on_tool,
    on_success_fn=lambda selector_info: (
        f"Long press on element with {selector_info} was successful."
    ),
    on_failure_fn=lambda selector_info: (
        "Failed to long press on element. " + f"Last attempt was with {selector_info}."
    ),
)
