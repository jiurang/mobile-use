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
from minitap.mobile_use.services.ocr.types import OcrText
from minitap.mobile_use.tools.bounds_validation import coordinates_look_like_hierarchy_derived
from minitap.mobile_use.tools.tool_wrapper import ToolWrapper
from minitap.mobile_use.tools.types import Target
from minitap.mobile_use.tools.utils import has_valid_selectors, validate_coordinates_bounds
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.ocr_view import resolve_ocr_index

logger = get_logger(__name__)


def get_tap_tool(ctx: MobileUseContext) -> BaseTool:
    @tool
    async def tap(
        agent_thought: str,
        target: Target,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[State, InjectedState],
    ):
        """
        Taps on a UI element identified by the 'target' object.

        The 'target' object allows specifying an element by its resource_id
        (with an optional index), its bounds, its text content (with an optional index),
        or the 1-based [i] of a line in the OCR text block (ocr_index).
        The tool uses a fallback strategy, trying the locators in that order.
        """
        # Track all attempts for better error reporting
        attempts: list[dict] = []
        success = False
        successful_selector: str | None = None

        controller = UnifiedMobileController(ctx)

        # 0. Reduce ocr_index to concrete bounds BEFORE anything else.
        #
        # `ocr_index` is the 1-based [i] printed in the OCR block, so it resolves
        # against the very list the model was shown (state.latest_ocr_results is
        # deliberately kept alive by cortex for exactly this). A resolved OCR box is
        # a legitimate coordinate source, so it enters the same chain as an
        # explicit bounds below and goes through the same validation.
        ocr_texts: list[OcrText] | None = state.latest_ocr_results or None
        ocr_hit: OcrText | None = None
        if target.ocr_index is not None:
            ocr_hit = resolve_ocr_index(ocr_texts, target.ocr_index)
            if ocr_hit is None:
                # Out of range / no OCR data. Never silently tap something else:
                # say exactly why the index could not be used and keep going with
                # whatever else the target carries.
                available = len(ocr_texts) if ocr_texts else 0
                attempts.append(
                    {
                        "selector": f"ocr_index={target.ocr_index}",
                        "error": (
                            "no OCR results available for this screen"
                            if not ocr_texts
                            else f"out of range; the OCR block had {available} line(s), "
                            f"valid range is 1..{available}"
                        ),
                    }
                )
            else:
                target = target.model_copy(update={"bounds": ocr_hit.bounds})

        # Validate target has at least one selector
        if not has_valid_selectors(target):
            attempts.append(
                {
                    "selector": "none",
                    "error": (
                        "No valid selector provided "
                        "(need bounds, resource_id, text, or ocr_index)"
                    ),
                }
            )

        # 1. Try with COORDINATES FIRST (visual approach)
        if not success and target.bounds:
            center = target.bounds.get_center()
            if ocr_hit is not None:
                # Echo what actually got tapped so the agent can see the mapping
                # instead of trusting an opaque index.
                selector_info = (
                    f"ocr_index={target.ocr_index} text='{ocr_hit.text}' "
                    f"coordinates ({center.x}, {center.y})"
                )
            else:
                selector_info = f"coordinates ({center.x}, {center.y})"

            # Validate bounds before attempting
            bounds_error = validate_coordinates_bounds(
                target, ctx.device.device_width, ctx.device.device_height
            )
            if bounds_error:
                logger.warning(f"Coordinates out of bounds: {bounds_error}")
                attempts.append(
                    {"selector": selector_info, "error": f"Out of bounds: {bounds_error}"}
                )
            else:
                # 屏幕矩形内不代表点得对：坐标若是从缩放后的截图上目测来的，
                # 一样能落在屏幕内但会点空。要求它与层级树中的真实元素高度重合。
                derived_ok, derived_reason = await coordinates_look_like_hierarchy_derived(
                    ctx=ctx, target=target, ocr_texts=ocr_texts
                )
                if not derived_ok:
                    logger.warning(f"Coordinates rejected: {derived_reason}")
                    attempts.append({"selector": selector_info, "error": derived_reason})
                else:
                    try:
                        center_point = target.bounds.get_center()
                        logger.info(f"Attempting tap with {selector_info}")
                        result = await controller.tap_at(x=center_point.x, y=center_point.y)
                        if result.error is None:
                            success = True
                            successful_selector = selector_info
                        else:
                            error_msg = result.error
                            logger.warning(f"Tap with {selector_info} failed: {error_msg}")
                            attempts.append({"selector": selector_info, "error": error_msg})
                    except Exception as e:
                        logger.warning(f"Exception during tap with {selector_info}: {e}")
                        attempts.append({"selector": selector_info, "error": str(e)})

        # 2. If coordinates failed or weren't provided, try with resource_id
        if not success and target.resource_id:
            selector_info = f"resource_id='{target.resource_id}' (index={target.resource_id_index})"
            try:
                logger.info(f"Attempting tap with {selector_info}")
                result = await controller.tap_element(
                    resource_id=target.resource_id,
                    index=target.resource_id_index or 0,
                )
                if result.error is None:
                    success = True
                    successful_selector = selector_info
                else:
                    error_msg = result.error
                    logger.warning(f"Tap with {selector_info} failed: {error_msg}")
                    attempts.append({"selector": selector_info, "error": error_msg})
            except Exception as e:
                logger.warning(f"Exception during tap with {selector_info}: {e}")
                attempts.append({"selector": selector_info, "error": str(e)})

        # 3. If resource_id failed or wasn't provided, try with text (last resort)
        if not success and target.text:
            selector_info = f"text='{target.text}' (index={target.text_index})"
            try:
                logger.info(f"Attempting tap with {selector_info}")
                result = await controller.tap_element(
                    text=target.text,
                    index=target.text_index or 0,
                )
                if result.error is None:
                    success = True
                    successful_selector = selector_info
                else:
                    error_msg = result.error
                    logger.warning(f"Tap with {selector_info} failed: {error_msg}")
                    attempts.append({"selector": selector_info, "error": error_msg})
            except Exception as e:
                logger.warning(f"Exception during tap with {selector_info}: {e}")
                attempts.append({"selector": selector_info, "error": str(e)})

        # Build result message
        if success:
            agent_outcome = tap_wrapper.on_success_fn(successful_selector)
        else:
            # Build detailed failure message with all attempts
            failure_details = "; ".join([f"{a['selector']}: {a['error']}" for a in attempts])
            agent_outcome = tap_wrapper.on_failure_fn(failure_details)

        tool_message = ToolMessage(
            tool_call_id=tool_call_id,
            content=agent_outcome,
            additional_kwargs={"attempts": attempts} if not success else {},
            status="success" if success else "error",
        )
        return Command(
            update=await state.asanitize_update(
                ctx=ctx,
                update={
                    "agents_thoughts": [agent_thought, agent_outcome],
                    EXECUTOR_MESSAGES_KEY: [tool_message],
                },
                agent="executor",
            ),
        )

    return tap


tap_wrapper = ToolWrapper(
    tool_fn_getter=get_tap_tool,
    on_success_fn=lambda selector_info: f"Tap on element with {selector_info} was successful.",
    on_failure_fn=lambda failure_details: f"Failed to tap on element. Attempts: {failure_details}",
)
