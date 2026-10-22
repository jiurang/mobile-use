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
from minitap.mobile_use.tools.live_target import resolve_live_bounds_by_id
from minitap.mobile_use.tools.tool_wrapper import ToolWrapper
from minitap.mobile_use.tools.types import Target
from minitap.mobile_use.tools.utils import (
    check_failure_breaker,
    has_valid_selectors,
    record_target_failure,
    record_target_failure_fact,
    resolve_target,
    target_label,
    validate_coordinates_bounds,
)
from minitap.mobile_use.utils.failed_targets import FailedTarget
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.screen_fingerprint import build_context, format_context

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
        blocked = False

        controller = UnifiedMobileController(ctx)

        # 0. 把 index / ocr_index 解析成具体的元素 BEFORE anything else.
        #
        # 索引优先于模型自己抄的坐标 —— 它指向的是模型当时看到的那一行，
        # 坐标和 id 都是代码给的，不会抄错（实测手抄的坐标有 16.8% 直接越界）。
        # 解析不了就把原因记进 attempts，让模型知道是索引错了，
        # 而不是让工具在别的地方乱点一下。
        ocr_texts: list[OcrText] | None = state.latest_ocr_results or None
        target, index_error = resolve_target(state=state, target=target)
        if index_error:
            logger.warning(f"Tap target could not be resolved: {index_error}")
            attempts.append(
                {
                    "selector": (
                        f"index={target.index}"
                        if target.index is not None
                        else f"ocr_index={target.ocr_index}"
                    ),
                    "error": index_error,
                }
            )

        # 0.5 熔断：同一个目标已经连着失败过，就别再点它了。
        #
        # 真机实测里点击失败几乎全是「目标已经不在屏上了」。模型此时如果
        # 继续报同一个目标，注定再失败一次，还要白花一次 1.2s 的 live dump。
        # 这里直接顶回去，并要求它重读屏幕、换一个现在还在的元素。
        breaker = check_failure_breaker(state=state, target=target)
        if breaker:
            logger.warning(f"Tap blocked by failure breaker: {breaker}")
            blocked = True
            attempts.append({"selector": target_label(target), "error": breaker})

        # Validate target has at least one selector
        if not has_valid_selectors(target):
            attempts.append(
                {
                    "selector": "none",
                    "error": (
                        "No valid selector provided "
                        "(need index, bounds, resource_id, text, or ocr_index)"
                    ),
                }
            )

        # 1. 有 resource_id → 按**身份**在当屏重新定位它，再点它的当前位置。
        #
        #    这一步取代了「坐标 + IoU 门限」。门限当初防的是「模型从缩放后的
        #    截图上目测坐标」，那个前提已经不存在了：截图通道在所有实际运行里
        #    都是关的，模型现在只报索引、坐标由代码给，一个字都不用抄。
        #
        #    按身份找没有门限就没有误伤；屏变了（元素移动、动画）也无所谓 ——
        #    找的是同一个元素的**新位置**，而不是拿旧坐标去赌。
        if not success and not blocked and target.resource_id:
            live_bounds, why = await resolve_live_bounds_by_id(
                ctx=ctx,
                resource_id=target.resource_id,
                index=target.resource_id_index or 0,
                hint=target.bounds,
            )
            if live_bounds is None:
                logger.warning(f"Target is not on the current screen: {why}")
                attempts.append({"selector": f"resource_id={target.resource_id}", "error": why})
            else:
                center = live_bounds.get_center()
                selector_info = (
                    f"resource_id='{target.resource_id}' -> coordinates ({center.x}, {center.y})"
                )
                try:
                    logger.info(f"Attempting tap with {selector_info}")
                    result = await controller.tap_at(x=center.x, y=center.y)
                    if result.error is None:
                        success = True
                        successful_selector = selector_info
                    else:
                        logger.warning(f"Tap with {selector_info} failed: {result.error}")
                        attempts.append({"selector": selector_info, "error": result.error})
                except Exception as e:
                    logger.warning(f"Exception during tap with {selector_info}: {e}")
                    attempts.append({"selector": selector_info, "error": str(e)})

        # 2. 没有 resource_id（OCR 行，实测只占模型点击的 4%）但有坐标 → 位置校验。
        #
        #    OCR 文本**不在层级树里**（它就是「树里没有的可见文字」），没有 id
        #    可查，只能靠几何判据确认「这一小块地方现在还是那段文字」。
        #    也不能改成按文本重匹配：OCR 的**内容会变**（点赞数在跳、时钟在走），
        #    位置比内容稳。
        if not success and not blocked and target.bounds and not target.resource_id:
            center = target.bounds.get_center()
            if target.index is not None:
                # Echo what actually got tapped so the agent can see the mapping
                # instead of trusting an opaque index.
                selector_info = f"index={target.index} coordinates ({center.x}, {center.y})"
            elif target.ocr_index is not None:
                # 回显命中的文本，模型能看见索引到底点到了什么
                selector_info = (
                    f"ocr_index={target.ocr_index} text='{target.text}' "
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
                    reason = derived_reason
                    if target.index is not None:
                        # 索引指向的是「模型看到的那一行」。校验不过，说明那一行的
                        # 坐标和**当屏**对不上 —— 屏变了（动画/跳转/列表刷新），
                        # 或者那一行的坐标本来就不可信。这两件事模型必须知道，
                        # 否则它只会把同一个索引原样再报一遍。
                        reason = (
                            f"{derived_reason} (this target came from screen element "
                            f"index {target.index}; the screen may have changed since "
                            f"it was read — re-read the screen and use a fresh index)"
                        )
                    attempts.append({"selector": selector_info, "error": reason})
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

        # 3. If the element wasn't found by id and there's no usable geometry,
        #    try with text (last resort)
        if not success and not blocked and target.text:
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
        failure_details = ""  # 成功时用不到，但要在分支外可见
        if success:
            agent_outcome = tap_wrapper.on_success_fn(successful_selector)
            # 🔴 附上**上下文指纹**（~40 字符）。
            #
            # 为什么：真机实测（2026-10-22，快手）同一个「搜索」入口的 id
            # 4 小时内从 ``right_tv`` 变成 ``thanos_home_top_search``，点赞按钮
            # 从 ``like_button`` 变成 ``like_element_click_layout`` —— 靠 id
            # 回放的宏，app 一发版就全废。而**邻居**（那几个 tab 的文字）
            # 没变。
            #
            # 工具结果文字本来就在下一轮 prompt 的历史里，也进 trace ⇒
            # 一条通道同时解决「宏要能编译」和「宏要能跨版本重定位」两件事。
            if target.resource_id:
                context = build_context(state.latest_screen_elements, target.resource_id)
                ctx_text = format_context(context)
                if ctx_text:
                    agent_outcome = f"{agent_outcome} {ctx_text}"
            # 成功即清账：模型已经换了目标，「连续失败」这件事到此为止。
            failed_targets: list[FailedTarget] = []
        else:
            # Build detailed failure message with all attempts
            failure_details = "; ".join([f"{a['selector']}: {a['error']}" for a in attempts])
            agent_outcome = tap_wrapper.on_failure_fn(failure_details)
            failed_targets = record_target_failure(
                state=state,
                target=target,
                reason=failure_details[:200],
            )

        # 跨子目标的失败事实（只增不减）：熔断账本成功时会清空，
        # 而汇报要交给 Cortex 的「失败事实」恰恰发生在子目标成功收尾的时候。
        if success:
            executor_failures = state.executor_failures
        else:
            executor_failures = record_target_failure_fact(
                state=state,
                target=target,
                reason=failure_details[:200],
            )

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
                    "failed_targets": failed_targets,
                    "executor_failures": executor_failures,
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
