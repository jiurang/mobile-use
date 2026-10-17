from pathlib import Path

from jinja2 import Template
from langchain_core.messages import (
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from minitap.mobile_use.agents.cortex.types import CortexOutput
from minitap.mobile_use.agents.planner.utils import get_current_subgoal
from minitap.mobile_use.constants import EXECUTOR_MESSAGES_KEY
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.controllers.controller_factory import create_device_controller
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.services.llm import get_llm, invoke_llm_with_timeout_message, with_fallback
from minitap.mobile_use.services.telemetry import telemetry
from minitap.mobile_use.tools.index import (
    EXECUTOR_WRAPPERS_TOOLS,
    VIDEO_RECORDING_WRAPPERS,
    format_tools_list,
)
from minitap.mobile_use.utils.conversations import get_screenshot_message_for_llm
from minitap.mobile_use.utils.decorators import wrap_with_callbacks
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.ocr_view import render_ocr_results
from minitap.mobile_use.utils.ui_hierarchy_view import render_ui_hierarchy

logger = get_logger(__name__)

# state.py:113 的 reducer 只增不减，而 cortex 每轮全量重发，所以这里要自己封顶。
# 取值与 contextor.MAX_AGENTS_THOUGHTS 保持一致。
MAX_CORTEX_AGENTS_THOUGHTS = 25


class CortexNode:
    def __init__(self, ctx: MobileUseContext):
        self.ctx = ctx

    @wrap_with_callbacks(
        before=lambda: logger.info("Starting Cortex Agent..."),
        on_success=lambda _: logger.success("Cortex Agent"),
        on_failure=lambda _: logger.error("Cortex Agent"),
    )
    async def __call__(self, state: State):
        executor_feedback = get_executor_agent_feedback(state)

        current_locked_app_package = (
            self.ctx.execution_setup.get_locked_app_package() if self.ctx.execution_setup else None
        )

        executor_wrappers = list(EXECUTOR_WRAPPERS_TOOLS)
        if self.ctx.video_recording_enabled:
            executor_wrappers.extend(VIDEO_RECORDING_WRAPPERS)

        system_message = Template(
            Path(__file__).parent.joinpath("cortex.md").read_text(encoding="utf-8")
        ).render(
            platform=self.ctx.device.mobile_platform.value,
            initial_goal=state.initial_goal,
            subgoal_plan=state.subgoal_plan,
            current_subgoal=get_current_subgoal(state.subgoal_plan),
            executor_feedback=executor_feedback,
            executor_tools_list=format_tools_list(ctx=self.ctx, wrappers=executor_wrappers),
            locked_app_package=current_locked_app_package,
            perception=self.ctx.perception,
        )
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(
                content="Here are my device info:\n"
                + self.ctx.device.to_str()
                + f"Device date: {state.device_date}\n"
                if state.device_date
                else "" + f"Focused app info: {state.focused_app_info}\n"
                if state.focused_app_info
                else ""
            ),
        ]
        # 这些 thought 绝不能塞成 assistant 回合。state.py:113 给每条加了 "[role] "
        # 前缀，模型会把它们当成 few-shot 照抄，于是无视 tool_choice 改用纯文本回答，
        # _parse() 再把这段中文当 JSON 解析 -> pydantic json_invalid -> 整个 goal 挂掉。
        # 压成单个 user 回合后，MiniMax-M3 的 tool_calls 实测恢复 2/2。
        # 见 v/probe_cortex_toolcall_ab.py、v/probe_cortex_history_len.py。
        thoughts = state.agents_thoughts[-MAX_CORTEX_AGENTS_THOUGHTS:]
        if thoughts:
            messages.append(
                HumanMessage(
                    content="Agent thoughts history (newest last):\n" + "\n".join(thoughts)
                )
            )

        if self.ctx.perception.ui_hierarchy and state.latest_ui_hierarchy:
            # 这里只为省 token 做展示层裁剪；state.latest_ui_hierarchy 本身保持完整，
            # 因为工具要拿它来解析 target。
            messages.append(HumanMessage(content=render_ui_hierarchy(state.latest_ui_hierarchy)))
        if self.ctx.perception.ocr and state.latest_ocr_results:
            messages.append(HumanMessage(content=render_ocr_results(state.latest_ocr_results)))
        if self.ctx.perception.screenshot and state.latest_screenshot:
            controller = create_device_controller(self.ctx)
            compressed_image_base64 = controller.get_compressed_b64_screenshot(
                state.latest_screenshot
            )
            messages.append(get_screenshot_message_for_llm(compressed_image_base64))

        llm = get_llm(ctx=self.ctx, name="cortex", temperature=1).with_structured_output(
            CortexOutput
        )
        llm_fallback = get_llm(
            ctx=self.ctx, name="cortex", use_fallback=True, temperature=1
        ).with_structured_output(CortexOutput)
        response: CortexOutput = await with_fallback(
            main_call=lambda: invoke_llm_with_timeout_message(
                llm.ainvoke(messages),
                agent_name="cortex",
                task_id=self.ctx.trace_id,
            ),
            fallback_call=lambda: invoke_llm_with_timeout_message(
                llm_fallback.ainvoke(messages),
                agent_name="cortex",
                task_id=self.ctx.trace_id,
                used_fallback=True,
            ),
        )  # type: ignore

        EMPTY_STRING_TOKENS = ["{}", "[]", "null", "", "None"]

        if response.decisions in EMPTY_STRING_TOKENS:
            response.decisions = None
        if response.goals_completion_reason in EMPTY_STRING_TOKENS:
            response.goals_completion_reason = None

        thought_parts = []
        if response.decisions_reason:
            thought_parts.append(f"Decisions reason: {response.decisions_reason}")
        if response.goals_completion_reason:
            thought_parts.append(f"Goals completion reason: {response.goals_completion_reason}")

        agent_thought = "\n\n".join(thought_parts)

        # Capture cortex decision telemetry (only non-sensitive flags)
        telemetry.capture_cortex_decision(
            task_id=self.ctx.trace_id,
            has_decisions=response.decisions is not None,
            has_goals_completion=response.goals_completion_reason is not None,
            completed_subgoals_count=len(response.complete_subgoals_by_ids or []),
        )

        return await state.asanitize_update(
            ctx=self.ctx,
            update={
                "agents_thoughts": [agent_thought],
                "structured_decisions": response.decisions,
                "complete_subgoals_by_ids": response.complete_subgoals_by_ids,
                "latest_ui_hierarchy": None,
                "latest_ocr_results": None,
                "latest_screenshot": None,
                "focused_app_info": None,
                "device_date": None,
                # Executor related fields
                EXECUTOR_MESSAGES_KEY: [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
                "cortex_last_thought": agent_thought,
            },
            agent="cortex",
        )


def get_executor_agent_feedback(state: State) -> str:
    if state.structured_decisions is None:
        return "None."
    executor_tool_messages = [m for m in state.executor_messages if isinstance(m, ToolMessage)]
    return (
        f"Latest UI decisions:\n{state.structured_decisions}"
        + "\n\n"
        + f"Executor feedback:\n{executor_tool_messages}"
    )
