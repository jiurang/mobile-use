import asyncio
import os
from functools import cache
from pathlib import Path

from jinja2 import Template
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
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
from minitap.mobile_use.utils.ui_hierarchy_view import render_hierarchy_and_labels

logger = get_logger(__name__)

# state.py:113 的 reducer 只增不减，而 cortex 每轮全量重发，所以这里要自己封顶。
# 取值与 contextor.MAX_AGENTS_THOUGHTS 保持一致。
MAX_CORTEX_AGENTS_THOUGHTS = 25

# 模型偶尔会用这些占位串表示“没有内容”，统一当作 None 处理。
EMPTY_STRING_TOKENS = frozenset({"{}", "[]", "null", "None", ""})

CORTEX_PROMPT_ENV = "MOBILE_USE_CORTEX_PROMPT"


@cache
def _compile_template(template_text: str) -> Template:
    """按模板文本缓存编译结果，避免每一步都重新编译 Jinja 模板。"""
    return Template(template_text)


def _normalize_optional_str(value: str | None) -> str | None:
    if value is None or value.strip() in EMPTY_STRING_TOKENS:
        return None
    return value


def load_cortex_prompt() -> str:
    """读 cortex 的 system prompt 模板。

    MOBILE_USE_CORTEX_PROMPT 可以指向另一个 .md（例如同目录的 cortex.zh.md），
    用来做中/英文提示词的 A/B 实测 —— 同一份 context 分别渲染两种语言，
    再各自真跑一遍，比较 token 与成功率。

    不设这个变量时读的还是同目录的 cortex.md，行为与从前逐字节一致，
    所以对生产是 no-op。路径不可读时只 warning 并回落到默认模板：
    指针写错不该让整条任务挂掉，降级 + 上报即可。
    """
    override = os.getenv(CORTEX_PROMPT_ENV)
    if override:
        path = Path(override).expanduser()
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8")
            logger.warning(
                f"[cortex] {CORTEX_PROMPT_ENV}={override!r} 不是可读文件，回落到默认 cortex.md"
            )
        except OSError as e:
            logger.warning(
                f"[cortex] 读取 {CORTEX_PROMPT_ENV}={override!r} 失败（{e}），回落到 cortex.md"
            )
    return Path(__file__).parent.joinpath("cortex.md").read_text(encoding="utf-8")


async def _invoke_cortex(
    llm,
    messages: list[BaseMessage],
    ctx: MobileUseContext,
    known_subgoal_ids: set[str],
    used_fallback: bool = False,
) -> CortexOutput:
    """调用 LLM 并校验输出。

    先过滤不存在的子目标 ID，再判断响应是否为空。空响应（没有 decisions、
    没有推理、没有完成原因、也没有有效的待完成子目标）无法推进任务，直接抛错
    让 with_fallback 切到备用模型，避免白耗 step 直到 recursion_limit。
    """
    response = await invoke_llm_with_timeout_message(
        llm.ainvoke(messages),
        agent_name="cortex",
        task_id=ctx.trace_id,
        used_fallback=used_fallback,
    )
    if response is None:
        raise ValueError("Cortex LLM returned no response")

    response.decisions = _normalize_optional_str(response.decisions)
    response.decisions_reason = _normalize_optional_str(response.decisions_reason)
    response.goals_completion_reason = _normalize_optional_str(response.goals_completion_reason)

    unknown_subgoal_ids = [
        subgoal_id
        for subgoal_id in response.complete_subgoals_by_ids
        if subgoal_id not in known_subgoal_ids
    ]
    if unknown_subgoal_ids:
        logger.warning(f"[cortex] Ignoring unknown subgoal IDs: {unknown_subgoal_ids}")
    response.complete_subgoals_by_ids = [
        subgoal_id
        for subgoal_id in response.complete_subgoals_by_ids
        if subgoal_id in known_subgoal_ids
    ]

    if (
        response.decisions is None
        and response.decisions_reason is None
        and response.goals_completion_reason is None
        and not response.complete_subgoals_by_ids
    ):
        raise ValueError("Cortex LLM returned an empty response")
    return response


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

        system_message = _compile_template(load_cortex_prompt()).render(
            platform=self.ctx.device.mobile_platform.value,
            initial_goal=state.initial_goal,
            subgoal_plan="\n".join(str(subgoal) for subgoal in state.subgoal_plan),
            current_subgoal=get_current_subgoal(state.subgoal_plan),
            executor_feedback=executor_feedback,
            executor_tools_list=format_tools_list(ctx=self.ctx, wrappers=executor_wrappers),
            locked_app_package=current_locked_app_package,
            perception=self.ctx.perception,
        )
        device_info_lines = ["Here are my device info:", self.ctx.device.to_str()]
        if state.device_date:
            device_info_lines.append(f"Device date: {state.device_date}")
        if state.focused_app_info:
            device_info_lines.append(f"Focused app info: {state.focused_app_info}")
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content="\n".join(device_info_lines)),
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
            tree_text, _ = render_hierarchy_and_labels(state.latest_ui_hierarchy)
            if tree_text:
                messages.append(HumanMessage(content=tree_text))
        if self.ctx.perception.ocr and state.latest_ocr_results:
            # 不去重：OCR 行的 [i] 是 Target.ocr_index 的取值，
            # 剔掉任何一条都会让模型手里的行号和 results 的下标对不上。
            ocr_text = render_ocr_results(state.latest_ocr_results)
            if ocr_text:
                messages.append(HumanMessage(content=ocr_text))
        if self.ctx.perception.screenshot and state.latest_screenshot:
            controller = create_device_controller(self.ctx)
            # 压缩是同步 PIL 操作，放到线程里避免阻塞事件循环。
            compressed_image_base64 = await asyncio.to_thread(
                controller.get_compressed_b64_screenshot, state.latest_screenshot
            )
            messages.append(get_screenshot_message_for_llm(compressed_image_base64))

        llm = get_llm(ctx=self.ctx, name="cortex", temperature=1).with_structured_output(
            CortexOutput
        )
        llm_fallback = get_llm(
            ctx=self.ctx, name="cortex", use_fallback=True, temperature=1
        ).with_structured_output(CortexOutput)
        known_subgoal_ids = {subgoal.id for subgoal in state.subgoal_plan}
        response: CortexOutput = await with_fallback(
            main_call=lambda: _invoke_cortex(llm, messages, self.ctx, known_subgoal_ids),
            fallback_call=lambda: _invoke_cortex(
                llm_fallback, messages, self.ctx, known_subgoal_ids, used_fallback=True
            ),
        )

        valid_subgoal_ids = response.complete_subgoals_by_ids

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
            completed_subgoals_count=len(valid_subgoal_ids),
        )

        update = {
            "structured_decisions": response.decisions,
            "complete_subgoals_by_ids": valid_subgoal_ids,
            "latest_ui_hierarchy": None,
            # latest_ocr_results 故意不清空：渲染给模型的 [i] 就是
            # Target.ocr_index 的取值，executor 工具执行时需要同一份
            # results 才能把索引还原成 bounds。图序是
            # contextor -> cortex -> executor -> executor_tools -> ... -> contextor，
            # 所以 executor_tools 跑的时候这份感知数据正是当屏的，
            # 下一次 contextor 会整体覆盖它。
            # latest_ui_hierarchy 仍然清空 —— 它没有索引契约，
            # 工具侧走 controller.get_ui_elements() 现取即可。
            "latest_screenshot": None,
            "focused_app_info": None,
            "device_date": None,
            # Executor related fields
            EXECUTOR_MESSAGES_KEY: [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
            "cortex_last_thought": agent_thought or None,
        }
        # 没有任何推理内容时不要把空字符串写进历史（会触发回调并污染 trace）。
        if agent_thought:
            update["agents_thoughts"] = [agent_thought]

        return await state.asanitize_update(
            ctx=self.ctx,
            update=update,
            agent="cortex",
        )


def get_executor_agent_feedback(state: State) -> str:
    if state.structured_decisions is None:
        return "None."
    feedback_parts = [
        str(message) for message in state.executor_messages if isinstance(message, ToolMessage)
    ]
    # 走 LLM 兜底路径时 executor 可能只回一条纯文本（没有 tool_calls），
    # 这段内容同样是 cortex 需要看到的执行结果，不能丢。
    last_ai_content = next(
        (
            message.content.strip()
            for message in reversed(state.executor_messages)
            if isinstance(message, AIMessage)
            and isinstance(message.content, str)
            and message.content.strip()
        ),
        None,
    )
    if last_ai_content:
        feedback_parts.append(last_ai_content)
    return (
        f"Latest UI decisions:\n{state.structured_decisions}"
        + "\n\n"
        + f"Executor feedback:\n{feedback_parts}"
    )
