from pathlib import Path

from jinja2 import Template
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai.chat_models import ChatVertexAI

from minitap.mobile_use.agents.cortex.dispatcher import dispatch_decisions
from minitap.mobile_use.constants import EXECUTOR_MESSAGES_KEY
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.services.llm import get_llm, invoke_llm_with_timeout_message, with_fallback
from minitap.mobile_use.services.telemetry import telemetry
from minitap.mobile_use.tools.index import (
    EXECUTOR_WRAPPERS_TOOLS,
    VIDEO_RECORDING_WRAPPERS,
    get_tools_from_wrappers,
)
from minitap.mobile_use.utils.decorators import wrap_with_callbacks
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


class ExecutorNode:
    """把 Cortex 的决策落成工具调用。

    两条路径
    --------
    1. **直连（默认，2026-10-17 加）**：Cortex 的 decisions JSON 里 `action`
       字段本身就是工具名，所以这一步是机械转写，不值得再花一次完整的 LLM
       往返。`cortex/dispatcher.py` 确定性地把它变成 tool_calls。
    2. **LLM 兜底**：dispatcher 任何一步不确定（JSON 写歪了、工具名不认识、
       必填参数缺失）时退回原来那次 Executor 调用，行为和改造前完全一致。

    「按顺序串行执行、一个失败就中止」不依赖 LLM —— 它实现在
    `ExecutorToolNode.__func` 的串行循环里，所以跳过 LLM 不会丢这条语义。
    """

    def __init__(self, ctx: MobileUseContext):
        self.ctx = ctx

    @wrap_with_callbacks(
        before=lambda: logger.info("Starting Executor Agent..."),
        on_success=lambda _: logger.success("Executor Agent"),
        on_failure=lambda _: logger.error("Executor Agent"),
    )
    async def __call__(self, state: State):
        structured_decisions = state.structured_decisions
        if not structured_decisions:
            logger.warning("No structured decisions found.")
            return await state.asanitize_update(
                ctx=self.ctx,
                update={
                    "agents_thoughts": [
                        "No structured decisions found, I cannot execute anything."
                    ],
                },
                agent="executor",
            )

        cortex_last_thought = (
            state.cortex_last_thought
            if state.cortex_last_thought
            else (state.agents_thoughts[-1] if state.agents_thoughts else "")
        )

        executor_wrappers = list(EXECUTOR_WRAPPERS_TOOLS)
        if self.ctx.video_recording_enabled:
            executor_wrappers.extend(VIDEO_RECORDING_WRAPPERS)

        tools = get_tools_from_wrappers(self.ctx, executor_wrappers)

        # ---- 快路径：确定性直连，省掉整一次 LLM 往返 ----
        # thought 源和下面 LLM 路径保持一致（cortex_last_thought 缺失时回退到
        # 最近一条 agent thought），否则直连和兜底两条路径产出的
        # agents_thoughts 会对不上，cortex 的历史就乱了。
        tool_calls = dispatch_decisions(
            decisions=structured_decisions,
            default_thought=cortex_last_thought,
            tools=tools,
        )
        if tool_calls is not None:
            response = AIMessage(content="", tool_calls=tool_calls)
            telemetry.capture_executor_dispatch(task_id=self.ctx.trace_id, dispatched=True)
            return await state.asanitize_update(
                ctx=self.ctx,
                update={
                    "cortex_last_thought": cortex_last_thought,
                    EXECUTOR_MESSAGES_KEY: [response],
                },
                agent="executor",
            )

        telemetry.capture_executor_dispatch(task_id=self.ctx.trace_id, dispatched=False)

        # 到这里才需要 Executor 的 prompt —— 直连成功时这段 Jinja 渲染是白做的。
        system_message = Template(
            Path(__file__).parent.joinpath("executor.md").read_text(encoding="utf-8")
        ).render(platform=self.ctx.device.mobile_platform.value)
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=cortex_last_thought),
            HumanMessage(content=structured_decisions),
            *state.executor_messages,
        ]

        llm = get_llm(ctx=self.ctx, name="executor")
        llm_fallback = get_llm(ctx=self.ctx, name="executor", use_fallback=True)

        llm_bind_tools_kwargs: dict = {
            "tools": tools,
        }

        # ChatGoogleGenerativeAI does not support the "parallel_tool_calls" keyword
        if not isinstance(llm, ChatGoogleGenerativeAI | ChatVertexAI):
            llm_bind_tools_kwargs["parallel_tool_calls"] = True

        llm = llm.bind_tools(**llm_bind_tools_kwargs)
        llm_fallback = llm_fallback.bind_tools(**llm_bind_tools_kwargs)
        response = await with_fallback(
            main_call=lambda: invoke_llm_with_timeout_message(
                llm.ainvoke(messages),
                agent_name="executor",
                task_id=self.ctx.trace_id,
            ),
            fallback_call=lambda: invoke_llm_with_timeout_message(
                llm_fallback.ainvoke(messages),
                agent_name="executor",
                task_id=self.ctx.trace_id,
                used_fallback=True,
            ),
        )
        return await state.asanitize_update(
            ctx=self.ctx,
            update={
                "cortex_last_thought": cortex_last_thought,
                EXECUTOR_MESSAGES_KEY: [response],
            },
            agent="executor",
        )
