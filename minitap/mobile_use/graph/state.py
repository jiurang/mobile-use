from typing import Annotated

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel

from minitap.mobile_use.agents.planner.types import Subgoal
from minitap.mobile_use.config import AgentNode
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.services.ocr.types import OcrText
from minitap.mobile_use.utils.failed_targets import FailedTarget
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.recorder import record_interaction
from minitap.mobile_use.utils.screen_elements import ScreenElement

logger = get_logger(__name__)


def take_last(a, b):
    return b


def merge_dicts(a, b):
    """Merge two dicts; keys in b overwrite keys in a."""
    return {**a, **b}


class State(BaseModel):
    messages: Annotated[list[AnyMessage], "Sequential messages", add_messages]
    remaining_steps: Annotated[int | None, "Remaining steps before the task is completed"] = None

    # planner related keys
    initial_goal: Annotated[str, "Initial goal given by the user"]

    # orchestrator related keys
    subgoal_plan: Annotated[list[Subgoal], "The current plan, made of subgoals"]

    # planner related keys（计划日志）
    # 计划本身会被 replan 换掉（completed + new），直接把它渲染给 cortex 会让提示词
    # 从被改写那一行起整段失效。日志只追加：每生成一版计划追加一段，旧段永不改动。
    subgoal_plan_log: Annotated[
        str,
        "Append-only log of every generated plan; the last PLAN section is the active one",
    ] = ""
    plan_version: Annotated[int, "How many plans have been generated so far"] = 0

    # contextor related keys
    latest_ui_hierarchy: Annotated[
        list[dict] | None, "Latest UI hierarchy of the device", take_last
    ]
    latest_ocr_results: Annotated[
        list[OcrText] | None,
        "Latest OCR results of the device screen (text + bounds + confidence)",
        take_last,
    ]
    latest_screenshot: Annotated[str | None, "Latest screenshot base64 of the device", take_last]
    focused_app_info: Annotated[str | None, "Focused app info", take_last]
    device_date: Annotated[str | None, "Date of the device", take_last]

    # 屏幕元素表：UI 层级树 + OCR 文本合并后的统一索引表。
    #
    # 为什么它必须活到 executor：`Target.index` 要还原成「模型当时看到的那
    # 个元素」，而索引只在**那一次裁剪**上成立。执行时重新 dump 再重新编号
    # 是不行的 —— 屏可能已经变了，编号会和模型看到的不一致。
    # 所以这张表和 `latest_ocr_results` 一样，cortex 不清空它。
    latest_screen_elements: Annotated[
        list[ScreenElement] | None,
        "Screen element table (UI tree + OCR, one shared 1-based index)",
        take_last,
    ] = None

    # 上一次「动过屏幕」的动作发生在什么时候（epoch 秒）。
    #
    # contextor 用它算「距离上一个动作多久了」：不够久就先等一等再抓屏。
    # 真机实测换页后新控件要 1~2s 才挂上，抓早了会拿到一棵控件没挂全的树。
    # 只有真的动过屏幕的动作才记时间 —— 记笔记之类的动作不记。
    last_action_at: Annotated[
        float | None,
        "Epoch seconds when the last screen-changing action was dispatched",
        take_last,
    ] = None

    # 连续失败的目标账本（见 utils/failed_targets.py）。
    #
    # 它同时是**熔断器**和**提示词素材**：连着失败同一目标到阈值后，工具
    # 不再执行它（避免原地打转、也避免点了别的东西）；下一轮提示词里会把
    # 这些目标列出来，告诉模型「这些已经不在屏上了，不要再用」。
    failed_targets: Annotated[
        list[FailedTarget],
        "Consecutive failed targets (the last one may be blocking further attempts)",
        take_last,
    ] = []

    # cortex related keys
    structured_decisions: Annotated[
        str | None,
        "Structured decisions made by the cortex, for the executor to follow",
        take_last,
    ]
    complete_subgoals_by_ids: Annotated[
        list[str],
        "List of subgoal IDs to complete",
        take_last,
    ]

    # executor related keys
    executor_messages: Annotated[list[AnyMessage], "Sequential Executor messages", add_messages]
    cortex_last_thought: Annotated[str | None, "Last thought of the cortex for the executor"]

    # common keys
    agents_thoughts: Annotated[
        list[str],
        "All thoughts and reasons that led to actions (why a tool was called, expected outcomes..)",
        take_last,
    ]

    # scratchpad for explicit memory
    scratchpad: Annotated[
        dict[str, str],
        "Persistent key-value storage for notes the agent can save and retrieve",
        merge_dicts,
    ] = {}

    async def asanitize_update(
        self,
        ctx: MobileUseContext,
        update: dict,
        agent: AgentNode | None = None,
    ):
        """
        Sanitizes the state update to ensure it is valid and apply side effect logic where required.
        The agent is required if the update contains the "agents_thoughts" key.
        """
        updated_agents_thoughts: str | list[str] | None = update.get("agents_thoughts", None)
        if updated_agents_thoughts is not None:
            if isinstance(updated_agents_thoughts, str):
                updated_agents_thoughts = [updated_agents_thoughts]
            elif isinstance(updated_agents_thoughts, list):
                updated_agents_thoughts = [t for t in updated_agents_thoughts if t is not None]
            else:
                raise ValueError("agents_thoughts must be a str or list[str]")

            if agent is None:
                raise ValueError("Agent is required when updating the 'agents_thoughts' key")
            update["agents_thoughts"] = await _add_agent_thoughts(
                ctx=ctx,
                old=self.agents_thoughts,
                new=updated_agents_thoughts,
                agent=agent,
            )
        return update


async def _add_agent_thoughts(
    ctx: MobileUseContext,
    old: list[str],
    new: list[str],
    agent: AgentNode,
) -> list[str]:
    if ctx.on_agent_thought:
        for thought in new:
            await ctx.on_agent_thought(agent, thought)

    named_thoughts = [f"[{agent}] {thought}" for thought in new]
    if (
        ctx.execution_setup
        and ctx.execution_setup.traces_path is not None
        and ctx.execution_setup.trace_name is not None
    ):
        await record_interaction(ctx, response=AIMessage(content=str(named_thoughts)))
    return old + named_thoughts
