"""Graph：Executor 自主执行的新拓扑。

拓扑（方案 §9.1 / §9.4，契约 §W8.2）
------------------------------------

    START → planner → plan_manager
    plan_manager --continue--> cortex
    plan_manager --replan--> planner
    plan_manager --end--> END

    cortex --(有待执行的目标)--> contextor
    cortex --(子目标完成 / 没有待跑目标 / 预算超限)--> plan_manager
    contextor → executor
    executor --(有动作)--> executor_tools
    executor --(done / 步数上限 / 没动作)--> cortex
    executor_tools → contextor

三条设计约束（每一条都有硬理由，不是风格偏好）
--------------------------------------------

1. **元素表契约**（方案 §9.1 / §9.3）：
   ``contextor → executor → executor_tools`` 必须跨在**同一轮**里 ——
   ``executor_tools`` 解析 ``Target.index`` 时读到的必须是 Executor 看到的那张表。
   所以工具执行完**必须回 contextor**（重新抓屏 + 重建表），
   **不能**直接回 executor（那样工具会拿到上一屏的编号，点错元素）。

2. **汇报不是节点**（2026-10-29）：
   ``executor → cortex`` 是**无条件**边。Executor 结束一次尝试（``done=True``）
   之后直接回 Cortex，Cortex 入口用 ``build_executor_report`` 把汇报构建出来
   （纯代码），消费完清掉 ``executor_done``。汇报的唯一消费者是 Cortex，
   所以不需要独立的 reporter 节点，也就少一个 super-step。

3. **看门狗**（方案 §6.5.1，R16）：
   Cortex 的每一次判断之后都会经过 PlanManager（要么 ``review``，要么
   下一轮从它出发）。PlanManager 是纯代码：完成提名由 Cortex 给出，计划级失败
   由「同一子目标下发超过 ``MAX_ISSUES_PER_SUBGOAL`` 次」这条规则兜底；
   没有它，Cortex 会一直重发到撞 ``recursion_limit`` **异常退出**。
"""

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from minitap.mobile_use.agents.contextor.contextor import ContextorNode
from minitap.mobile_use.agents.cortex.cortex import CortexNode
from minitap.mobile_use.agents.executor.executor import ExecutorNode
from minitap.mobile_use.agents.executor.tool_node import ExecutorToolNode
from minitap.mobile_use.agents.plan_manager.plan_manager import (
    PlanManagerNode,
    plan_manager_gate,
    post_cortex_gate,
)
from minitap.mobile_use.agents.planner.planner import PlannerNode
from minitap.mobile_use.constants import EXECUTOR_MESSAGES_KEY
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.tools.index import executor_wrappers, get_tools_from_wrappers
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


def post_executor_gate(
    state: State,
) -> Literal["invoke_tools", "report", "retry"]:
    """Executor 之后的路由。

    * ``invoke_tools`` → ``executor_tools``（执行动作）
    * ``retry``       → ``executor``（**动作被拒了，回 executor 重新决策**）
    * ``report``      → ``cortex``（收尾：Cortex 入口自己构建汇报）

    「有没有动作」从局部历史**最后一条 AIMessage 的 tool_calls** 判断 ——
    Executor 只写 AIMessage（带不带 tool_calls），工具写的是 ToolMessage。

    ⚠️ **``retry`` 这条边是方案 §3.1.1「拒绝理由回灌」能生效的前提**
    （独立验收 2026-10-21 查出的致命 bug 2）。

    拒绝时 Executor 写的是「AIMessage(无 tool_calls) + HumanMessage(拒绝理由)」。
    如果这里只看「最后一条 AIMessage 有没有 tool_calls」，会因为那条 AIMessage
    确实没有 tool_calls 而 `break` → 直接进 cortex 收尾 ——
    **拒绝理由模型永远看不到，Executor 的自主循环在第一次出岔子时就终止了**。

    ⇒ 判据改成「**最后一条消息是不是 HumanMessage**」：
    是 → 说明上一轮动作被拒了、理由已回灌 → 回 executor 重新决策。

    为什么不用 ToolMessage 承载拒绝理由：它没有对应的 assistant tool_call
    （拒绝时根本没派发出去），OpenAI 兼容端点会 400（致命 bug 3）。
    """
    if state.executor_done:
        # 目标达成 / 步数上限：不再执行动作，直接收尾。
        logger.info("[graph] Executor is done; going to cortex")
        return "report"

    messages = state.executor_messages or []
    if messages and isinstance(messages[-1], HumanMessage):
        # 上一轮的动作被 dispatcher 拒了，理由已经回灌。
        # 屏幕没变过（动作根本没派发），所以**不用**重新抓屏，直接重新决策。
        logger.info("[graph] Previous action was rejected; going back to executor")
        return "retry"

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                logger.info(f"[graph] Executing {len(tool_calls)} tool call(s):")
                for tool_call in tool_calls:
                    logger.info(f"[graph]  - {tool_call}")
                return "invoke_tools"
            break

    # 兜底：没有动作可执行 → 也去 cortex 收尾（不能让流程卡在这儿）。
    logger.info("[graph] No tool calls to execute; going to cortex")
    return "report"


async def get_graph(
    ctx: MobileUseContext,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """构建整张图。

    ``checkpointer`` 只影响「状态能不能在进程重启后复原」，不改拓扑、不改判定：
    传入时每个 Turn 以 ``<session_id>:<turn_id>`` 为 thread，崩溃后可以从最后
    一个 checkpoint 续跑；不传时就是原来的纯内存运行。
    """
    graph_builder = StateGraph(State)

    ## Define nodes
    graph_builder.add_node("planner", PlannerNode(ctx))
    graph_builder.add_node("plan_manager", PlanManagerNode(ctx))

    graph_builder.add_node("contextor", ContextorNode(ctx))

    graph_builder.add_node("cortex", CortexNode(ctx))

    graph_builder.add_node("executor", ExecutorNode(ctx))

    executor_tools = get_tools_from_wrappers(
        ctx=ctx, wrappers=executor_wrappers(ctx)
    )

    executor_tool_node = ExecutorToolNode(
        tools=executor_tools,
        messages_key=EXECUTOR_MESSAGES_KEY,
        trace_id=ctx.trace_id,
    )
    graph_builder.add_node("executor_tools", executor_tool_node)

    ## Linking nodes
    graph_builder.add_edge(START, "planner")
    graph_builder.add_edge("planner", "plan_manager")

    # 计划层汇合点：PlanManager 之后直接判 continue / replan / end。
    graph_builder.add_conditional_edges(
        "plan_manager",
        plan_manager_gate,
        {
            "continue": "cortex",
            "replan": "planner",
            "end": END,
        },
    )

    # Cortex 下发目标 → 抓屏；完成提名 / 预算超限 → PlanManager。
    graph_builder.add_conditional_edges(
        "cortex",
        post_cortex_gate,
        {
            "execute": "contextor",
            "review": "plan_manager",
            "budget_exceeded": "plan_manager",
        },
    )

    # 抓屏 → 决策一步。
    graph_builder.add_edge("contextor", "executor")

    # 有动作 → 执行；做完了 → 交给 Cortex 收尾（它在入口构建汇报）；被拒了 →
    # 回 executor 重新决策。
    graph_builder.add_conditional_edges(
        "executor",
        post_executor_gate,
        {
            "invoke_tools": "executor_tools",
            "report": "cortex",
            # 动作被拒 → 回 executor（屏幕没变，不用重新抓屏）
            "retry": "executor",
        },
    )

    # 工具执行完 → **回 contextor**（重新抓屏、重建元素表）。
    # 不能直接回 executor：那样工具解析 Target.index 时拿到的是别的屏的编号。
    graph_builder.add_edge("executor_tools", "contextor")

    return graph_builder.compile(checkpointer=checkpointer)
