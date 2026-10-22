"""Graph：Executor 自主执行的新拓扑。

拓扑（方案 §9.1 / §9.4，契约 §W8.2）
------------------------------------

    START → planner → orchestrator → convergence
    convergence --continue--> cortex
    convergence --replan--> planner
    convergence --end--> END

    cortex --(有待执行的目标)--> contextor
    cortex --(子目标完成 / 没有待跑目标)--> orchestrator
    contextor → executor
    executor --(有动作)--> executor_tools
    executor --(done / 步数上限 / 没动作)--> reporter
    executor_tools → summarizer → contextor
    reporter → orchestrator
    orchestrator → convergence

三条设计约束（每一条都有硬理由，不是风格偏好）
--------------------------------------------

1. **元素表契约**（方案 §9.1 / §9.3）：
   ``contextor → executor → executor_tools`` 必须跨在**同一轮**里 ——
   ``executor_tools`` 解析 ``Target.index`` 时读到的必须是 Executor 看到的那张表。
   所以工具执行完**必须回 contextor**（重新抓屏 + 重建表），
   **不能**直接回 executor（那样工具会拿到上一屏的编号，点错元素）。

2. **看门狗**（方案 §6.5.1，R16）：
   ``reporter → orchestrator`` 是**无条件**边 —— 每个子目标结束都让 Orchestrator
   看一眼计划。没有它，计划级失败（子目标根本做不到）就没有任何处理者：
   Cortex 会一直说「没完成」一直重发，最后撞 ``recursion_limit`` **异常退出**
   （不是「重新拆计划」）。

3. **汇报是新旧有别的**：
   ``state.executor_report`` 每子目标变 —— Reporter 写一份、Cortex 读一份。
   读过的汇报必须作废，否则下一轮 Cortex 会把**同一份**汇报再判一遍
   （「完成了 → 推进到下一个子目标」），中间的子目标就被跳过了。
   作废由 :class:`ReportLedger`（一次运行级的账本）记住「已经判过哪一份」，
   由 :func:`make_convergence_node` 在 state 里真正清掉。
"""

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from minitap.mobile_use.agents.contextor.contextor import ContextorNode
from minitap.mobile_use.agents.cortex.cortex import CortexNode
from minitap.mobile_use.agents.executor.executor import ExecutorNode
from minitap.mobile_use.agents.executor.tool_node import ExecutorToolNode
from minitap.mobile_use.agents.executor.types import ExecutorReport
from minitap.mobile_use.agents.orchestrator.orchestrator import OrchestratorNode
from minitap.mobile_use.agents.planner.planner import PlannerNode
from minitap.mobile_use.agents.planner.utils import (
    all_completed,
    get_current_subgoal,
    one_of_them_is_failure,
)
from minitap.mobile_use.agents.reporter.reporter import ReporterNode
from minitap.mobile_use.agents.summarizer.summarizer import SummarizerNode
from minitap.mobile_use.constants import EXECUTOR_MESSAGES_KEY
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.tools.index import (
    EXECUTOR_WRAPPERS_TOOLS,
    VIDEO_RECORDING_WRAPPERS,
    get_tools_from_wrappers,
)
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


class ReportLedger:
    """记住「哪一份汇报已经被 Cortex 判过」。

    为什么需要它
    ------------
    ``state.executor_report`` 是**每子目标变**的字段：Reporter 写一份、Cortex 读一份。
    读过的汇报必须作废 —— 不作废的话，下一次进 Cortex 会把**同一份**汇报再判一遍
    （「完成了 → 推进到下一个子目标」），中间的子目标就被跳过了。

    契约给的两个办法（清空 state 字段 / 加一个「本轮已验收」标记字段）都要动
    ``graph/state.py``，那个文件不在 W8 的名下。所以这里用**一次运行级的账本**：
    ``get_graph(ctx)`` 每次建图都新建一个，跟着这次任务的生命周期。

    汇报对象每个子目标新建一个，所以用**对象身份**就能区分新旧。
    这里持有强引用，保证 ``id`` 不会被回收后复用（否则两份汇报可能被误判成同一份）。
    """

    def __init__(self) -> None:
        self._seen: ExecutorReport | None = None

    def is_seen(self, report: ExecutorReport | None) -> bool:
        """这份汇报是不是已经判过了。"""
        return report is not None and report is self._seen

    def mark_seen(self, report: ExecutorReport) -> None:
        """记下「这份已经判过了」。"""
        self._seen = report


def convergence_gate(
    state: State,
) -> Literal["continue", "replan", "end"]:
    """Check if all subgoals are completed at convergence point."""
    logger.info("Starting convergence_gate")

    if one_of_them_is_failure(state.subgoal_plan):
        logger.info("One of the subgoals is in failure state, asking to replan")
        return "replan"

    if all_completed(state.subgoal_plan):
        logger.info("All subgoals are completed, ending the goal")
        return "end"

    if not get_current_subgoal(state.subgoal_plan):
        logger.info("No subgoal running, ending the goal")
        return "end"

    return "continue"


def post_cortex_gate(
    state: State,
    ledger: ReportLedger,
) -> Literal["execute", "review"]:
    """Cortex 之后的路由。

    * ``review``  → Orchestrator：子目标完成的**正式验收**（标 SUCCESS → convergence）；
      没有待执行的目标时也走这里 —— 看门狗：计划级失败得有人处理。
    * ``execute`` → Contextor：抓屏 → Executor 跑当前目标。

    汇报的新旧
    ----------
    ``executor_report`` 每子目标变，**必须**判它是不是新的：
    已经判过的那份（ledger 记得）不再触发验收，否则同一份汇报会被判两遍。

    目标的新旧
    ----------
    ``executor_steps == 0`` 是「目标刚下发」的信号：Cortex 下发新目标时把它归零
    （契约 §W6.4），Executor 每跑一轮 +1。所以它是「这个目标还没跑过」的判据 ——
    少了它，已经做完的目标会被**再跑一遍**（子目标被重复执行）。
    """
    report = state.executor_report

    # 汇报说做成了 → 走验收。两个条件同时满足时**优先 review**：
    # 先验收完成，再继续下一个子目标（契约 §W8.3）。
    if report is not None and report.outcome == "completed" and not ledger.is_seen(report):
        logger.info("[graph] Subgoal reported as completed; going to orchestrator for review")
        ledger.mark_seen(report)
        return "review"

    # 有目标、而且这个目标还没被执行过 → 去执行。
    if state.executor_goal and state.executor_steps == 0:
        logger.info(f"[graph] Executing subgoal: {state.executor_goal}")
        return "execute"

    # 没有待执行的目标 → 交给 Orchestrator（看门狗）。
    logger.info("[graph] No pending subgoal to execute; going to orchestrator")
    return "review"


def post_executor_gate(
    state: State,
) -> Literal["invoke_tools", "report", "retry"]:
    """Executor 之后的路由。

    * ``invoke_tools`` → ``executor_tools``（执行动作）
    * ``retry``       → ``executor``（**动作被拒了，回 executor 重新决策**）
    * ``report``      → ``reporter``（收尾汇报）

    「有没有动作」从局部历史**最后一条 AIMessage 的 tool_calls** 判断 ——
    Executor 只写 AIMessage（带不带 tool_calls），工具写的是 ToolMessage。

    ⚠️ **``retry`` 这条边是方案 §3.1.1「拒绝理由回灌」能生效的前提**
    （独立验收 2026-10-21 查出的致命 bug 2）。

    拒绝时 Executor 写的是「AIMessage(无 tool_calls) + HumanMessage(拒绝理由)」。
    如果这里只看「最后一条 AIMessage 有没有 tool_calls」，会因为那条 AIMessage
    确实没有 tool_calls 而 `break` → 直接进 reporter ——
    **拒绝理由模型永远看不到，Executor 的自主循环在第一次出岔子时就终止了**。

    ⇒ 判据改成「**最后一条消息是不是 HumanMessage**」：
    是 → 说明上一轮动作被拒了、理由已回灌 → 回 executor 重新决策。

    为什么不用 ToolMessage 承载拒绝理由：它没有对应的 assistant tool_call
    （拒绝时根本没派发出去），OpenAI 兼容端点会 400（致命 bug 3）。
    """
    if state.executor_done:
        # 目标达成 / 步数上限：不再执行动作，直接收尾汇报。
        logger.info("[graph] Executor is done; going to reporter")
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

    # 兜底：没有动作可执行 → 也去 reporter 收尾（不能让流程卡在这儿）。
    logger.info("[graph] No tool calls to execute; going to reporter")
    return "report"


def make_convergence_node(ledger: ReportLedger):
    """造一个带账本的 convergence 节点。

    convergence 是计划层的汇合点，顺便干两件事：

    1. **作废已判过的汇报** —— 被 Cortex 判过的那份从 state 里清掉。
       不清的话下一次进 Cortex 会看到同一份旧汇报，又会「推进到下一个子目标」，
       中间的子目标就被跳过了。
    2. **验收提名** —— 汇报说「做成了」时，把当前子目标放进
       ``complete_subgoals_by_ids``：Orchestrator 只审阅这个名单里的子目标
       （``orchestrator.py`` 里名单为空它连 LLM 都不调），名单空着就没人验收，
       子目标永远标不了 SUCCESS，任务就永远收不了尾。
    """

    async def convergence_node(state: State):
        update: dict = {}
        report = state.executor_report

        if report is not None and ledger.is_seen(report):
            update["executor_report"] = None
        elif (
            report is not None
            and report.outcome == "completed"
            and not state.complete_subgoals_by_ids
        ):
            current_subgoal = get_current_subgoal(state.subgoal_plan)
            if current_subgoal is not None:
                update["complete_subgoals_by_ids"] = [current_subgoal.id]

        return update

    return convergence_node


async def get_graph(ctx: MobileUseContext) -> CompiledStateGraph:
    graph_builder = StateGraph(State)

    # 一次建图 = 一次任务运行：汇报账本跟着这次任务的生命周期。
    ledger = ReportLedger()

    ## Define nodes
    graph_builder.add_node("planner", PlannerNode(ctx))
    graph_builder.add_node("orchestrator", OrchestratorNode(ctx))

    graph_builder.add_node("contextor", ContextorNode(ctx))

    graph_builder.add_node("cortex", CortexNode(ctx))

    graph_builder.add_node("executor", ExecutorNode(ctx))

    executor_wrappers = list(EXECUTOR_WRAPPERS_TOOLS)
    if ctx.video_recording_enabled:
        executor_wrappers.extend(VIDEO_RECORDING_WRAPPERS)

    executor_tool_node = ExecutorToolNode(
        tools=get_tools_from_wrappers(ctx=ctx, wrappers=executor_wrappers),
        messages_key=EXECUTOR_MESSAGES_KEY,
        trace_id=ctx.trace_id,
    )
    graph_builder.add_node("executor_tools", executor_tool_node)

    graph_builder.add_node("reporter", ReporterNode(ctx))

    graph_builder.add_node("summarizer", SummarizerNode(ctx))

    graph_builder.add_node(node="convergence", action=make_convergence_node(ledger), defer=True)

    ## Linking nodes
    graph_builder.add_edge(START, "planner")
    graph_builder.add_edge("planner", "orchestrator")
    graph_builder.add_edge("orchestrator", "convergence")

    # Cortex 下发目标 → 抓屏；子目标完成 → 验收。
    graph_builder.add_conditional_edges(
        "cortex",
        lambda state: post_cortex_gate(state, ledger),
        {
            "execute": "contextor",
            "review": "orchestrator",
        },
    )

    # 抓屏 → 决策一步。
    graph_builder.add_edge("contextor", "executor")

    # 有动作 → 执行；做完了 → 汇报；被拒了 → 回 executor 重新决策。
    graph_builder.add_conditional_edges(
        "executor",
        post_executor_gate,
        {
            "invoke_tools": "executor_tools",
            "report": "reporter",
            # 动作被拒 → 回 executor（屏幕没变，不用重新抓屏）
            "retry": "executor",
        },
    )

    # 工具执行完 → 裁剪 messages → **回 contextor**（重新抓屏、重建元素表）。
    # 不能直接回 executor：那样工具解析 Target.index 时拿到的是别的屏的编号。
    graph_builder.add_edge("executor_tools", "summarizer")
    graph_builder.add_edge("summarizer", "contextor")

    # 汇报 → Orchestrator（看门狗，**无条件**）。
    # 没有这条边，计划级失败就没有处理者（R16）：Cortex 一直重发到撞 recursion_limit。
    graph_builder.add_edge("reporter", "orchestrator")

    graph_builder.add_conditional_edges(
        source="convergence",
        path=convergence_gate,
        path_map={
            "continue": "cortex",
            "replan": "planner",
            "end": END,
        },
    )

    return graph_builder.compile()
