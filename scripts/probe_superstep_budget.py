"""探针 #4：LangGraph 的 super-step 到底怎么算 + 改造后一个动作步消耗几个 super-step。

问题（M12 / 方案 §10 R21）
--------------------------
`sdk/agent.py` 里 `recursion_limit = task.request.max_steps`。但 LangGraph 的
recursion_limit 数的是**图节点执行次数（super-step）**，既不是 LLM 调用次数，
也不是「动作步」。改造后图里多了 reporter / orchestrator / convergence，
同一个 `max_steps` 能跑的动作数就变了 —— 必须实测，不能估。

本探针**不调 LLM、不碰设备**：只复刻图的**拓扑**（同样的 `State`、同样的边、
同样的条件边、同样的 `defer=True`），节点体用脚本化的假实现。
super-step 的数量只由**拓扑**决定（每个 super-step 调度哪些节点），
与节点体里跑 LLM 还是跑 adb 无关 —— 所以这个复刻对「super-step 计数」是等价的。

实测四件事
----------
1. 一次「动作步」（executor 决定 → 执行工具 → 回 contextor）消耗几个 super-step
2. 一个 `recursion_limit` 实际允许多少个 super-step
   （LangGraph 内部有 +1 偏移：`_loop.py:1859` `stop = step + recursion_limit + 1`）
3. 改造前 / 改造后，同一个 `max_steps` 能跑多少个动作步
4. 结论：换算系数该取多少、余量该留多少

跑法：
    uv run python v/probe_superstep_budget.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "minitap"))

from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.constants import END, START  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402
from langgraph.graph import StateGraph  # noqa: E402
from langgraph.graph.message import REMOVE_ALL_MESSAGES  # noqa: E402

from minitap.mobile_use.agents.executor.types import ExecutorReport  # noqa: E402
from minitap.mobile_use.agents.planner.types import Subgoal, SubgoalStatus  # noqa: E402
from minitap.mobile_use.agents.planner.utils import (  # noqa: E402
    all_completed,
    complete_subgoals_by_ids,
    get_current_subgoal,
    get_subgoals_by_ids,
    nothing_started,
    start_next_subgoal,
)
from minitap.mobile_use.graph.graph import (  # noqa: E402
    ReportLedger,
    convergence_gate,
    make_convergence_node,
    post_cortex_gate,
    post_executor_gate,
)
from minitap.mobile_use.graph.state import State  # noqa: E402

# ---------------------------------------------------------------------------
# 复刻用的脚本参数
# ---------------------------------------------------------------------------

N_SUBGOALS = 3  # 计划里有几个子目标
STEPS_PER_SUBGOAL = 4  # 每个子目标跑几个动作步（= 几个工具执行）
GOAL = "读出 Wi-Fi 名称"


def _plan(n: int) -> list[Subgoal]:
    return [
        Subgoal(id=f"sg{i}", description=f"subgoal {i}", status=SubgoalStatus.NOT_STARTED)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 新拓扑的复刻（与 minitap/mobile_use/graph/graph.py 逐条对应）
# ---------------------------------------------------------------------------


def build_new_graph():
    """复刻**改造后**的拓扑。

    节点体是脚本化的假实现（不调 LLM、不碰设备），但：
      - State 是真的 `State`
      - 条件边函数是真的 `post_cortex_gate` / `post_executor_gate` / `convergence_gate`
      - `convergence` 的 `defer=True` 是真的
      - 边和路径映射与 `graph.py` 逐条一致
    """
    g = StateGraph(State)
    ledger = ReportLedger()

    async def planner(state: State):
        return {"subgoal_plan": _plan(N_SUBGOALS)}

    async def orchestrator(state: State):
        # 照抄 `agents/orchestrator/orchestrator.py` 的分支结构，只是把
        # 「调 LLM 验收」换成「按 complete_subgoals_by_ids 验收」。
        plan = state.subgoal_plan
        if nothing_started(plan) or get_current_subgoal(plan) is None:
            plan = start_next_subgoal(plan)
            return {"subgoal_plan": plan, "complete_subgoals_by_ids": []}
        to_examine = get_subgoals_by_ids(plan, state.complete_subgoals_by_ids)
        if not to_examine:
            return {"complete_subgoals_by_ids": []}
        plan = complete_subgoals_by_ids(plan, state.complete_subgoals_by_ids)
        if not all_completed(plan):
            plan = start_next_subgoal(plan)
        return {"subgoal_plan": plan, "complete_subgoals_by_ids": []}

    async def contextor(state: State):
        return {}

    async def cortex(state: State):
        # 照抄 `agents/cortex/cortex.py:412-455` 的语义：
        # 汇报判完成 → 提名给 orchestrator；否则下发目标 + 归零局部历史。
        report = state.executor_report
        current = get_current_subgoal(state.subgoal_plan)
        update: dict = {"latest_ui_hierarchy": None, "executor_report": None}
        if report is not None and report.outcome == "completed":
            update["complete_subgoals_by_ids"] = [current.id] if current else []
        else:
            update["executor_goal"] = GOAL
            update["executor_messages"] = [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
            update["executor_steps"] = 0
            update["complete_subgoals_by_ids"] = []
        return update

    async def executor(state: State):
        step_index = state.executor_steps + 1
        if state.executor_steps >= STEPS_PER_SUBGOAL:
            # 目标达成：写一条没有 tool_calls 的 AIMessage → graph 走 reporter
            return {
                "executor_done": True,
                "executor_steps": step_index,
                "executor_messages": [AIMessage(content="goal reached")],
            }
        # 一个动作步
        return {
            "executor_done": False,
            "executor_steps": step_index,
            "executor_messages": [
                AIMessage(
                    content="tap it",
                    tool_calls=[
                        {
                            "name": "tap",
                            "args": {"target": {"index": 1}},
                            "id": f"c{step_index}",
                        }
                    ],
                )
            ],
        }

    async def executor_tools(state: State):
        return {"executor_messages": [ToolMessage(content="ok", tool_call_id="c1")]}

    async def summarizer(state: State):
        return {}

    async def reporter(state: State):
        return {
            "executor_report": ExecutorReport(
                outcome="completed", summary="done", evidence="ok"
            )
        }

    g.add_node("planner", planner)
    g.add_node("orchestrator", orchestrator)
    g.add_node("contextor", contextor)
    g.add_node("cortex", cortex)
    g.add_node("executor", executor)
    g.add_node("executor_tools", executor_tools)
    g.add_node("summarizer", summarizer)
    g.add_node("reporter", reporter)
    g.add_node("convergence", action=make_convergence_node(ledger), defer=True)

    g.add_edge(START, "planner")
    g.add_edge("planner", "orchestrator")
    g.add_edge("orchestrator", "convergence")
    g.add_conditional_edges(
        "cortex",
        lambda state: post_cortex_gate(state, ledger),
        {"execute": "contextor", "review": "orchestrator"},
    )
    g.add_edge("contextor", "executor")
    g.add_conditional_edges(
        "executor",
        post_executor_gate,
        {"invoke_tools": "executor_tools", "report": "reporter", "retry": "executor"},
    )
    g.add_edge("executor_tools", "summarizer")
    g.add_edge("summarizer", "contextor")
    g.add_edge("reporter", "orchestrator")
    g.add_conditional_edges(
        source="convergence",
        path=convergence_gate,
        path_map={"continue": "cortex", "replan": "planner", "end": END},
    )
    return g.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# 改造前的拓扑（git show HEAD:minitap/mobile_use/graph/graph.py）
# ---------------------------------------------------------------------------


def build_old_graph():
    """复刻**改造前**的拓扑（对照用）。

    老图的 executor 是哑执行器：Cortex 吐一份 decisions，Executor 翻译成
    tool_calls。老图的 `executor_steps` 语义是「本子目标已执行步数」，
    改造后换了语义 —— 这里只关心 super-step 计数。
    """
    g = StateGraph(State)
    ledger = ReportLedger()

    async def planner(state: State):
        return {"subgoal_plan": _plan(N_SUBGOALS)}

    async def orchestrator(state: State):
        plan = state.subgoal_plan
        if nothing_started(plan) or get_current_subgoal(plan) is None:
            plan = start_next_subgoal(plan)
            return {"subgoal_plan": plan, "complete_subgoals_by_ids": []}
        to_examine = get_subgoals_by_ids(plan, state.complete_subgoals_by_ids)
        if not to_examine:
            return {"complete_subgoals_by_ids": []}
        plan = complete_subgoals_by_ids(plan, state.complete_subgoals_by_ids)
        if not all_completed(plan):
            plan = start_next_subgoal(plan)
        return {"subgoal_plan": plan, "complete_subgoals_by_ids": []}

    async def contextor(state: State):
        return {}

    async def cortex(state: State):
        # 老 cortex：每轮都发一份 decisions 下去（没有 reporter / 自主子目标）
        return {"structured_decisions": json.dumps([{"action": "tap"}])}

    async def executor(state: State):
        return {
            "executor_messages": [
                AIMessage(
                    content="tap it",
                    tool_calls=[{"name": "tap", "args": {}, "id": "c1"}],
                )
            ]
        }

    async def executor_tools(state: State):
        return {"executor_messages": [ToolMessage(content="ok", tool_call_id="c1")]}

    async def summarizer(state: State):
        return {}

    g.add_node("planner", planner)
    g.add_node("orchestrator", orchestrator)
    g.add_node("contextor", contextor)
    g.add_node("cortex", cortex)
    g.add_node("executor", executor)
    g.add_node("executor_tools", executor_tools)
    g.add_node("summarizer", summarizer)
    g.add_node("convergence", action=make_convergence_node(ledger), defer=True)

    g.add_edge(START, "planner")
    g.add_edge("planner", "orchestrator")
    g.add_edge("orchestrator", "convergence")
    g.add_edge("contextor", "cortex")
    g.add_conditional_edges(
        "cortex",
        lambda state: (
            ["review_subgoals"]
            if state.complete_subgoals_by_ids or not state.structured_decisions
            else []
        )
        + (["execute_decisions"] if state.structured_decisions else []),
        {"review_subgoals": "orchestrator", "execute_decisions": "executor"},
    )
    g.add_conditional_edges(
        "executor",
        lambda state: (
            "invoke_tools"
            if state.executor_messages
            and isinstance(state.executor_messages[-1], AIMessage)
            and getattr(state.executor_messages[-1], "tool_calls", None)
            else "skip"
        ),
        {"invoke_tools": "executor_tools", "skip": "summarizer"},
    )
    g.add_edge("executor_tools", "summarizer")
    g.add_edge("summarizer", "convergence")
    g.add_conditional_edges(
        source="convergence",
        path=convergence_gate,
        path_map={"continue": "contextor", "replan": "planner", "end": END},
    )
    return g.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# 计数
# ---------------------------------------------------------------------------


async def run_and_count(graph, *, recursion_limit: int) -> dict:
    """跑一遍图，数 super-step / 节点执行次数 / 动作步数。"""
    # 初始 state 照抄 `sdk/agent.py:1213` 的 `_get_graph_state`
    init = {
        "messages": [],
        "initial_goal": "goal",
        "subgoal_plan": [],
        "latest_ui_hierarchy": None,
        "latest_ocr_results": None,
        "latest_screen_elements": None,
        "latest_screenshot": None,
        "focused_app_info": None,
        "device_date": None,
        "structured_decisions": None,
        "complete_subgoals_by_ids": [],
        "agents_thoughts": [],
        "remaining_steps": 400,
        "executor_messages": [],
        "cortex_last_thought": None,
        "scratchpad": {},
    }
    super_steps = 0
    node_runs: dict[str, int] = {}
    action_steps = 0
    executor_runs = 0
    try:
        async for mode, payload in graph.astream(
            input=init,
            config={
                "recursion_limit": recursion_limit,
                "configurable": {"thread_id": f"probe-{recursion_limit}"},
            },
            stream_mode=["checkpoints", "updates"],
        ):
            if mode == "checkpoints":
                # 每个 super-step 落一次 checkpoint；metadata["step"] 是权威步号
                meta = payload.get("metadata") or {}
                super_steps = max(super_steps, int(meta.get("step", 0)))
            elif mode == "updates":
                for node in payload:
                    node_runs[node] = node_runs.get(node, 0) + 1
                    if node == "executor_tools":
                        action_steps += 1
                    if node == "executor":
                        executor_runs += 1
    except GraphRecursionError as e:
        return {
            "recursion_limit": recursion_limit,
            "super_steps": super_steps,
            "node_runs": node_runs,
            "action_steps": action_steps,
            "executor_runs": executor_runs,
            "recursion_error": True,
            "error": str(e)[:120],
        }
    return {
        "recursion_limit": recursion_limit,
        "super_steps": super_steps,
        "node_runs": node_runs,
        "action_steps": action_steps,
        "executor_runs": executor_runs,
        "recursion_error": False,
    }


async def main() -> None:
    global STEPS_PER_SUBGOAL

    print("=" * 78)
    print("0) 计数器的自检：super_steps 是否等于节点执行次数")
    print("=" * 78)
    STEPS_PER_SUBGOAL = 4
    r = await run_and_count(build_new_graph(), recursion_limit=1000)
    total_nodes = sum(r["node_runs"].values())
    print(f"  super_steps(checkpoint step) = {r['super_steps']}")
    print(f"  节点执行次数合计            = {total_nodes}")
    print(f"  相等？ {r['super_steps'] == total_nodes}")
    print(f"  node_runs = {r['node_runs']}")

    print()
    print("=" * 78)
    print("1) 新拓扑：每个「动作步」消耗几个 super-step")
    print("   （动作步 = executor 节点跑一轮；工具执行 = executor_tools 跑一轮）")
    print("=" * 78)
    print("  K = 每个子目标跑几个动作步；S = 计划里有几个子目标")
    for s in (1, 3):
        global N_SUBGOALS
        N_SUBGOALS = s
        for k in (1, 2, 4, 8):
            STEPS_PER_SUBGOAL = k
            r = await run_and_count(build_new_graph(), recursion_limit=1000)
            n_ex = r["executor_runs"]
            n_tools = r["action_steps"]
            print(
                f"  S={s} K={k:2d} | super_steps={r['super_steps']:4d} "
                f"executor步={n_ex:3d} 工具执行={n_tools:3d} | "
                f"super_steps/executor步 = {r['super_steps'] / n_ex:5.2f}"
            )
            print(f"        {r['node_runs']}")
    N_SUBGOALS = 3

    print()
    print("=" * 78)
    print("2) recursion_limit 实际允许多少个 super-step（LangGraph 内部有 +1 偏移）")
    print("=" * 78)
    STEPS_PER_SUBGOAL = 4
    for rl in (3, 5, 10, 20, 50, 100):
        r = await run_and_count(build_new_graph(), recursion_limit=rl)
        print(
            f"  recursion_limit={rl:4d} -> super_steps executed={r['super_steps']:4d}  "
            f"executor步={r['executor_runs']:3d}  recursion_error={r['recursion_error']}"
        )

    print()
    print("=" * 78)
    print("3) 改造前：每个动作步消耗几个 super-step")
    print("=" * 78)
    for rl in (30, 100, 400):
        old = await run_and_count(build_old_graph(), recursion_limit=rl)
        n_tools = max(old["action_steps"], 1)
        print(
            f"  recursion_limit={rl:4d} -> super_steps={old['super_steps']:4d} "
            f"工具执行={old['action_steps']:4d} | "
            f"super_steps/动作步 = {old['super_steps'] / n_tools:.2f}"
        )

    print()
    print("=" * 78)
    print("4) 改造后（未换算前）：同一个 max_steps 只能跑几个动作步")
    print("=" * 78)
    STEPS_PER_SUBGOAL = 4
    for rl in (30, 100, 400):
        new = await run_and_count(build_new_graph(), recursion_limit=rl)
        print(
            f"  recursion_limit={rl:4d} -> super_steps={new['super_steps']:4d} "
            f"executor步={new['executor_runs']:3d} 工具执行={new['action_steps']:3d} "
            f"-> 每个动作步 {new['super_steps'] / max(new['executor_runs'], 1):.2f} super-step"
        )


if __name__ == "__main__":
    asyncio.run(main())
