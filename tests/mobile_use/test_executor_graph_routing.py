"""Executor 自主执行改造的**路由层**回归测试。

这一轮补的是独立验收（2026-10-21）查出的盲区：``grep`` 全 ``tests/`` 目录，
``get_graph`` / ``post_executor_gate`` / ``plan_manager_gate`` / ``post_cortex_gate``
**零引用** —— 4 个路由级 bug 全部住在没有一行测试的代码里，其中 3 个是致命级：

* ① ``done: true`` + ``actions: []`` 的完成路径不可达（被 dispatcher 当成空动作拒了）；
* ② 动作被 dispatcher 拒后，**拒绝理由模型永远看不到**（直接进了收尾）；
* ③ 拒绝回灌用了 ToolMessage，协议非法 → OpenAI 兼容端点 400。

①②③ 已经修好了。本文件的作用是**把路由行为锁进测试**，让这类 bug 以后改不动。

锁两类东西：

* **路由函数的判定**（R1~R12）—— 纯函数，不接设备、不调 LLM，毫秒级。
* **整图的结构** —— 图能编译、关键边存在、节点齐全。

不测「整图跑通」：那要真机 + LLM，交给真机 A/B。
"""

from __future__ import annotations

import asyncio
import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "minitap"))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from minitap.mobile_use.agents.plan_manager.plan_manager import (  # noqa: E402
    MAX_ISSUES_PER_SUBGOAL,
    PlanManagerNode,
    plan_manager_gate,
    post_cortex_gate,
)
from minitap.mobile_use.agents.planner.types import Subgoal, SubgoalStatus  # noqa: E402
from minitap.mobile_use.config import LLM, LLMConfig, LLMConfigUtils, LLMWithFallback  # noqa: E402
from minitap.mobile_use.context import (  # noqa: E402
    DeviceContext,
    DevicePlatform,
    MobileUseContext,
)
from minitap.mobile_use.graph.graph import get_graph, post_executor_gate  # noqa: E402
from minitap.mobile_use.graph.state import State  # noqa: E402

# ---------------------------------------------------------------------------
# 假 ctx / 假 state
# ---------------------------------------------------------------------------


def _ctx() -> MobileUseContext:
    """造一个**不接设备**的 ctx。

    照抄 `tests/mobile_use/test_executor_autonomy.py` 里的同名辅助函数 ——
    那个文件已经验证过它足够让节点构造起来。

    `get_graph` 在建图时真正读到的 ctx 字段只有三个：
    `ctx.video_recording_enabled`（决定要不要挂录像工具）、
    `ctx.trace_id`（工具节点要用）、以及传给各节点的 ctx 本身
    （各节点的 `__init__` 只存引用，不碰设备）。
    `ctx.execution_setup` **建图时读不到** —— 它只在节点**运行时**被读
    （`state.asanitize_update` → `record_interaction`），所以这里不造。
    """

    def llm() -> LLMWithFallback:
        return LLMWithFallback(
            provider="openai", model="gpt-5", fallback=LLM(provider="openai", model="o4-mini")
        )

    return MobileUseContext(
        trace_id="test",
        device=DeviceContext(
            host_platform="WINDOWS",
            mobile_platform=DevicePlatform.ANDROID,
            device_id="test",
            device_width=1220,
            device_height=2712,
        ),
        llm_config=LLMConfig(
            planner=llm(),
            orchestrator=llm(),
            contextor=llm(),
            cortex=llm(),
            executor=llm(),
            utils=LLMConfigUtils(outputter=llm(), hopper=llm()),
        ),
    )


def _state(
    *,
    executor_messages: list | None = None,
    executor_done: bool = False,
    executor_goal: str | None = None,
    executor_steps: int = 0,
    subgoal_plan: list[Subgoal] | None = None,
    complete_subgoals_by_ids: list[str] | None = None,
    agents_thoughts: list[str] | None = None,
) -> State:
    """造一个**最小**的 State。

    只填 State 里**必填**的字段（pydantic 会拒绝缺字段），其余走默认值 ——
    路由函数只读 `executor_*` 和 `subgoal_plan` 这几项。
    """
    return State(
        initial_goal="打开 WLAN 页面",
        subgoal_plan=subgoal_plan if subgoal_plan is not None else [],
        latest_ui_hierarchy=None,
        latest_ocr_results=None,
        focused_app_info=None,
        device_date=None,
        complete_subgoals_by_ids=(
            complete_subgoals_by_ids if complete_subgoals_by_ids is not None else []
        ),
        executor_messages=executor_messages if executor_messages is not None else [],
        agents_thoughts=agents_thoughts if agents_thoughts is not None else [],
        # 路由函数真正关心的三个字段
        executor_done=executor_done,
        executor_goal=executor_goal,
        executor_steps=executor_steps,
    )


def _subgoal(sid: str, status: SubgoalStatus) -> Subgoal:
    return Subgoal(id=sid, description=f"子目标 {sid}", status=status)


# ---------------------------------------------------------------------------
# R1~R4 —— post_executor_gate（Executor 之后的路由）
# ---------------------------------------------------------------------------


def test_r1_ai_message_with_tool_calls_goes_to_invoke_tools():
    """R1：Executor 刚写了一条带 tool_calls 的 AIMessage → 去执行动作。

    这是主循环的正常一步。判据是「最后一条 AIMessage 带不带 tool_calls」。
    """
    state = _state(
        executor_messages=[
            AIMessage(content="", tool_calls=[{"name": "tap", "args": {"index": 7}, "id": "c1"}])
        ]
    )
    assert post_executor_gate(state) == "invoke_tools"


def test_r2_executor_done_goes_to_report():
    """R2：``executor_done=True`` → 去 cortex 收尾（它在入口构建汇报）。

    致命 bug ① 的路由层防线：「完成」这条路必须**可达** ——
    模型说 ``done: true`` 时 actions 是空数组，任何「有没有动作」的判据
    都会把它判成「没动作可做」而绕开收尾。``executor_done`` 是**独立信号**，
    不依赖消息形态。
    """
    state = _state(
        executor_done=True,
        executor_messages=[AIMessage(content="goal reached")],
    )
    assert post_executor_gate(state) == "report"


def test_r2b_executor_done_wins_over_everything():
    """R2b：``executor_done=True`` 时，无论最后一条消息长什么样都去 report。

    为什么锁这条：完成信号和「消息形态」是**两套独立事实**。
    只要判据里掺进消息形态（哪怕是「有 tool_calls 就先执行」），
    完成路径就可能再次变得不可达 —— 这正是致命 bug ① 的形状。
    """
    # 最后一条是 HumanMessage（拒绝理由）时，done 仍然优先
    state = _state(
        executor_done=True,
        executor_messages=[
            AIMessage(content="goal reached"),
            HumanMessage(content="Your previous action was NOT executed: ..."),
        ],
    )
    assert post_executor_gate(state) == "report"

    # 最后一条是带 tool_calls 的 AIMessage 时，done 仍然优先
    state = _state(
        executor_done=True,
        executor_messages=[
            AIMessage(content="", tool_calls=[{"name": "tap", "args": {"index": 7}, "id": "c1"}])
        ],
    )
    assert post_executor_gate(state) == "report"


def test_r3_rejection_reason_goes_back_to_executor():
    """R3：最后一条消息是 **HumanMessage** → 回 executor 重新决策。

    致命 bug ② 的防线：dispatcher 拒掉动作后，Executor 写的是
    「AIMessage(无 tool_calls) + HumanMessage(拒绝理由)」。
    如果路由只看「最后一条 AIMessage 有没有 tool_calls」，那条 AIMessage
    确实没有 → 直接进收尾 → **拒绝理由模型永远看不到**，
    Executor 的自主循环在第一次出岔子时就终止了。

    ⇒ 判据必须是「最后一条消息是不是 HumanMessage」。
    """
    state = _state(
        executor_messages=[
            AIMessage(content="我要点那个按钮"),
            HumanMessage(
                content=(
                    "Your previous action was NOT executed: 'teleport' is not a known "
                    "tool; available tools are: tap, back. Decide again."
                )
            ),
        ]
    )
    assert post_executor_gate(state) == "retry"


def test_r4_no_messages_falls_back_to_report():
    """R4：一条消息都没有 → 去 cortex 收尾（兜底，不能卡住）。

    路由函数不能返回第四种结果，也不能抛异常 —— 卡在 executor 节点
    意味着整个任务挂到 ``recursion_limit`` 才退出。
    """
    assert post_executor_gate(_state()) == "report"


def test_r4b_ai_message_without_tool_calls_falls_back_to_report():
    """R4b：最后一条 AIMessage **不带** tool_calls → 也去 cortex 收尾。

    这是「模型只想不动手」的形态。必须收尾，不能当成「有动作」执行。
    """
    state = _state(executor_messages=[AIMessage(content="我先看看这个页面")])
    assert post_executor_gate(state) == "report"


# ---------------------------------------------------------------------------
# R7~R8 —— post_cortex_gate（Cortex 之后的路由）
# ---------------------------------------------------------------------------


def test_r7_pending_goal_with_zero_steps_goes_to_execute():
    """R7：有 ``executor_goal`` 且 ``executor_steps == 0`` → 去执行。

    ``executor_steps == 0`` 是「这个目标还没跑过」的信号：
    Cortex 下发新目标时把它归零，Executor 每跑一轮 +1。
    少了它，已经做完的目标会被**再跑一遍**（子目标被重复执行）。
    """
    state = _state(executor_goal="读出 Wi-Fi 名称", executor_steps=0)
    assert post_cortex_gate(state) == "execute"


def test_r7b_already_run_goal_is_not_executed_again():
    """R7b：目标还在，但 ``executor_steps > 0`` → **不是** execute。

    这个目标已经跑过了。再执行一遍 = 同一个子目标做两遍。
    没有待办目标 → 交给 plan_manager 当看门狗。
    """
    state = _state(executor_goal="读出 Wi-Fi 名称", executor_steps=3)
    assert post_cortex_gate(state) != "execute"
    assert post_cortex_gate(state) == "review"


def test_r8_no_report_and_no_goal_goes_to_review():
    """R8：既没有新汇报、也没有待跑目标 → 去 plan_manager（看门狗兜底）。

    这是 R16 的落点：没有这条兜底，计划级失败（子目标根本做不到）
    就没有任何处理者 —— Cortex 会一直说「没完成」一直重发，
    最后撞 ``recursion_limit`` **异常退出**（不是「重新拆计划」）。
    """
    state = _state()
    assert post_cortex_gate(state) == "review"


def test_r7c_reissue_budget_exceeded_goes_to_budget_exceeded():
    """R7c：同一个子目标下发次数超过预算 → 不再执行，交给 PlanManager 熔断。

    预算是 ``MAX_ISSUES_PER_SUBGOAL`` 次尝试：第 ``MAX+1`` 次下发前拦下。
    数的是 ``[cortex] Issued goal for subgoal <id>:`` 这条 append-only thought。
    """
    subgoal_plan = [_subgoal("sg1", SubgoalStatus.PENDING)]
    marker = "[cortex] Issued goal for subgoal sg1:"

    at_budget = _state(
        subgoal_plan=subgoal_plan,
        executor_goal="读出 Wi-Fi 名称",
        executor_steps=0,
        agents_thoughts=[f"{marker} attempt {i}" for i in range(MAX_ISSUES_PER_SUBGOAL)],
    )
    assert post_cortex_gate(at_budget) == "execute"

    over_budget = _state(
        subgoal_plan=subgoal_plan,
        executor_goal="读出 Wi-Fi 名称",
        executor_steps=0,
        agents_thoughts=[
            f"{marker} attempt {i}" for i in range(MAX_ISSUES_PER_SUBGOAL + 1)
        ],
    )
    assert post_cortex_gate(over_budget) == "budget_exceeded"


# ---------------------------------------------------------------------------
# R9~R11 —— plan_manager_gate（计划层的汇合点）
# ---------------------------------------------------------------------------


def test_r9_failure_subgoal_triggers_replan():
    """R9：有子目标 FAILURE → 重新拆计划。

    计划里已经有做不成的子目标，继续往下跑只会重复失败。
    """
    state = _state(
        subgoal_plan=[
            _subgoal("a", SubgoalStatus.SUCCESS),
            _subgoal("b", SubgoalStatus.FAILURE),
            _subgoal("c", SubgoalStatus.NOT_STARTED),
        ]
    )
    assert plan_manager_gate(state) == "replan"


def test_r10_all_success_ends_the_run():
    """R10：全部 SUCCESS → 结束。

    ``all_completed`` 对**空计划**也返回 True —— 计划空着就是没活可干，
    结束是对的。
    """
    state = _state(
        subgoal_plan=[
            _subgoal("a", SubgoalStatus.SUCCESS),
            _subgoal("b", SubgoalStatus.SUCCESS),
        ]
    )
    assert plan_manager_gate(state) == "end"


def test_r11_in_progress_continues():
    """R11：正常进行中（有 PENDING 的当前子目标）→ 继续走 cortex。

    注意「有当前子目标」是判据：计划里全是 NOT_STARTED 时没有当前子目标
    → 结束（不该在没有目标的情况下空转）。
    """
    state = _state(
        subgoal_plan=[
            _subgoal("a", SubgoalStatus.SUCCESS),
            _subgoal("b", SubgoalStatus.PENDING),
            _subgoal("c", SubgoalStatus.NOT_STARTED),
        ]
    )
    assert plan_manager_gate(state) == "continue"


# ---------------------------------------------------------------------------
# PlanManager 节点 —— 计划状态推进（纯代码，不调 LLM）
# ---------------------------------------------------------------------------


def _run_plan_manager(state: State) -> dict:
    return asyncio.run(PlanManagerNode(_ctx())(state))


def test_plan_manager_starts_the_first_subgoal():
    state = _state(subgoal_plan=[_subgoal("sg1", SubgoalStatus.NOT_STARTED)])
    update = _run_plan_manager(state)
    plan = update["subgoal_plan"]
    assert plan[0].status == SubgoalStatus.PENDING
    assert any("Starting the first subgoal" in t for t in update["agents_thoughts"])


def test_plan_manager_accepts_completion_and_starts_next():
    state = _state(
        subgoal_plan=[
            _subgoal("sg1", SubgoalStatus.PENDING),
            _subgoal("sg2", SubgoalStatus.NOT_STARTED),
        ],
        complete_subgoals_by_ids=["sg1"],
    )
    update = _run_plan_manager(state)
    plan = update["subgoal_plan"]
    assert plan[0].status == SubgoalStatus.SUCCESS
    assert plan[1].status == SubgoalStatus.PENDING
    assert update["complete_subgoals_by_ids"] == []
    assert any("Accepted completion" in t for t in update["agents_thoughts"])


def test_plan_manager_ends_after_all_subgoals_complete():
    state = _state(
        subgoal_plan=[_subgoal("sg1", SubgoalStatus.PENDING)],
        complete_subgoals_by_ids=["sg1"],
    )
    update = _run_plan_manager(state)
    assert update["subgoal_plan"][0].status == SubgoalStatus.SUCCESS
    assert any("All subgoals completed" in t for t in update["agents_thoughts"])


def test_plan_manager_breaks_the_reissue_loop():
    marker = "[cortex] Issued goal for subgoal sg1:"
    state = _state(
        subgoal_plan=[_subgoal("sg1", SubgoalStatus.PENDING)],
        executor_goal="读出 Wi-Fi 名称",
        executor_steps=0,
        agents_thoughts=[
            f"{marker} attempt {i}" for i in range(MAX_ISSUES_PER_SUBGOAL + 1)
        ],
    )
    update = _run_plan_manager(state)
    assert update["subgoal_plan"][0].status == SubgoalStatus.FAILURE
    assert any("exceeded the reissue budget" in t for t in update["agents_thoughts"])


# ---------------------------------------------------------------------------
# 整图结构 —— 图能编译、关键边存在、节点齐全
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _compiled_graph():
    """编译一次图，全文件复用（建图要造 16 个工具对象，不便宜）。"""
    return asyncio.run(get_graph(_ctx()))


def _edges() -> set[tuple[str, str]]:
    """编译后的图里的 (source, target) 对。

    条件边在 drawable 里是**展开**的：path_map 里每个分支各占一条边，
    边上还带 `data`（分支名）。所以「图里有没有这条边」= 「这条路径通不通」。
    """
    drawable = _compiled_graph().get_graph()
    return {(e.source, e.target) for e in drawable.edges}


def _branch(source: str, target: str) -> str | None:
    """取一条边上的分支名（条件边的 `data`）。"""
    drawable = _compiled_graph().get_graph()
    for e in drawable.edges:
        if e.source == source and e.target == target:
            return getattr(e, "data", None)
    return None


def test_graph_compiles_without_a_device():
    """整图能编译（不接设备、不调 LLM）。

    编译不过 = 拓扑写错了，整条链都跑不起来。
    """
    graph = _compiled_graph()
    assert graph is not None


def test_graph_has_executor_to_cortex_edge():
    """``executor → cortex``（``report`` 分支）必须存在。

    Reporter 已于 2026-10-29 合并进 cortex：Executor 结束一次尝试之后直接回
    Cortex，由它在入口用纯代码构建汇报（看门狗 R16 由 Cortex 之后的
    PlanManager 承担，见下一个测试）。少这条边，计划级失败就没有处理者。
    """
    assert ("executor", "cortex") in _edges()
    assert _branch("executor", "cortex") == "report"


def test_graph_has_executor_tools_to_contextor_edge():
    """``executor_tools → contextor``：工具执行完**必须回 contextor** 重新抓屏。

    不能直接回 executor：那样工具解析 ``Target.index`` 时拿到的是
    **上一屏**的编号，点错元素（索引契约，方案 §9.1 / R19）。

    Summarizer 已于 2026-10-29 移除：它只裁剪 ``state.messages``，而 v2 里
    没有任何节点写那个字段，节点恒为 no-op。裁剪职责不再存在，边直接连回
    contextor。
    """
    assert ("executor_tools", "contextor") in _edges()


def test_graph_has_contextor_to_executor_edge():
    """``contextor → executor``：抓屏 → 决策（主循环入口）。"""
    assert ("contextor", "executor") in _edges()


def test_graph_has_executor_retry_self_loop():
    """``executor → executor``（retry 边）必须存在。

    致命 bug ② 的修复：动作被 dispatcher 拒了以后，
    **拒绝理由要回灌给模型**（HumanMessage），模型带着理由重新决策。
    少了这条自环，理由就永远送不到模型手里 ——
    路由直接进了收尾，Executor 的自主循环第一次出岔子就终止。
    """
    assert ("executor", "executor") in _edges()
    assert _branch("executor", "executor") == "retry", (
        "executor 自环必须是 'retry' 分支（不能是别的分支名）"
    )


def test_graph_has_all_required_nodes():
    """节点齐全：少一个节点整条链就断在那一段。"""
    drawable = _compiled_graph().get_graph()
    nodes = set(drawable.nodes)
    for required in (
        "planner",
        "plan_manager",
        "contextor",
        "cortex",
        "executor",
        "executor_tools",
    ):
        assert required in nodes, f"图里少了节点 {required}"
    assert "orchestrator" not in nodes, "orchestrator 已由纯代码 plan_manager 替代"
    assert "reporter" not in nodes, "reporter 已合并进 cortex（汇报是纯代码）"
    assert "summarizer" not in nodes, "summarizer 已于 2026-10-29 移除"


def test_graph_routes_plan_manager_through_gate():
    """``plan_manager`` 的条件边就是 plan_manager_gate（continue / replan / end）。"""
    assert ("plan_manager", "cortex") in _edges()
    assert _branch("plan_manager", "cortex") == "continue"
    assert ("plan_manager", "planner") in _edges()
    assert _branch("plan_manager", "planner") == "replan"
    assert ("plan_manager", "__end__") in _edges()
    assert _branch("plan_manager", "__end__") == "end"


def test_graph_routes_cortex_to_execute_or_plan_manager():
    """Cortex 的 ``execute`` 边直达 contextor；其余分支回 plan_manager。"""
    assert ("cortex", "contextor") in _edges()
    assert _branch("cortex", "contextor") == "execute"
    assert ("cortex", "plan_manager") in _edges()
