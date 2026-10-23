"""Executor 自主执行改造的**路由层**回归测试。

这一轮补的是独立验收（2026-10-21）查出的盲区：``grep`` 全 ``tests/`` 目录，
``get_graph`` / ``post_executor_gate`` / ``post_cortex_gate`` / ``convergence_gate``
**零引用** —— 4 个路由级 bug 全部住在没有一行测试的代码里，其中 3 个是致命级：

* ① ``done: true`` + ``actions: []`` 的完成路径不可达（被 dispatcher 当成空动作拒了）；
* ② 动作被 dispatcher 拒后，**拒绝理由模型永远看不到**（直接进了 reporter）；
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

from minitap.mobile_use.agents.executor.types import ExecutorReport  # noqa: E402
from minitap.mobile_use.agents.planner.types import Subgoal, SubgoalStatus  # noqa: E402
from minitap.mobile_use.config import LLM, LLMConfig, LLMConfigUtils, LLMWithFallback  # noqa: E402
from minitap.mobile_use.context import (  # noqa: E402
    DeviceContext,
    DevicePlatform,
    MobileUseContext,
)
from minitap.mobile_use.graph.graph import (  # noqa: E402
    ReportLedger,
    convergence_gate,
    get_graph,
    post_cortex_gate,
    post_executor_gate,
)
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
    executor_report: ExecutorReport | None = None,
    subgoal_plan: list[Subgoal] | None = None,
) -> State:
    """造一个**最小**的 State。

    只填 State 里**必填**的字段（pydantic 会拒绝缺字段），其余走默认值 ——
    路由函数只读 `executor_*` 和 `subgoal_plan` 这几项。
    """
    return State(
        messages=[],
        initial_goal="打开 WLAN 页面",
        subgoal_plan=subgoal_plan if subgoal_plan is not None else [],
        latest_ui_hierarchy=None,
        latest_ocr_results=None,
        focused_app_info=None,
        device_date=None,
        structured_decisions=None,
        complete_subgoals_by_ids=[],
        executor_messages=executor_messages if executor_messages is not None else [],
        cortex_last_thought=None,
        agents_thoughts=[],
        # 路由函数真正关心的四个字段
        executor_done=executor_done,
        executor_goal=executor_goal,
        executor_steps=executor_steps,
        executor_report=executor_report,
    )


def _subgoal(sid: str, status: SubgoalStatus) -> Subgoal:
    return Subgoal(id=sid, description=f"子目标 {sid}", status=status)


def _report(outcome: str = "completed", summary: str = "做完了") -> ExecutorReport:
    return ExecutorReport(outcome=outcome, summary=summary)


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
    """R2：``executor_done=True`` → 去 reporter 收尾。

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
    确实没有 → 直接进 reporter → **拒绝理由模型永远看不到**，
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
    """R4：一条消息都没有 → 去 reporter（兜底，不能卡住）。

    路由函数不能返回第四种结果，也不能抛异常 —— 卡在 executor 节点
    意味着整个任务挂到 ``recursion_limit`` 才退出。
    """
    assert post_executor_gate(_state()) == "report"


def test_r4b_ai_message_without_tool_calls_falls_back_to_report():
    """R4b：最后一条 AIMessage **不带** tool_calls → 也去 reporter。

    这是「模型只想不动手」的形态。必须收尾，不能当成「有动作」执行。
    """
    state = _state(executor_messages=[AIMessage(content="我先看看这个页面")])
    assert post_executor_gate(state) == "report"


# ---------------------------------------------------------------------------
# R5~R8 —— post_cortex_gate（Cortex 之后的路由）
# ---------------------------------------------------------------------------


def test_r5_new_completed_report_goes_to_review():
    """R5：有一份**新的**（账本没见过）completed 汇报 → 去 orchestrator 验收。

    「子目标完成」的正式验收在 orchestrator（它把子目标标 SUCCESS）。
    汇报是**每子目标变**的字段，所以必须判新旧：同一份汇报被判两遍，
    中间的子目标就被跳过了。
    """
    ledger = ReportLedger()
    state = _state(executor_report=_report("completed"))
    assert post_cortex_gate(state, ledger) == "review"


def test_r5b_completed_report_wins_over_pending_goal():
    """R5b：既有新汇报、又有待跑目标 → **优先 review**。

    顺序是刻意的：先验收完成，再继续下一个子目标（契约 §W8.3）。
    反过来会让「刚做完的子目标」被当成没做完，验收永远轮不到。
    """
    ledger = ReportLedger()
    state = _state(
        executor_report=_report("completed"),
        executor_goal="读出 Wi-Fi 名称",
        executor_steps=0,
    )
    assert post_cortex_gate(state, ledger) == "review"


def test_r6_seen_report_is_not_reviewed_again():
    """R6：**同一份**汇报（账本已经判过了）→ 不是 review。

    账本记得「已经判过哪一份」。同一份汇报再触发一次验收 =
    同一个子目标被推进两次 → 中间的子目标被跳过。
    （`convergence` 节点会作废已判过的汇报，但在这之前路由层就得先挡住。）
    """
    ledger = ReportLedger()
    report = _report("completed")
    ledger.mark_seen(report)

    state = _state(
        executor_report=report,
        executor_goal="读出 Wi-Fi 名称",
        executor_steps=0,
    )
    assert post_cortex_gate(state, ledger) != "review"
    # 具体落到哪：应该继续去执行下一个目标
    assert post_cortex_gate(state, ledger) == "execute"


def test_r7_pending_goal_with_zero_steps_goes_to_execute():
    """R7：有 ``executor_goal`` 且 ``executor_steps == 0`` → 去执行。

    ``executor_steps == 0`` 是「这个目标还没跑过」的信号：
    Cortex 下发新目标时把它归零，Executor 每跑一轮 +1。
    少了它，已经做完的目标会被**再跑一遍**（子目标被重复执行）。
    """
    ledger = ReportLedger()
    state = _state(executor_goal="读出 Wi-Fi 名称", executor_steps=0)
    assert post_cortex_gate(state, ledger) == "execute"


def test_r7b_already_run_goal_is_not_executed_again():
    """R7b：目标还在，但 ``executor_steps > 0`` → **不是** execute。

    这个目标已经跑过了。再执行一遍 = 同一个子目标做两遍。
    没有待办目标 → 交给 orchestrator 当看门狗。
    """
    ledger = ReportLedger()
    state = _state(executor_goal="读出 Wi-Fi 名称", executor_steps=3)
    assert post_cortex_gate(state, ledger) != "execute"
    assert post_cortex_gate(state, ledger) == "review"


def test_r8_no_report_and_no_goal_goes_to_review():
    """R8：既没有新汇报、也没有待跑目标 → 去 orchestrator（看门狗兜底）。

    这是 R16 的落点：没有这条兜底，计划级失败（子目标根本做不到）
    就没有任何处理者 —— Cortex 会一直说「没完成」一直重发，
    最后撞 ``recursion_limit`` **异常退出**（不是「重新拆计划」）。
    """
    ledger = ReportLedger()
    state = _state()
    assert post_cortex_gate(state, ledger) == "review"


# ---------------------------------------------------------------------------
# R9~R11 —— convergence_gate（计划层的汇合点）
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
    assert convergence_gate(state) == "replan"


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
    assert convergence_gate(state) == "end"


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
    assert convergence_gate(state) == "continue"


# ---------------------------------------------------------------------------
# R12 —— ReportLedger（汇报的新旧账本）
# ---------------------------------------------------------------------------


def test_r12_ledger_distinguishes_reports_by_identity_not_by_value():
    """R12：账本按**对象身份**区分汇报，不按内容。

    两份内容完全相同的汇报（pydantic 认为它们相等）必须被当成**两份**：
    汇报对象每个子目标新建一个，内容相同不代表是同一次尝试。
    如果哪天改成按内容判重，第二个子目标的汇报就会被误判成「已经判过」
    → 直接被跳过。
    """
    ledger = ReportLedger()
    first = _report("completed", summary="同样的内容")
    second = _report("completed", summary="同样的内容")

    # 前置：两者在 pydantic 层面**相等**（内容一样）
    assert first == second
    # 但账本必须按身份区分
    assert ledger.is_seen(first) is False
    assert ledger.is_seen(second) is False

    ledger.mark_seen(first)
    assert ledger.is_seen(first) is True
    assert ledger.is_seen(second) is False


def test_r12b_ledger_ignores_none():
    """R12b：``None`` 汇报不算「判过了」（否则第一份汇报会被直接吞掉）。"""
    ledger = ReportLedger()
    assert ledger.is_seen(None) is False


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


def test_graph_has_reporter_to_orchestrator_watchdog_edge():
    """**无条件**边 ``reporter → orchestrator`` 必须存在（看门狗，R16）。

    没有它，计划级失败就没有任何处理者：Cortex 一直重发同一句
    「没完成」，最后撞 ``recursion_limit`` 异常退出，而不是「重新拆计划」。

    必须是**无条件**边 —— 挂成条件边就意味着「有时候没有处理者」。
    """
    assert ("reporter", "orchestrator") in _edges()
    assert _branch("reporter", "orchestrator") is None, (
        "reporter → orchestrator 必须是无条件边（条件边 = 有时候没人处理失败）"
    )


def test_graph_has_executor_tools_to_summarizer_edge():
    """``executor_tools → summarizer``：工具执行完先裁剪 messages。"""
    assert ("executor_tools", "summarizer") in _edges()


def test_graph_has_summarizer_to_contextor_edge():
    """``summarizer → contextor``：工具执行完**必须回 contextor** 重新抓屏。

    不能直接回 executor：那样工具解析 ``Target.index`` 时拿到的是
    **上一屏**的编号，点错元素（索引契约，方案 §9.1 / R19）。
    """
    assert ("summarizer", "contextor") in _edges()


def test_graph_has_contextor_to_executor_edge():
    """``contextor → executor``：抓屏 → 决策（主循环入口）。"""
    assert ("contextor", "executor") in _edges()


def test_graph_has_executor_retry_self_loop():
    """``executor → executor``（retry 边）必须存在。

    致命 bug ② 的修复：动作被 dispatcher 拒了以后，
    **拒绝理由要回灌给模型**（HumanMessage），模型带着理由重新决策。
    少了这条自环，理由就永远送不到模型手里 ——
    路由直接进了 reporter，Executor 的自主循环第一次出岔子就终止。
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
        "orchestrator",
        "contextor",
        "cortex",
        "executor",
        "executor_tools",
        "reporter",
        "summarizer",
        "convergence",
    ):
        assert required in nodes, f"图里少了节点 {required}"
