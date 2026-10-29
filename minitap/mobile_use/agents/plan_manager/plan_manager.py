"""PlanManager：纯代码的计划状态推进器 + 规则看门狗。

2026-10-29：替代 Orchestrator 的 LLM 验收。

原来的 Orchestrator 只在 Cortex 提名完成时才真正调 LLM；它看不到
``ExecutorReport``，只能凭二手 thoughts 做二次确认。那次 LLM 调用是每个
完成子目标的固定开销之一，但判断力并不比提名方强。现在：

* **完成判断只有一个源**：Cortex 提名，PlanManager 只做确定性的状态写入；
* **失败兜底改成规则**：同一个子目标下发超过 ``MAX_ISSUES_PER_SUBGOAL`` 次
  → 标 ``FAILURE`` → replan。它是「Cortex 一直重发到撞 recursion_limit」
  的确定性替代，不再依赖 LLM 的 ``needs_replaning``；
* 计划状态的写全部集中在本节点，其它节点只读。

路由拆成两个纯函数：

* :func:`post_cortex_gate` —— Cortex 之后：execute / review / budget_exceeded；
* :func:`plan_manager_gate` —— PlanManager 之后：continue / replan / end。
"""

from __future__ import annotations

from typing import Any, Literal

from minitap.mobile_use.agents.cortex.issue_log import count_goal_issues
from minitap.mobile_use.agents.planner.utils import (
    all_completed,
    complete_subgoals_by_ids,
    fail_current_subgoal,
    get_current_subgoal,
    nothing_started,
    one_of_them_is_failure,
    start_next_subgoal,
)
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.utils.decorators import wrap_with_callbacks
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

#: 同一个子目标最多允许下发几次。
#:
#: 数的是 ``Issued goal for subgoal <id>`` 这条 thought —— 初发、规则换招、
#: LLM 换招都算。3 = 最多 3 次尝试；第 4 次在下发前被拦下。
#:
#: 这个预算比 Cortex 的 ``RULE_RETRY_BUDGET``（1 次规则换招）宽：规则换招只是
#: 省钱手段，用完仍可以让 LLM 继续换招；这里是计划级的最后一道闸，超过就认定
#: 「这个子目标在当前计划下走不通」，标 FAILURE 让 Planner 重新拆。
MAX_ISSUES_PER_SUBGOAL = 3


def should_break_reissue_loop(state: State) -> bool:
    """当前子目标是不是已经被下发到超过预算了。

    只在「刚下发、还没跑过」时成立（``executor_goal`` 有值且
    ``executor_steps == 0``）。一次尝试结束后 ``executor_steps > 0``，那不是
    一次新下发，不该消耗预算。
    """
    current = get_current_subgoal(state.subgoal_plan)
    if current is None:
        return False
    if not (state.executor_goal and state.executor_steps == 0):
        return False
    return count_goal_issues(state.agents_thoughts, current.id) > MAX_ISSUES_PER_SUBGOAL


def post_cortex_gate(
    state: State,
) -> Literal["execute", "review", "budget_exceeded"]:
    """Cortex 之后的路由。

    * ``execute`` → Contextor：Cortex 刚下发了一个还没跑过的目标。
    * ``review`` → PlanManager：验收完成提名，或处理计划失败 / 看门狗。
    * ``budget_exceeded`` → PlanManager：这个子目标已经下发太多次，交给
      PlanManager 标 FAILURE 并 replan。

    汇报的新旧不需要在这里判：Cortex 消费完汇报就把 ``executor_done`` 清成
    False，所以「有待跑的目标」仍然是 ``executor_steps == 0`` 这一个信号。
    """
    if state.executor_goal and state.executor_steps == 0:
        if should_break_reissue_loop(state):
            logger.info("[plan_manager] Reissue budget exceeded; routing to PlanManager")
            return "budget_exceeded"
        logger.info(f"[plan_manager] Executing subgoal: {state.executor_goal}")
        return "execute"

    logger.info("[plan_manager] No pending subgoal to execute; going to PlanManager")
    return "review"


def plan_manager_gate(
    state: State,
) -> Literal["continue", "replan", "end"]:
    """计划层汇合点：继续下一个子目标 / 重规划 / 结束。"""
    logger.info("Starting plan_manager_gate")

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


class PlanManagerNode:
    """计划状态的唯一写者：启动 / 验收 / 失败，全部是确定性代码。"""

    def __init__(self, ctx: MobileUseContext) -> None:
        self.ctx = ctx

    @wrap_with_callbacks(
        before=lambda: logger.info("Starting PlanManager..."),
        on_success=lambda _: logger.success("PlanManager"),
        on_failure=lambda _: logger.error("PlanManager"),
    )
    async def __call__(self, state: State) -> dict[str, Any]:
        update: dict[str, Any] = {}
        thoughts: list[str] = []

        current = get_current_subgoal(state.subgoal_plan)

        if should_break_reissue_loop(state):
            # 预算熔断：在「刚下发、还没跑过」时把当前子目标判死，交给 Planner。
            assert current is not None
            update["subgoal_plan"] = fail_current_subgoal(state.subgoal_plan)
            update["complete_subgoals_by_ids"] = []
            thoughts.append(
                f"Subgoal {current.id} exceeded the reissue budget "
                f"({MAX_ISSUES_PER_SUBGOAL}); replanning"
            )
        elif one_of_them_is_failure(state.subgoal_plan):
            # Cortex 已经标了 FAILURE；路由会去 replan，这里不启动新目标。
            pass
        elif all_completed(state.subgoal_plan):
            # 全部完成：路由会去 END。
            pass
        elif state.complete_subgoals_by_ids:
            # Cortex 提名完成。真正写 SUCCESS 的是这里（Cortex 只提名）。
            completed_ids = [
                subgoal.id
                for subgoal in state.subgoal_plan
                if subgoal.id in state.complete_subgoals_by_ids
            ]
            plan = complete_subgoals_by_ids(
                subgoals=state.subgoal_plan,
                ids=state.complete_subgoals_by_ids,
            )
            thoughts.append("Accepted completion for " + ", ".join(completed_ids))
            if not all_completed(plan):
                plan = start_next_subgoal(plan)
                thoughts.append(f"Starting the next subgoal: {get_current_subgoal(plan)}")
            else:
                thoughts.append("All subgoals completed")
            update["subgoal_plan"] = plan
            update["complete_subgoals_by_ids"] = []
        elif nothing_started(state.subgoal_plan) or current is None:
            # 首次规划，或 replan 之后还没有 PENDING 的子目标：启动下一个。
            first = nothing_started(state.subgoal_plan)
            plan = start_next_subgoal(state.subgoal_plan)
            next_subgoal = get_current_subgoal(plan)
            if next_subgoal is not None:
                thoughts.append(
                    ("Starting the first subgoal: " if first else "Starting the next subgoal: ")
                    + str(next_subgoal)
                )
                update["subgoal_plan"] = plan
                update["complete_subgoals_by_ids"] = []
        # else: 防御性空转。reporter 合并进 cortex 之后，本节点只从 planner
        # （首次规划）或 cortex（review）进来，后者的三种决定各自带走一个分支，
        # 走到这里说明状态自相矛盾（有当前子目标、没人失败、也没人完成）——
        # 于是交给 gate 判 continue，让 Cortex 重新决定，而不是在这里猜。

        if update.get("subgoal_plan") is not None and self.ctx.on_plan_changes:
            await self.ctx.on_plan_changes(update["subgoal_plan"], False)

        if not update and not thoughts:
            return {}
        if thoughts:
            update["agents_thoughts"] = thoughts
        return await state.asanitize_update(
            ctx=self.ctx,
            update=update,
            agent="plan_manager",
        )
