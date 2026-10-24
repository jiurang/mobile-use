"""Orchestrator：验收子目标 + 看门狗。

它是**完成判定的验收方**
----------------------
Cortex 只说「我判定这个子目标完成了」，真正把子目标标成 ``SUCCESS`` 的是
这里（:func:`complete_subgoals_by_ids`）。两处都能判完成，就会出现两个互相
覆盖的判断源 —— 所以分工是硬的：Cortex 提名，Orchestrator 验收。

它同时是**看门狗**
----------------
「计划还走得通吗」这个判断只在这里做：``needs_replaning`` ⇒ 当前子目标标
``FAILURE`` ⇒ convergence 判出「计划里有失败」⇒ Planner 重规划。

⚠️ 这条链是「子目标失败了没人管」的唯一兜底。真机日志里出现过「跑了很久却
一次 replan 都没有」的情况 —— 那要么是一次没失败过（正常），要么是**看门狗
断了**（Cortex 一直重发到撞 recursion_limit）。这两种光看别的指标分不开，
所以 planner 那边专门埋了 replan 计数来验证这条链还活着。

它是**唯一**会写计划状态的节点
------------------------------
除了「验收通过」（标 SUCCESS）和「计划走不通」（标 FAILURE）之外，别的节点
只读计划不改计划。
"""

from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from minitap.mobile_use.agents.orchestrator.types import OrchestratorOutput
from minitap.mobile_use.agents.planner.utils import (
    all_completed,
    complete_subgoals_by_ids,
    fail_current_subgoal,
    get_current_subgoal,
    get_subgoals_by_ids,
    nothing_started,
    start_next_subgoal,
)
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.prompt_language import load_prompt_template
from minitap.mobile_use.services.llm import get_llm, invoke_llm_with_timeout_message, with_fallback
from minitap.mobile_use.utils.decorators import wrap_with_callbacks
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.thoughts_compaction import (
    PLANNER_AUDIENCE,
    render_thoughts_history,
)

logger = get_logger(__name__)


class OrchestratorNode:
    def __init__(self, ctx: MobileUseContext):
        self.ctx = ctx

    @wrap_with_callbacks(
        before=lambda: logger.info("Starting Orchestrator Agent..."),
        on_success=lambda _: logger.success("Orchestrator Agent"),
        on_failure=lambda _: logger.error("Orchestrator Agent"),
    )
    async def __call__(self, state: State) -> dict[str, Any]:
        no_subgoal_started = nothing_started(state.subgoal_plan)
        current_subgoal = get_current_subgoal(state.subgoal_plan)

        if no_subgoal_started or not current_subgoal:
            state.subgoal_plan = start_next_subgoal(state.subgoal_plan)
            new_subgoal = get_current_subgoal(state.subgoal_plan)
            thoughts = [
                (
                    f"Starting the first subgoal: {new_subgoal}"
                    if no_subgoal_started
                    else f"Starting the next subgoal: {new_subgoal}"
                )
            ]
            return await _get_state_update(
                ctx=self.ctx, state=state, thoughts=thoughts, update_plan=True
            )

        subgoals_to_examine = get_subgoals_by_ids(
            subgoals=state.subgoal_plan,
            ids=state.complete_subgoals_by_ids,
        )
        if len(subgoals_to_examine) <= 0:
            return await _get_state_update(
                ctx=self.ctx, state=state, thoughts=["No subgoal to examine."]
            )

        system_message = load_prompt_template(
            Path(__file__).parent.joinpath("orchestrator.md"),
            self.ctx.prompt_language,
        ).render(platform=self.ctx.device.mobile_platform.value)
        human_message = load_prompt_template(
            Path(__file__).parent.joinpath("human.md"),
            self.ctx.prompt_language,
        ).render(
            initial_goal=state.initial_goal,
            subgoal_plan="\n".join(str(s) for s in state.subgoal_plan),
            subgoals_to_examine="\n".join(str(s) for s in subgoals_to_examine),
            # 同样压缩：orchestrator 每次调用都带全量 thoughts，是仅次于 cortex
            # 的第二大消费方（实测 122 轮任务里占 thoughts 总成本的 13%）。
            # 压缩规则与 cortex 完全一致（同一个 render_thoughts_history），
            # 所以两个 agent 看到的摘要形态相同，不会出现口径分裂。
            #
            # 受众也跟 cortex 一致（PLANNER_AUDIENCE）：丢掉 executor 的成功
            # 流水，只留失败类 —— orchestrator 同样是规划者，不看屏、不点元素。
            # 两个 agent 的口径必须一致，否则「同一个事实两边看到的不一样」。
            agent_thoughts=render_thoughts_history(
                state.agents_thoughts,
                enabled=self.ctx.perception.thoughts_compaction,
                audience=PLANNER_AUDIENCE,
            ),
        )
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message),
        ]

        llm = get_llm(ctx=self.ctx, name="orchestrator", temperature=1).with_structured_output(
            OrchestratorOutput
        )
        llm_fallback = get_llm(
            ctx=self.ctx, name="orchestrator", use_fallback=True, temperature=1
        ).with_structured_output(OrchestratorOutput)
        response: OrchestratorOutput = await with_fallback(
            main_call=lambda: invoke_llm_with_timeout_message(
                llm.ainvoke(messages),
                agent_name="orchestrator",
                task_id=self.ctx.trace_id,
            ),
            fallback_call=lambda: invoke_llm_with_timeout_message(
                llm_fallback.ainvoke(messages),
                agent_name="orchestrator",
                task_id=self.ctx.trace_id,
                used_fallback=True,
            ),
        )  # type: ignore
        if response.needs_replaning:
            thoughts = [response.reason]
            state.subgoal_plan = fail_current_subgoal(state.subgoal_plan)
            thoughts.append("==== END OF PLAN, REPLANNING ====")
            return await _get_state_update(
                ctx=self.ctx, state=state, thoughts=thoughts, update_plan=True
            )

        state.subgoal_plan = complete_subgoals_by_ids(
            subgoals=state.subgoal_plan,
            ids=response.completed_subgoal_ids,
        )
        thoughts = [response.reason]
        if all_completed(state.subgoal_plan):
            logger.success("All the subgoals have been completed successfully.")
            return await _get_state_update(
                ctx=self.ctx, state=state, thoughts=thoughts, update_plan=True
            )

        if current_subgoal.id not in response.completed_subgoal_ids:
            # The current subgoal is not yet complete.
            return await _get_state_update(
                ctx=self.ctx, state=state, thoughts=thoughts, update_plan=True
            )

        state.subgoal_plan = start_next_subgoal(state.subgoal_plan)
        new_subgoal = get_current_subgoal(state.subgoal_plan)
        thoughts.append(f"==== NEXT SUBGOAL: {new_subgoal} ====")
        return await _get_state_update(
            ctx=self.ctx, state=state, thoughts=thoughts, update_plan=True
        )


async def _get_state_update(
    ctx: MobileUseContext,
    state: State,
    thoughts: list[str],
    update_plan: bool = False,
) -> dict[str, Any]:
    update = {
        "agents_thoughts": thoughts,
        "complete_subgoals_by_ids": [],
    }
    if update_plan:
        update["subgoal_plan"] = state.subgoal_plan
        if ctx.on_plan_changes:
            await ctx.on_plan_changes(state.subgoal_plan, False)
    return await state.asanitize_update(ctx=ctx, update=update, agent="orchestrator")
