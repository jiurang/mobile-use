from functools import cache
from pathlib import Path

from jinja2 import Template
from langchain_core.messages import HumanMessage, SystemMessage

from minitap.mobile_use.agents.planner.types import PlannerOutput, Subgoal, SubgoalStatus
from minitap.mobile_use.agents.planner.utils import (
    generate_id,
    one_of_them_is_failure,
    render_plan_skeleton,
)
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.controllers.platform_specific_commands_controller import (
    get_current_foreground_package_async,
)
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.services.llm import get_llm, invoke_llm_with_timeout_message, with_fallback
from minitap.mobile_use.services.telemetry import telemetry
from minitap.mobile_use.tools.index import (
    EXECUTOR_WRAPPERS_TOOLS,
    VIDEO_RECORDING_WRAPPERS,
    format_tools_list,
)
from minitap.mobile_use.utils.decorators import wrap_with_callbacks
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

MAX_AGENTS_THOUGHTS = 25

_STATUS_LABELS = {
    SubgoalStatus.NOT_STARTED: "TODO",
    SubgoalStatus.PENDING: "IN PROGRESS",
    SubgoalStatus.SUCCESS: "DONE",
    SubgoalStatus.FAILURE: "FAILED",
}


@cache
def _load_template(filename: str) -> Template:
    return Template(Path(__file__).parent.joinpath(filename).read_text(encoding="utf-8"))


def _format_plan(subgoals: list[Subgoal]) -> str:
    """Render the previous plan for the prompt. IDs are omitted on purpose:
    they are regenerated on every plan and would only be noise for the LLM."""
    if not subgoals:
        return "(no previous plan)"
    lines = []
    for subgoal in subgoals:
        line = f"- [{_STATUS_LABELS[subgoal.status]}] {subgoal.description}"
        if subgoal.completion_reason:
            line += f" (reason: {subgoal.completion_reason})"
        lines.append(line)
    return "\n".join(lines)


def _build_subgoals(output: PlannerOutput) -> list[Subgoal]:
    """Validate and normalize the LLM output into fresh subgoals."""
    descriptions: list[str] = []
    seen: set[str] = set()
    for raw in output.subgoals:
        description = (raw.description or "").strip()
        if not description or description.lower() in seen:
            continue
        seen.add(description.lower())
        descriptions.append(description)

    if not descriptions:
        raise ValueError("Planner returned an empty plan")

    used_ids: set[str] = set()
    subgoals: list[Subgoal] = []
    for description in descriptions:
        subgoal_id = generate_id()
        while subgoal_id in used_ids:
            subgoal_id = generate_id()
        used_ids.add(subgoal_id)
        subgoals.append(
            Subgoal(
                id=subgoal_id,
                description=description,
                status=SubgoalStatus.NOT_STARTED,
            )
        )
    return subgoals


async def _invoke_planner(
    llm,
    messages: list[SystemMessage | HumanMessage],
    ctx: MobileUseContext,
    used_fallback: bool = False,
) -> list[Subgoal]:
    """Call the LLM and validate its output so that an invalid plan from the
    main model can be retried with the fallback model."""
    response = await invoke_llm_with_timeout_message(
        llm.ainvoke(messages),
        agent_name="planner",
        task_id=ctx.trace_id,
        used_fallback=used_fallback,
    )
    if response is None:
        raise ValueError("Planner LLM returned no response")
    return _build_subgoals(response)


def _append_plan_section(
    previous_log: str, version: int, is_replan: bool, subgoals: list[Subgoal]
) -> str:
    """把这一版计划追加到计划日志末尾。**不改动之前任何一段。**

    改写旧段会击穿 cortex 提示词的前缀（见 doc/cortex提示词组成v2.md Q2）：
    replan 时旧表会在中部被改写、新子目标的 ID 还是随机生成的，改一处等于
    后面全部作废。只追加则天然单调。
    """
    header = f"PLAN v{version}" + (" (replan)" if is_replan else "")
    section = f"{header}:\n{render_plan_skeleton(subgoals)}"
    return f"{previous_log}\n\n{section}" if previous_log else section


class PlannerNode:
    def __init__(self, ctx: MobileUseContext):
        self.ctx = ctx

    @wrap_with_callbacks(
        before=lambda: logger.info("Starting Planner Agent..."),
        on_success=lambda _: logger.success("Planner Agent"),
        on_failure=lambda _: logger.error("Planner Agent"),
    )
    async def __call__(self, state: State):
        needs_replan = one_of_them_is_failure(state.subgoal_plan)

        # 重规划计数（方案 §11.6「replan 触发计数」）。
        #
        # 这个埋点唯一的用途是**验证看门狗还活着（R16）**：R16 要求「计划级失败
        # 不能没人管」，新拓扑里的处理链是
        # ``reporter → orchestrator → convergence → replan → planner`` ——
        # 每一次真正走到 replan，就证明这条链通了。反过来，「跑了很久却一次
        # replan 都没有」有两种可能：一次没失败过（正常），或者**看门狗断了**
        # （子目标失败了没人接手，Cortex 一直重发到撞 recursion_limit）。
        # 这两种光看别的指标分不开。
        #
        # 埋在这里而不是 `convergence_gate`：那个是纯函数，拿不到 task_id；
        # 而 planner 正好算得出 `needs_replan`。
        #
        # ⚠️ 只报数量，不报子目标原文（原文是用户数据）。
        if needs_replan:
            telemetry.capture_replan(
                task_id=self.ctx.trace_id,
                plan_version=state.plan_version,
                failed_subgoals=sum(
                    1 for s in state.subgoal_plan if s.status == SubgoalStatus.FAILURE
                ),
                completed_subgoals=sum(
                    1 for s in state.subgoal_plan if s.status == SubgoalStatus.SUCCESS
                ),
                total_subgoals=len(state.subgoal_plan),
            )

        current_locked_app_package = (
            self.ctx.execution_setup.get_locked_app_package() if self.ctx.execution_setup else None
        )
        current_foreground_app = await get_current_foreground_package_async(self.ctx)

        executor_wrappers = list(EXECUTOR_WRAPPERS_TOOLS)
        if self.ctx.video_recording_enabled:
            executor_wrappers.extend(VIDEO_RECORDING_WRAPPERS)

        system_message = _load_template("planner.md").render(
            platform=self.ctx.device.mobile_platform.value,
            executor_tools_list=format_tools_list(ctx=self.ctx, wrappers=executor_wrappers),
            locked_app_package=current_locked_app_package,
            current_foreground_app=current_foreground_app,
            video_recording_enabled=self.ctx.video_recording_enabled,
        )
        human_message = _load_template("human.md").render(
            action="replan" if needs_replan else "plan",
            initial_goal=state.initial_goal,
            previous_plan=_format_plan(state.subgoal_plan),
            agent_thoughts="\n".join(state.agents_thoughts[-MAX_AGENTS_THOUGHTS:]),
        )
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message),
        ]

        llm = get_llm(ctx=self.ctx, name="planner").with_structured_output(PlannerOutput)
        llm_fallback = get_llm(
            ctx=self.ctx, name="planner", use_fallback=True
        ).with_structured_output(PlannerOutput)
        new_subgoals = await with_fallback(
            main_call=lambda: _invoke_planner(llm, messages, self.ctx),
            fallback_call=lambda: _invoke_planner(
                llm_fallback, messages, self.ctx, used_fallback=True
            ),
        )
        # On replan, the LLM only returns the remaining steps: already completed
        # subgoals are kept as-is so they are never executed a second time.
        completed_subgoals = [
            subgoal for subgoal in state.subgoal_plan if subgoal.status == SubgoalStatus.SUCCESS
        ]
        if completed_subgoals:
            logger.info(f"♻️ Keeping {len(completed_subgoals)} completed subgoal(s) as-is")
        subgoals_plan = completed_subgoals + new_subgoals
        logger.info("📜 Generated plan:")
        logger.info("\n".join(str(s) for s in subgoals_plan))

        if self.ctx.on_plan_changes:
            await self.ctx.on_plan_changes(subgoals_plan, needs_replan)

        return await state.asanitize_update(
            ctx=self.ctx,
            update={
                "subgoal_plan": subgoals_plan,
                # 计划日志只追加：v1、v2……旧段永不改动，cortex 渲染的就是它。
                "plan_version": state.plan_version + 1,
                "subgoal_plan_log": _append_plan_section(
                    previous_log=state.subgoal_plan_log,
                    version=state.plan_version + 1,
                    is_replan=needs_replan,
                    subgoals=new_subgoals,
                ),
            },
            agent="planner",
        )
