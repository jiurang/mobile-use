"""Executor：自主执行者。

Cortex 只给一个子目标（一句自然语言），Executor 自己看屏、决策、执行，循环到
子目标完成或熔断。

三处设计撑着这一环：

- 不 bind_tools 设备工具。那 16 个工具的 schema 不进缓存键，每轮都按满价计费；
  工具用法改由 ``executor.md`` 里的文本工具表教，只绑定 ``ExecutorOutput``
  一个结构化输出 schema。
- dispatcher 拒绝动作时要把拒绝理由回灌给模型。退回 LLM 只会拿回同样的坏动作。
- prompt 里每轮变化的内容（屏幕元素表）一律放最后，避免破坏前缀缓存。
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable

from minitap.mobile_use.agents.cortex.dispatcher import dispatch_decisions_with_reason
from minitap.mobile_use.agents.executor.progress import (
    action_fingerprint,
    detect_stall,
    screen_state_fingerprint,
    should_nudge,
)
from minitap.mobile_use.agents.executor.prompt import (
    build_executor_messages,
    render_device_state,
    render_dispatch_rejection,
    render_done_fallback,
    render_extra_actions_notice,
    render_failed_fallback,
    render_no_goal_message,
    render_retry_nudge,
    render_stall_notice,
    render_step_limit_message,
    render_vision_failure,
)
from minitap.mobile_use.agents.executor.types import ExecutorOutput
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.services.llm import (
    StructuredOutputError,
    get_structured_llm,
    invoke_llm_with_timeout_message,
)
from minitap.mobile_use.services.telemetry import telemetry
from minitap.mobile_use.services.vision import resolve_vision_targets
from minitap.mobile_use.tools.index import (
    EXECUTOR_WRAPPERS_TOOLS,
    VIDEO_RECORDING_WRAPPERS,
    get_tools_from_wrappers,
)
from minitap.mobile_use.utils.app_names import resolve_app_display_name
from minitap.mobile_use.utils.decorators import wrap_with_callbacks
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

# 单个子目标最多执行几步；默认不限（None），超了才强制收尾并汇报给 Cortex。
# 放开上限是有原因的：上限会腰斩正常子目标，也会掩盖原地打转的真因。
# 兜底是任务级的全局步预算（``constants.DEFAULT_MAX_STEPS``，最终撞
# ``recursion_limit``）；代价是一个子目标可能吃光整个任务预算，撞
# ``recursion_limit`` 时任务直接失败。
# 要恢复上限：把这里改成整数即可（代码路径都还在）。
MAX_STEPS_PER_SUBGOAL: int | None = None


def _rejection_code(reason: str | None) -> str:
    """把拒绝理由压成短代码（只进遥测，不参与控制流）。

    理由原文可能含屏幕上的文字（用户数据），不外传。这里按子串匹配 dispatcher
    的自然语言措辞，匹配失败退化成 unknown，改措辞会影响分布。
    """
    if not reason:
        return "unknown"
    if "not a known tool" in reason:
        return "unknown_tool"
    if "not a valid argument" in reason:
        return "unknown_arg"
    if "missing required argument" in reason:
        return "missing_arg"
    if "must be an object" in reason:
        return "bad_arg_type"
    if "could not be parsed" in reason:
        return "unparseable"
    if "agent_thought is required" in reason:
        return "no_thought"
    return "unknown"


class ExecutorNode:
    """决策一步；循环在 graph 层（``graph/graph.py``），每一步是一个 super-step。

    这样状态（尤其屏幕元素表）在步骤之间保持一致，索引契约才成立。
    """

    def __init__(self, ctx: MobileUseContext) -> None:
        self.ctx = ctx

    @wrap_with_callbacks(
        before=lambda: logger.info("Starting Executor Agent..."),
        on_success=lambda _: logger.success("Executor Agent"),
        on_failure=lambda _: logger.error("Executor Agent"),
    )
    async def __call__(self, state: State) -> dict[str, Any]:
        goal = state.executor_goal
        if not goal:
            # 正常流程不会走到这，但兜住比崩掉好。
            logger.warning("[executor] No goal in state; nothing to execute")
            return await state.asanitize_update(
                ctx=self.ctx,
                update={
                    "executor_done": True,
                    "executor_messages": [
                        AIMessage(content=render_no_goal_message(language=self.ctx.prompt_language))
                    ],
                },
                agent="executor",
            )

        wrappers = list(EXECUTOR_WRAPPERS_TOOLS)
        if self.ctx.video_recording_enabled:
            wrappers.extend(VIDEO_RECORDING_WRAPPERS)
        tools = get_tools_from_wrappers(self.ctx, wrappers)

        # 全任务累计动作步数：本节点每跑一轮 +1。
        # ``executor_steps`` 会在 Cortex 换目标时归零，``total_steps`` 不归零，
        # 供 SDK 上报整个任务的动作步数。无目标兜底分支不计入两者。
        step_index = state.executor_steps + 1
        next_total_steps = state.total_steps + 1

        async def emit(
            *messages,
            done: bool = False,
            gave_up: bool = False,
            progress: list[dict] | None = None,
        ) -> dict[str, Any]:
            """写一步的结果。几个出口共用同一份计数器。"""
            update: dict[str, Any] = {
                "executor_done": done,
                "executor_failed": gave_up,
                "executor_steps": step_index,
                "total_steps": next_total_steps,
                "executor_messages": list(messages),
            }
            if progress is not None:
                update["executor_progress"] = progress
            return await state.asanitize_update(
                ctx=self.ctx,
                update=update,
                agent="executor",
            )

        if MAX_STEPS_PER_SUBGOAL is not None and step_index > MAX_STEPS_PER_SUBGOAL:
            # 熔断：仅配了上限时触发；到上限不再调 LLM，直接收尾交给 Cortex 换招。
            logger.warning(
                f"[executor] Step limit reached ({MAX_STEPS_PER_SUBGOAL}); "
                "stopping this subgoal attempt"
            )
            return await emit(
                AIMessage(
                    content=render_step_limit_message(
                        MAX_STEPS_PER_SUBGOAL, language=self.ctx.prompt_language
                    )
                ),
                done=True,
            )

        # 「没有进展」的检测，只提醒，不拦截。
        #
        # 实测 executor 会在卡住的子目标上重复几十步，既不说完成也不说做不到，
        # 最终撞 recursion_limit 崩掉。这里把「重复」作为事实告诉模型，由它自己
        # 决定换招还是放弃；不截断，也不改结果。
        progress = list(state.executor_progress or [])
        stall_signal = detect_stall(progress)
        stall_notice = ""
        nudged = False
        if stall_signal is not None and should_nudge(stall_signal):
            stall_notice = render_stall_notice(
                stall_signal.description, language=self.ctx.prompt_language
            )
            nudged = True
            logger.info(
                f"[executor] No progress detected ({stall_signal.kind}, "
                f"{stall_signal.repeats}x in {stall_signal.window} steps); "
                "telling the model it may report 'cannot be done'"
            )

        response = await self._decide(state=state, goal=goal, stall_notice=stall_notice)

        # 目标达成：不碰 actions，直接收尾。
        #
        # done 时 actions 是空数组，会被 dispatcher 判成空动作拒绝，所以必须
        # 先在派发前判定。
        if response.done:
            return await emit(
                AIMessage(
                    content=response.done_reason
                    or response.agent_thought
                    or render_done_fallback(language=self.ctx.prompt_language)
                ),
                done=True,
            )

        # 明确放弃：收尾，outcome 判 failed 而不是 completed。
        #
        # 没有这条出口时「做不到」在契约里不可表达，模型只能一直试到撞
        # recursion_limit。有了它，Cortex 可以换招或放弃，任务继续往下走。
        if response.failed:
            logger.info("[executor] Model reported the goal cannot be achieved; wrapping up")
            return await emit(
                AIMessage(
                    content=response.failed_reason
                    or response.agent_thought
                    or render_failed_fallback(language=self.ctx.prompt_language)
                ),
                done=True,
                gave_up=True,
            )

        # 动作转 tool_calls（确定性直连）。
        #
        # 一轮只执行一个动作：元素编号只在那一屏上成立，多执行的动作其下标指向的
        # 是已变更的屏幕。这里做硬兜底：多余的丢弃并回灌给模型。
        actions = list(response.actions[:1])
        if len(response.actions) > 1:
            logger.info(
                f"[executor] Model returned {len(response.actions)} actions; "
                "executing only the first one (element indices are only valid "
                "for one screen)"
            )

        # 视觉定位：模型用 `target.vision` 描述一个两条文字通道都叫不出名字的
        # 目标时，在这里把它解析成当屏坐标（一次截图 + 一次小视觉调用）。
        #
        # 失败就**不派发** —— 和 dispatcher 拒绝同一条出口：把具体原因回灌给模型，
        # 让它改用 index 或换个描述。绝不能带着空的 target 去执行（那会点空）。
        actions, vision_error = await resolve_vision_targets(self.ctx, actions)
        if vision_error:
            logger.info(f"[executor] Vision lookup failed: {vision_error}")
            return await emit(
                AIMessage(content=response.agent_thought or ""),
                HumanMessage(
                    content=render_vision_failure(vision_error, language=self.ctx.prompt_language)
                ),
            )

        decisions_json = json.dumps(actions, ensure_ascii=False)
        tool_calls, reject_reason = dispatch_decisions_with_reason(
            decisions=decisions_json,
            default_thought=response.agent_thought,
            tools=tools,
        )

        if tool_calls is None:
            # dispatcher 拒了：回灌理由让模型重新决策。退回 LLM 只会重复同样的错。
            logger.info(f"[executor] Dispatch rejected: {reject_reason}")
            telemetry.capture_executor_dispatch(
                task_id=self.ctx.trace_id,
                dispatched=False,
                action_count=len(response.actions),
                rejection_code=_rejection_code(reject_reason),
            )
            # 用 HumanMessage 而非 ToolMessage：拒绝时没有对应的 tool_call_id，
            # OpenAI 兼容端点会 400。
            return await emit(
                AIMessage(content=response.agent_thought or ""),
                HumanMessage(
                    content=render_dispatch_rejection(
                        reject_reason or "", language=self.ctx.prompt_language
                    )
                ),
            )
        telemetry.capture_executor_dispatch(
            task_id=self.ctx.trace_id,
            dispatched=True,
            action_count=len(actions),
        )

        # 丢弃了额外动作：告诉模型那些动作没执行，别误以为做过。
        extra_note = ""
        if len(response.actions) > 1:
            extra_note = render_extra_actions_notice(
                len(response.actions), language=self.ctx.prompt_language
            )

        return await emit(
            AIMessage(content=extra_note, tool_calls=tool_calls),
            progress=[
                *progress,
                {
                    "action": action_fingerprint(actions[0] if actions else None),
                    "screen": screen_state_fingerprint(
                        state.latest_screen_elements, state.latest_ocr_results
                    ),
                    "nudged": nudged,
                },
            ],
        )

    async def _decide(
        self,
        *,
        state: State,
        goal: str,
        stall_notice: str = "",
    ) -> ExecutorOutput:
        """问一次 LLM，拿结构化输出。

        只绑定 ``ExecutorOutput`` 一个 schema（见 ``get_structured_llm``）。
        解析失败不返回 None，内部会降级成「空动作 + 说明」。
        """
        # 设备状态模板块，措辞在 executor.md / executor-zh.md 里：
        # 息屏 / 无感知通道要告知模型，否则它会在黑屏上反复尝试。
        device_state = render_device_state(
            device_date=state.device_date,
            # 给模型看应用名（设备 APK 里读，读不到退回内置表）；state 里仍存包名，
            # app lock / 宏前置条件要精确包名。
            focused_app_info=await resolve_app_display_name(self.ctx, state.focused_app_info),
            screen_off=state.screen_state == "off",
            no_perception=not (self.ctx.perception.ui_hierarchy or self.ctx.perception.ocr),
            language=self.ctx.prompt_language,
        )
        device_state_lines = [device_state] if device_state else []

        messages = build_executor_messages(
            ctx=self.ctx,
            goal=goal,
            screen_elements=state.latest_screen_elements,
            failures=state.executor_failures,
            local_messages=state.executor_messages,
            device_state_lines=device_state_lines,
            # iOS / 云设备：无统一元素表时用 legacy OCR 块
            # （编号与 latest_ocr_results 下标严格对应）。
            ocr_results=state.latest_ocr_results,
            stall_notice=stall_notice,
        )

        # 按 provider 能力分派结构化输出（llamacpp 走原生 json_schema）。
        schema = ExecutorOutput
        runner = get_structured_llm(ctx=self.ctx, name="executor", schema=schema)

        response, failure_reason = await self._invoke(runner, messages, schema)

        # 解析失败：带 nudge 重试一次。
        #
        # 主 / fallback 是本机同一个模型时 ``with_fallback`` 兜不住，异常会冒到
        # 顶层杀掉任务。这里自己重试并明确告诉模型错在哪（failure_reason 来自
        # 解析层），再失败就降级成「空动作 + 说明」，让流程继续。
        if response is None:
            logger.warning(
                f"[executor] LLM returned no parseable output ({failure_reason}); retrying once"
            )
            retry_messages = [
                *messages,
                HumanMessage(
                    content=render_retry_nudge(failure_reason, language=self.ctx.prompt_language)
                ),
            ]
            response, retry_reason = await self._invoke(runner, retry_messages, schema)
            if response is None:
                logger.warning(
                    "[executor] Still unparseable after retry "
                    f"({retry_reason}); degrading to an empty "
                    "action list so the loop can continue"
                )
                response = ExecutorOutput(
                    actions=[],
                    agent_thought=(
                        "I could not produce a readable action this round; "
                        "reporting back so the supervisor can redirect me."
                    ),
                )

        # 遥测：这一步花了多少 token。
        # usage 由 ``services/llm.py`` 的 ``capture_llm_usage`` 统一上报
        # （``invoke_llm_with_timeout_message`` 内部已开 / 关捕获槽），
        # 本节点不重复埋点。
        telemetry.capture_executor_step(
            task_id=self.ctx.trace_id,
            step_index=state.executor_steps + 1,
            actions_count=len(response.actions),
            done=bool(response.done),
        )

        return response

    async def _invoke(
        self,
        runner: Runnable[Any, Any],
        messages: list[BaseMessage],
        schema: type[ExecutorOutput],
    ) -> tuple[ExecutorOutput | None, str | None]:
        """调一次 LLM；解析失败返回 ``(None, 原因)``，不抛。

        ``with_fallback`` 在这里没用：本机 fallback 常与主模型相同，重试结果一样。
        第二个返回值是给模型看的具体失败原因，便于它照着改。
        """
        try:
            response = await invoke_llm_with_timeout_message(
                runner.ainvoke(messages),
                agent_name="executor",
                task_id=self.ctx.trace_id,
            )
        except StructuredOutputError as e:
            # 只记 facts（工具名 / finish_reason），不记 detail。
            # detail 含模型原文，屏幕文字属于用户数据。
            logger.warning(f"[executor] structured output unusable: {e.reason} | {e.facts}")
            return None, e.reason
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[executor] LLM call failed: {type(e).__name__}: {str(e)[:200]}")
            return None, f"{type(e).__name__}: {str(e)[:200]}"
        # 必须对着实际使用的 schema 类判，不能写死 ExecutorOutput：schema 由调用方
        # 传进来，写死一旦对不上就会恒为 False，动作被丢弃。
        if not isinstance(response, schema):
            return None, f"the answer was not an {schema.__name__} object"
        return response, None

    # 屏幕截图不进 Executor 的 prompt：它只有 UI 元素表和 OCR 两条文字通道。
    # 截图本身仍用于 OCR、坐标校验，以及 `target.vision` 的按需视觉定位
    # （services/vision.py：一次独立小调用，图不进本地历史，也不动前缀缓存）。
