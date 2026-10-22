"""Executor：自主执行者。

和改造前的差别
--------------
改造前 Executor 是**哑执行器**：Cortex 吐一份 JSON，Executor 把它机械翻译成
tool_calls，自己不调 LLM（走 `dispatch_decisions` 确定性直连）。

改造后 Executor 是**决策者**：Cortex 只给它一个**子目标**（一句自然语言），
Executor 自己看屏、自己决定下一个动作、自己执行，循环到子目标完成或熔断。

三个关键设计
------------

1. **不 bind_tools 那 16 个设备工具。**
   bind_tools 的 JSON schema 是 4,008 token，而且**不参与缓存键**（实测 E5），
   所以每轮都按满价计费。Executor 的工具用法靠 `executor.md` 的**文本工具表**
   教（365 token，静态，可缓存）。结构化输出只绑定 `ExecutorOutput` 这一个
   小 schema（`services/llm.py:529` 的 `_ToolCallStructuredRunnable`）。

2. **dispatcher 失败要回灌，不能退回 LLM。**
   旧架构里 dispatcher 失败 → 退回 LLM 重新转写，那是有意义的。
   新架构里 Executor 的 LLM **就是决策者**，「退回 LLM」退回来的是同一个
   决策者，它看不到自己错在哪 → 大概率输出同样的坏动作 → 死循环。
   所以要把**拒绝理由**当成工具结果回灌，让它带着理由重新决策。

3. **prompt 里每轮变的东西一律放最后。**
   屏幕元素表每轮变，放中间会把它之后的全部历史每轮重算 → 前缀缓存命中率
   **递减**。见 `prompt.py` 的模块文档。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage

from minitap.mobile_use.agents.cortex.dispatcher import dispatch_decisions_with_reason
from minitap.mobile_use.agents.executor.prompt import build_executor_messages
from minitap.mobile_use.agents.executor.types import ExecutorOutput
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.services.llm import get_llm, invoke_llm_with_timeout_message
from minitap.mobile_use.services.telemetry import telemetry
from minitap.mobile_use.tools.index import (
    EXECUTOR_WRAPPERS_TOOLS,
    VIDEO_RECORDING_WRAPPERS,
    get_tools_from_wrappers,
)
from minitap.mobile_use.utils.conversations import get_screenshot_message_for_llm
from minitap.mobile_use.utils.decorators import wrap_with_callbacks
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

# 单个子目标最多执行几步。超了就强制收尾，汇报给 Cortex 决定怎么办。
#
# 为什么不是 8：真机运行日志（probe_out/dump_run.log）实测单个任务的
# 轮/子目标 ≈ 4.0~5.3，但那是 n=1 的样本；新架构的 Executor 只有一句目标陈述、
# 没有全局视野，遇到走不通的子目标**不会放弃，只会磨**。
# 所以宁可先设宽一点（12），等 telemetry 统计出真实分布再收紧。
MAX_STEPS_PER_SUBGOAL = 12


def _rejection_code(reason: str | None) -> str:
    """把拒绝理由压成一个短代码（给遥测用）。

    **只放代码，不放理由原文** —— 理由里可能含屏幕上的文字（用户数据），
    而遥测是落库上报的。理由原文会回灌给模型（那是另一条路，不外传）。
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
    """决策一步，然后由 graph 决定是执行还是收尾。

    节点本身**不做循环**——循环在 graph 层（`graph/graph.py`），
    这样每一步都是一个 super-step，状态（尤其屏幕元素表）在步骤之间
    保持一致，索引契约才成立。
    """

    def __init__(self, ctx: MobileUseContext):
        self.ctx = ctx

    @wrap_with_callbacks(
        before=lambda: logger.info("Starting Executor Agent..."),
        on_success=lambda _: logger.success("Executor Agent"),
        on_failure=lambda _: logger.error("Executor Agent"),
    )
    async def __call__(self, state: State):
        goal = state.executor_goal
        if not goal:
            # 没有目标就没有可执行的东西。正常流程不会走到这（graph 保证
            # 有目标才进 executor），但兜住比崩掉好。
            logger.warning("[executor] No goal in state; nothing to execute")
            return await state.asanitize_update(
                ctx=self.ctx,
                update={
                    "executor_done": True,
                    "executor_messages": [
                        AIMessage(content="no goal was provided, cannot act")
                    ],
                },
                agent="executor",
            )

        wrappers = list(EXECUTOR_WRAPPERS_TOOLS)
        if self.ctx.video_recording_enabled:
            wrappers.extend(VIDEO_RECORDING_WRAPPERS)
        tools = get_tools_from_wrappers(self.ctx, wrappers)

        # 全任务累计的动作步数：**executor 节点每跑一轮 +1**。
        #
        # 口径和 `executor_steps` 的增量严格一致（同一个节点、同一个时机写），
        # 区别只有一个：`executor_steps` 会被 Cortex 换目标时归零
        # （`agents/cortex/cortex.py:429`），`total_steps` **不归零** ——
        # 它是「整个任务跑了多少个动作步」，供 SDK 上报（方案 §11.5 M13）。
        #
        # 注意：上面那个「没有目标」的兜底分支**不**计数 —— 它和
        # `executor_steps` 一样不动计数器，保持两个口径恒等。
        step_index = state.executor_steps + 1
        next_total_steps = state.total_steps + 1

        async def emit(*messages, done: bool = False):
            """写一步的结果。五个出口共用同一份计数器。"""
            return await state.asanitize_update(
                ctx=self.ctx,
                update={
                    "executor_done": done,
                    "executor_steps": step_index,
                    "total_steps": next_total_steps,
                    "executor_messages": list(messages),
                },
                agent="executor",
            )

        if step_index > MAX_STEPS_PER_SUBGOAL:
            # 熔断：到上限了就不该再调 LLM（那只会继续磨）。
            # 直接收尾，把「已经试过什么」交给 Cortex 换招。
            logger.warning(
                f"[executor] Step limit reached ({MAX_STEPS_PER_SUBGOAL}); "
                "stopping this subgoal attempt"
            )
            return await emit(
                AIMessage(
                    content=(
                        f"step limit of {MAX_STEPS_PER_SUBGOAL} reached; "
                        "stopping and reporting"
                    )
                ),
                done=True,
            )

        response = await self._decide(state=state, goal=goal)

        # ---- 模型要图：同一轮补一张再问一次 ----
        #
        # 不新开一轮：多一轮意味着多一次完整 prompt 重算 + 一次生成。
        # 同轮补图只有真正需要图的那少数轮次才多花这一次。
        if response.need_screenshot:
            screenshot_message = await self._capture_screenshot(state=state)
            if screenshot_message is not None:
                # 二问：带上截图重新决策一次。解析失败由 `_decide` 内部降级，
                # 返回的永远是一个可用的 ExecutorOutput。
                response = await self._decide(
                    state=state,
                    goal=goal,
                    screenshot_message=screenshot_message,
                )

        # ---- 目标达成 → 直接收尾，**不碰 actions** ----
        #
        # ⚠️ 顺序至关重要：`done: true` 时 `actions` 是**空数组**，
        # 而空数组会被 dispatcher 判成「空动作拒绝」。所以 `done` 必须
        # **在派发之前**判定 —— 否则这里提前 return，`done` 路径永远不可达
        # （独立验收 2026-10-21 查出的致命 bug 1）。
        if response.done:
            return await emit(
                AIMessage(
                    content=response.done_reason or response.agent_thought or "goal reached"
                ),
                done=True,
            )

        # ---- 动作 → tool_calls（确定性直连）----
        #
        # **一轮只执行一个动作** —— 这是索引契约的前提，不是 prompt 偏好。
        #
        # 元素表的编号只在**那一屏**上成立。如果一轮执行多个动作，
        # 第二个动作的 `Target.index` 指向的是**第一个动作执行前**的屏
        # —— 屏幕已经被第一个动作改过了，那个编号可能已经不是同一个元素了
        # （`state.py:67-77` 明确警告过这件事）。
        #
        # prompt 里写了「一次只做一个动作」，但那是**软约束**；这里做**硬兜底**：
        # 多出来的动作本轮不执行，回灌给模型让它下一轮重新决定。
        actions = list(response.actions[:1])
        if len(response.actions) > 1:
            logger.info(
                f"[executor] Model returned {len(response.actions)} actions; "
                "executing only the first one (element indices are only valid "
                "for one screen)"
            )

        decisions_json = json.dumps(actions, ensure_ascii=False)
        tool_calls, reject_reason = dispatch_decisions_with_reason(
            decisions=decisions_json,
            default_thought=response.agent_thought,
            tools=tools,
        )

        if tool_calls is None:
            # dispatcher 拒了。**回灌理由**，让 Executor 带着理由重新决策。
            # 不能退回 LLM 重来——那退回来的是同一个决策者，会输出同样的坏动作。
            logger.info(f"[executor] Dispatch rejected: {reject_reason}")
            telemetry.capture_executor_dispatch(
                task_id=self.ctx.trace_id,
                dispatched=False,
                action_count=len(response.actions),
                rejection_code=_rejection_code(reject_reason),
            )
            # ⚠️ 回灌**不能**用 ToolMessage：它的 tool_call_id 找不到对应的
            # assistant tool_call（拒绝时根本没有 tool_call），
            # OpenAI 兼容端点会 400（独立验收查出的致命 bug 3）。
            # 用 HumanMessage —— 语义上也更对：这不是「工具返回了错误」，
            # 是「你的上一条动作**根本没被派发**」。
            return await emit(
                AIMessage(content=response.agent_thought or ""),
                HumanMessage(
                    content=(
                        f"Your previous action was NOT executed: {reject_reason}. "
                        "Decide again — read the Tools table and the current "
                        "screen element list."
                    )
                ),
            )
        telemetry.capture_executor_dispatch(
            task_id=self.ctx.trace_id,
            dispatched=True,
            action_count=len(actions),
        )

        # 丢弃了额外动作 → 告诉模型，让它知道那些动作**没被执行**，
        # 下一轮要重新决定（否则它会以为那些动作已经做过了）。
        extra_note = ""
        if len(response.actions) > 1:
            extra_note = (
                f" Only the first of your {len(response.actions)} actions was executed; "
                "the rest were discarded and must be decided again next round."
            )

        return await emit(AIMessage(content=extra_note, tool_calls=tool_calls))

    async def _decide(
        self,
        *,
        state: State,
        goal: str,
        screenshot_message=None,
    ) -> ExecutorOutput:
        """问一次 LLM，拿结构化输出。

        **只绑定 `ExecutorOutput` 这一个 schema**（`services/llm.py:529` 的
        `_ToolCallStructuredRunnable`），**不** bind 那 16 个设备工具。

        解析失败**不会**返回 None —— 内部已经降级成「空动作 + 说明」，
        调用方拿到的永远是一个可用的 `ExecutorOutput`。
        """
        from minitap.mobile_use.services.llm import _ToolCallStructuredRunnable

        device_state_lines: list[str] = []
        if state.device_date:
            device_state_lines.append(f"Device date: {state.device_date}")
        if state.focused_app_info:
            device_state_lines.append(f"Focused app info: {state.focused_app_info}")
        if state.screen_state == "off":
            # 息屏时截图全黑、元素表只剩 systemui 残留 —— 不告诉模型，
            # 它就会在黑屏上反复尝试（真机实测：30 步 → 51 步没跑完）。
            device_state_lines.append(
                "Screen is OFF (the device is asleep). The screenshot is black and the "
                "element list only holds systemui leftovers, so there is nothing to "
                "act on. Call wake_screen FIRST, then read the screen again on the "
                "next turn."
            )

        messages = build_executor_messages(
            ctx=self.ctx,
            goal=goal,
            screen_elements=state.latest_screen_elements,
            failures=state.executor_failures,
            local_messages=state.executor_messages,
            device_state_lines=device_state_lines,
            screenshot_message=screenshot_message,
            # iOS / 云设备：没有统一元素表时用 legacy OCR 块（编号与
            # latest_ocr_results 的下标严格对应，工具侧按那个下标解析）。
            ocr_results=state.latest_ocr_results,
        )

        llm = get_llm(ctx=self.ctx, name="executor")
        # `_ToolCallStructuredRunnable` 的 llm 形参标的是 ChatOpenAI，但
        # `get_llm` 返回的是 BaseChatModel（anthropic 渠道甚至不是 ChatOpenAI）。
        # 它只用到 `bind_tools`，所以这里放宽成 Any —— services/llm.py 不动它的签名。
        runner = _ToolCallStructuredRunnable(cast("Any", llm), ExecutorOutput)

        response = await self._invoke(runner, messages)

        # ---- 解析失败 → 带 nudge 重试一次 ----
        #
        # 真机实测（2026-10-21，Redmi Turbo 4）：模型有时不调用工具，直接写一段
        # 散文（"The WLAN page is now open... done: true"）。那段文字里**没有 `{`**，
        # 连 JSON 兜底提取都做不了 → 解析失败。
        #
        # 而 `with_fallback` 兜不住：它会调 fallback 模型，但**本机 override 里
        # fallback 和主模型是同一个模型** → 重试同样失败 → 异常一路冒到任务顶层，
        # **整个任务挂掉**。
        #
        # ⇒ 这里自己重试一次，并把「你必须调用工具」明确告诉模型；
        # 还不行就降级成「空动作 + 说明」，让流程继续（Reporter 会看到它，
        # 汇报成 failed，Cortex 换招重来）—— 绝不因为一次格式失误就杀掉整个任务。
        if response is None:
            logger.warning("[executor] LLM returned no parseable output; retrying once")
            retry_messages = [
                *messages,
                HumanMessage(
                    content=(
                        "Your last answer could not be read. You must call the "
                        "ExecutorOutput tool with the fields (actions, agent_thought, "
                        "done, done_reason, need_screenshot). Do not write the output "
                        "as plain text — a plain-text answer is discarded."
                    )
                ),
            ]
            response = await self._invoke(runner, retry_messages)
            if response is None:
                logger.warning(
                    "[executor] Still unparseable after retry; degrading to an empty "
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
        #
        # ⚠️ **不要**在这里自己开捕获槽（`open_capture`）：
        # `invoke_llm_with_timeout_message` 内部已经开了一个槽并在 finally 里
        # `close_capture()`（services/llm.py:141），外层再开的会被抢先关掉。
        #
        # ⇒ usage 由 `services/llm.py` 的 `telemetry.capture_llm_usage`
        # 统一上报，本节点不重复埋点。
        #
        # （2026-10-21 更正：这里原来还写着「那个槽在 LangGraph 的图里根本收不到
        # 东西」—— **假的**。真机 graph 实测 `probe_out/executor_e2e.json`：
        # executor 21 次调用 / 120,772 input / 65,672 cached / 18,114 output，
        # 全是真实非零值。捕获槽装的是同一个 list 对象，跨图可见。
        # 详见 `services/llm.py` 的 `_record_usage` 文档字符串。）
        telemetry.capture_executor_step(
            task_id=self.ctx.trace_id,
            step_index=state.executor_steps + 1,
            actions_count=len(response.actions),
            done=bool(response.done),
        )

        return response

    async def _invoke(self, runner, messages) -> ExecutorOutput | None:
        """调一次 LLM；解析失败返回 None（不抛）。

        `with_fallback` 在这里没用：它切到 fallback 模型，但本机 override 里
        fallback 和主模型常常是同一个模型，重试结果一样。
        """
        try:
            response = await invoke_llm_with_timeout_message(
                runner.ainvoke(messages),
                agent_name="executor",
                task_id=self.ctx.trace_id,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[executor] LLM call failed: {type(e).__name__}: {str(e)[:200]}")
            return None
        return response if isinstance(response, ExecutorOutput) else None

    async def _capture_screenshot(self, *, state: State):
        """抓一张当屏截图给模型（on_demand）。

        用 `state.latest_screenshot`（Contextor 刚抓的）——决策时 executor
        还没动过屏幕，所以它就是**当前**画面，不用重抓（重抓会多花一次
        adb 往返，还可能拿到不同帧）。
        """
        from minitap.mobile_use.controllers.controller_factory import create_device_controller

        screenshot = state.latest_screenshot
        if not screenshot:
            return None
        controller = create_device_controller(self.ctx)
        # 压缩是同步 PIL 操作，放到线程里避免阻塞事件循环。
        compressed = await asyncio.to_thread(
            controller.get_compressed_b64_screenshot, screenshot
        )
        return get_screenshot_message_for_llm(compressed)
