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
   小 schema（走 `services/llm.py` 的 `get_structured_llm`：默认把 schema 当
   工具绑，llamacpp 这类 provider 走原生 json_schema 语法约束）。

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

import json

from langchain_core.messages import AIMessage, HumanMessage

from minitap.mobile_use.agents.cortex.dispatcher import dispatch_decisions_with_reason
from minitap.mobile_use.agents.executor.prompt import build_executor_messages
from minitap.mobile_use.agents.executor.types import ExecutorOutput
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.services.llm import (
    StructuredOutputError,
    get_structured_llm,
    invoke_llm_with_timeout_message,
)
from minitap.mobile_use.services.telemetry import telemetry
from minitap.mobile_use.tools.index import (
    EXECUTOR_WRAPPERS_TOOLS,
    VIDEO_RECORDING_WRAPPERS,
    get_tools_from_wrappers,
)
from minitap.mobile_use.utils.decorators import wrap_with_callbacks
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

# 单个子目标最多执行几步。**默认不限（None）** —— 超了才强制收尾，汇报给 Cortex。
#
# 🔴 **2026-10-22 改：原来是 12，现在是「不限」。**
#
# 两条实测证据都说明那个 12 是错的：
#
# 1. **它会腰斩正常子目标。** 真机跑一个「往上滚到结果页顶部，确认第一条是否带
#    『广告』标识」的子目标 —— 12 步不够，被截断成 `outcome=blocked`
#    → 触发换招 → 重发。**子目标本来是做得完的**（用户提供的现场：
#    `swipe_percentages(...)` 之后紧跟 `step limit of 12 reached`）。
#    12 这个数的依据只是**一次**运行的 4.0~5.3 步/子目标（n=1），而
#    「往上滚到顶部再读」这种子目标天然就比平均值长得多。
#
# 2. **它掩盖了真因（限制-症状倒挂）。** 执行器原地打转时，它把「打转」腰斩成
#    「这个子目标走不通」→ 换招重来 → 清空局部历史 → 更瞎。
#    **限制把症状盖住了，还顺手制造了新的因果。**
#
# ⇒ 现在**默认不限**：子目标做到模型说 done / 失败为止。
#
# **兜底是谁**：全局动作步预算（`constants.DEFAULT_MAX_STEPS` → LangGraph 的
# `recursion_limit`）。那是**任务级**预算，不是子目标级的。
# ⚠️ 代价要说清：没有子目标级上限，**一个子目标就能吃掉整个任务预算**，
# 撞到 `recursion_limit` 时是 `GraphRecursionError`（任务失败，不是优雅收尾）。
# 这是「不做限制」的必然代价 —— 要不要把全局预算耗尽改成**优雅收尾 + 上报**
# （而不是崩），是下一个要拍板的点。
#
# 要恢复上限：把这个值改成整数（代码路径都还在，只是默认不触发）。
MAX_STEPS_PER_SUBGOAL: int | None = None


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
                    "executor_messages": [AIMessage(content="no goal was provided, cannot act")],
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

        if MAX_STEPS_PER_SUBGOAL is not None and step_index > MAX_STEPS_PER_SUBGOAL:
            # 熔断：**只有在配了上限时才触发**（默认 None = 不限，
            # 见上面 `MAX_STEPS_PER_SUBGOAL` 的说明）。
            # 到上限了就不该再调 LLM（那只会继续磨）。
            # 直接收尾，把「已经试过什么」交给 Cortex 换招。
            logger.warning(
                f"[executor] Step limit reached ({MAX_STEPS_PER_SUBGOAL}); "
                "stopping this subgoal attempt"
            )
            return await emit(
                AIMessage(
                    content=(
                        f"step limit of {MAX_STEPS_PER_SUBGOAL} reached; stopping and reporting"
                    )
                ),
                done=True,
            )

        response = await self._decide(state=state, goal=goal)

        # 🔴 **2026-10-22 删掉了「同轮二问」**（原来叫 `need_screenshot`）。
        #
        # 原来这里：模型说「我需要看图」→ 补一张图 → **再问一次 LLM**。
        # 那是整整一次 LLM 调用（真机实测 executor 单次决策中位 5,276ms），
        # 换来的只是「模型偶尔能看见颜色/图标/遮挡」。
        #
        # 现在截图通道整条关掉（见 `agents/executor/types.py` 的说明）。
        # 模型从���只靠 **UI 元素表 + OCR** 决策。

        # ---- 目标达成 → 直接收尾，**不碰 actions** ----
        #
        # ⚠️ 顺序至关重要：`done: true` 时 `actions` 是**空数组**，
        # 而空数组会被 dispatcher 判成「空动作拒绝」。所以 `done` 必须
        # **在派发之前**判定 —— 否则这里提前 return，`done` 路径永远不可达
        # （独立验收 2026-10-21 查出的致命 bug 1）。
        if response.done:
            return await emit(
                AIMessage(content=response.done_reason or response.agent_thought or "goal reached"),
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
    ) -> ExecutorOutput:
        """问一次 LLM，拿结构化输出。

        **只绑定 `ExecutorOutput` 这一个 schema**（走 `services/llm.py` 的
        `get_structured_llm`），**不** bind 那 16 个设备工具。

        解析失败**不会**返回 None —— 内部已经降级成「空动作 + 说明」，
        调用方拿到的永远是一个可用的 `ExecutorOutput`。
        """
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
            # iOS / 云设备：没有统一元素表时用 legacy OCR 块（编号与
            # latest_ocr_results 的下标严格对应，工具侧按那个下标解析）。
            ocr_results=state.latest_ocr_results,
        )

        # 结构化输出按 **provider 能力**分派：llamacpp 走原生 json_schema 语法约束，
        # 其它 provider 保持原样（schema 当工具绑）—— 见 `get_structured_llm`。
        runner = get_structured_llm(ctx=self.ctx, name="executor", schema=ExecutorOutput)

        response, failure_reason = await self._invoke(runner, messages)

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
        #
        # 🔴 **2026-10-22：nudge 必须带上「错在哪」。**
        #
        # 以前这句是笼统的「Your last answer could not be read」—— 模型不知道自己
        # 错在哪，就会**换个姿势再犯同一个错**。真机实测（执行 812，快手极速版）：
        # 模型喊错工具名（`focus_and_input_text` 而不是 `ExecutorOutput`）→ 被
        # 解析成空动作 → 回灌笼统理由 → **连着 8 轮零动作**，每轮回灌的都是
        # 同一条无关理由（"you returned an empty action list"）。
        #
        # 现在 `failure_reason` 是解析层给的**具体**理由（喊错了哪个工具名 /
        # 写了散文 / 答案为空），模型能照着改。
        if response is None:
            logger.warning(
                f"[executor] LLM returned no parseable output ({failure_reason}); retrying once"
            )
            retry_messages = [
                *messages,
                HumanMessage(
                    content=(
                        f"Your last answer could not be used: {failure_reason}. "
                        "You must call the "
                        "ExecutorOutput tool with the fields (actions, agent_thought, "
                        "done, done_reason). Do not write the output "
                        "as plain text — a plain-text answer is discarded."
                    )
                ),
            ]
            response, retry_reason = await self._invoke(runner, retry_messages)
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

    async def _invoke(self, runner, messages) -> tuple[ExecutorOutput | None, str | None]:
        """调一次 LLM；解析失败返回 ``(None, 原因)``（不抛）。

        ``with_fallback`` 在这里没用：它切到 fallback 模型，但本机 override 里
        fallback 和主模型常常是同一个模型，重试结果一样。

        第二个返回值是**给模型看的**失败原因（短、能照着改）：解析层已经能分清
        「喊错了工具名」「写了散文」「答案为空」，这里原样带上去 ——
        上层才能把它讲回模型，而不是笼统地说「读不懂」。
        """
        try:
            response = await invoke_llm_with_timeout_message(
                runner.ainvoke(messages),
                agent_name="executor",
                task_id=self.ctx.trace_id,
            )
        except StructuredOutputError as e:
            # 只记 facts（工具名 / finish_reason），**不记 detail** ——
            # detail 里有模型的原文，屏幕上的文字就是用户数据，不该进日志。
            logger.warning(f"[executor] structured output unusable: {e.reason} | {e.facts}")
            return None, e.reason
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[executor] LLM call failed: {type(e).__name__}: {str(e)[:200]}")
            return None, f"{type(e).__name__}: {str(e)[:200]}"
        if not isinstance(response, ExecutorOutput):
            return None, f"the answer was not an {ExecutorOutput.__name__} object"
        return response, None

    # 🔴 **2026-10-22 删掉了 `_capture_screenshot`。**
    #
    # 原来它给「同轮二问」补图：把 `state.latest_screenshot`（一张全分辨率
    # base64 PNG）解码 → 等比缩到 610x1356 → 重编码成 JPEG（实测 65ms），
    # 然后**再问一次 LLM**（真机实测 executor 单次决策中位 5,276ms）。
    #
    # 现在截图通道整条关掉（见 `agents/executor/types.py` 的说明），
    # 这条路没有消费者了。
