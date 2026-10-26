"""Reporter：把一次子目标尝试的执行日志压成一份汇报。

**只做压缩，不判断**（方案 §6.2.1）
--------------------------------

汇报分两层：

* **结论层** —— ``outcome``（做成了吗）+ ``summary``（人话）+ ``evidence``（证据）
* **事实层** —— ``actions_taken`` / ``failed_targets`` / ``blocked_reason``

分层的意义是让「事实不丢」有**结构性保证**，而不是靠提示词写得好：

* ``outcome`` 由 :func:`determine_outcome` 判定（纯代码），**不经 LLM** ——
  Reporter 没有机会把失败说成完成（方案 §6.2.1、R9/R10）。
* ``actions_taken`` / ``failed_targets`` / ``blocked_reason`` **从 state 里直搬**，
  逐字保留 —— 依据 ``utils/thoughts_compaction.py`` 的硬规则「事实不能压」：
  失败事实就是 Cortex 判断「换招还是放弃」的依据，压没了它就会重复派发同一个子目标。
* LLM 只写 ``summary`` 和**挑** ``evidence``（从工具结果里挑一条作为达成证据，
  §6.2.2：挑证据是需要一点判断力的，但**不需要**强模型）。

所以这个节点挂掉不会让整条任务挂掉：LLM 调不通就降级成一份纯代码生成的 summary，
流程继续往下走 —— 汇报失败不该让整条任务崩。
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Literal, cast

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field

from minitap.mobile_use.agents.cortex.issue_log import count_goal_issues
from minitap.mobile_use.agents.executor.executor import MAX_STEPS_PER_SUBGOAL
from minitap.mobile_use.agents.executor.types import ExecutorReport, determine_outcome
from minitap.mobile_use.agents.planner.utils import get_current_subgoal
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.prompt_language import (
    PromptLanguage,
    load_prompt_template,
    pick,
)
from minitap.mobile_use.services.llm import (
    get_structured_llm,
    invoke_llm_with_timeout_message,
    with_fallback,
)
from minitap.mobile_use.services.telemetry import telemetry
from minitap.mobile_use.utils.message_content import content_to_text
from minitap.mobile_use.utils.failed_targets import FailedTarget, describe, make_key
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

REPORTER_PROMPT_PATH = Path(__file__).parent / "reporter.md"

# 证据最多留多长：工具结果里可能有整屏文本，太长会把 Cortex 的 prompt 撑大。
#
# ⚠️ 2026-10-21 从 400 提到 1600：证据现在有**两段**（动作结果 + 屏幕可见文字），
# 400 装不下屏幕那一段 —— 装不下就等于没给（而「没给」正是重发的根因）。
# Cortex 的 prompt 本来只有 ~1,341 tok，多这 ~400 tok 完全扛得住。
#
# ⚠️ 2026-10-24 从 1600 提到 2400：屏幕那一段现在最多 1600 字符
# （`MAX_SCREEN_EVIDENCE_CHARS`），动作结果那截也得留得下 ——
# 两段加起来超了要**砍动作结果，不砍屏幕文字**（见 `compose_evidence`）。
MAX_EVIDENCE_LEN = 2400

# 动作参数在汇报里最多留多长（同一个理由）。
MAX_ARGS_LEN = 200


class ReporterSummary(BaseModel):
    """LLM 只写这两个字段。

    **不要**往这里加 ``outcome`` / ``actions_taken`` / ``failed_targets`` ——
    那些是事实层，由代码直搬（见模块文档）。schema 越小，模型越不容易
    「顺手」改写事实。
    """

    summary: str = Field(
        description=(
            "One or two sentences: what was done and what the result was. "
            "No narration of every step - the action list is already there."
        ),
    )
    evidence: str | None = Field(
        default=None,
        description=(
            "The concrete result from the execution log that proves the subgoal "
            "was reached (quote it). Leave empty when the attempt did not reach "
            "the goal."
        ),
    )


class ReporterNode:
    """汇报节点：执行日志 → 一份汇报。

    节点本身**不做判断**：结论由代码判（:func:`determine_outcome`），事实由代码搬，
    LLM 只写人话。所以这个节点是整个架构里唯一一个「不会把事实说错」的地方。

    省一次调用
    ----------
    真机 A/B（2026-10-21）测出：改造后**每个子目标多出 2 次固定调用**
    （reporter + cortex），而子目标内的步数常常只有 1~2 步 ——
    这两次就是纯开销。

    ⇒ `llm_free=True`（默认）时**不调 LLM**：summary 用代码生成，
    evidence 从执行日志里挑。事实字段本来就全是代码直搬的，
    summary 只影响「给人看的措辞」，**不影响正确性**。
    真要更详细的总结时用 `llm_free=False`。
    """

    def __init__(self, ctx: MobileUseContext, *, llm_free: bool = True):
        self.ctx = ctx
        self.llm_free = llm_free

    async def __call__(self, state: State):
        outcome = determine_outcome_for(state)
        failed_targets = build_failed_targets(state.executor_failures)
        actions_taken = build_actions_taken(state.executor_messages)
        log_text = render_execution_log(state.executor_messages)

        summary, evidence = await self._write_summary(
            state=state,
            outcome=outcome,
            log_text=log_text,
            failed_targets=failed_targets,
        )

        report = ExecutorReport(
            outcome=outcome,
            summary=summary,
            evidence=evidence,
            actions_taken=actions_taken,
            failed_targets=failed_targets,
            blocked_reason=blocked_reason_for(state),
            failure_reason=failure_reason_for(state),
        )

        telemetry.capture_reporter_report(
            task_id=self.ctx.trace_id,
            outcome=outcome,
            summary_len=len(summary),
            has_evidence=bool(evidence),
            failed_count=len(failed_targets),
        )

        # 子目标级进度：这是方案 §11.6「steps_per_subgoal」指标的**唯一来源** ——
        # 步数上限（§4.3）只能靠它校准。Reporter 是唯一同时知道
        # 「这个子目标跑了多少步」和「结果如何」的地方。
        telemetry.capture_subgoal_progress(
            task_id=self.ctx.trace_id,
            subgoal_id=_current_subgoal_id(state),
            steps=state.executor_steps,
            outcome=outcome,
            retry_count=_retry_count(state),
        )

        return await state.asanitize_update(
            ctx=self.ctx,
            update={"executor_report": report},
        )

    async def _write_summary(
        self,
        *,
        state: State,
        outcome: str,
        log_text: str,
        failed_targets: list[FailedTarget],
    ) -> tuple[str, str | None]:
        """写 summary、挑 evidence。

        **默认不调 LLM**（`llm_free=True`）—— 见模块文档「省一次调用」那一节：
        事实字段本来就是代码直搬的，summary 只影响「给人看的措辞」，
        不影响正确性；而每个子目标省一次 LLM 调用，在长任务上是实打实的成本。

        需要更详细的总结时（`llm_free=False`）才调 LLM。
        """
        if self.llm_free:
            return (
                code_summary(
                    outcome=outcome,
                    failed_count=len(failed_targets),
                    failure_reason=failure_reason_for(state),
                ),
                pick_evidence(
                    None,
                    log_text=log_text,
                    outcome=outcome,
                    messages=state.executor_messages,
                    screen_text=visible_screen_text(state),
                ),
            )

        messages = build_reporter_messages(
            goal=state.executor_goal,
            outcome=outcome,
            log_text=log_text,
            failed_targets=failed_targets,
            language=self.ctx.prompt_language,
        )

        # 结构化输出按 **provider 能力**分派：llamacpp 走原生 json_schema 语法约束，
        # 其它 provider 保持原样（schema 当工具绑）—— 见 `get_structured_llm`。
        runner = get_structured_llm(ctx=self.ctx, name="reporter", schema=ReporterSummary)
        runner_fallback = get_structured_llm(
            ctx=self.ctx, name="reporter", schema=ReporterSummary, use_fallback=True
        )

        try:
            response = cast(
                "ReporterSummary",
                await with_fallback(
                    main_call=lambda: invoke_llm_with_timeout_message(
                        runner.ainvoke(messages),
                        agent_name="reporter",
                        task_id=self.ctx.trace_id,
                    ),
                    fallback_call=lambda: invoke_llm_with_timeout_message(
                        runner_fallback.ainvoke(messages),
                        agent_name="reporter",
                        task_id=self.ctx.trace_id,
                        used_fallback=True,
                    ),
                ),
            )
        except Exception as e:  # noqa: BLE001
            # 汇报挂掉不该让整条任务挂掉：事实层（outcome / 失败清单）还在，
            # 上层看到的是「没完成 + 失败清单」，比没有汇报强。
            logger.warning(f"[reporter] LLM call failed, falling back to a code summary: {e}")
            return (
                code_summary(
                    outcome=outcome,
                    failed_count=len(failed_targets),
                    failure_reason=failure_reason_for(state),
                ),
                None,
            )

        if response is None:
            return (
                code_summary(
                    outcome=outcome,
                    failed_count=len(failed_targets),
                    failure_reason=failure_reason_for(state),
                ),
                None,
            )

        summary = (response.summary or "").strip()
        if not summary:
            summary = code_summary(
                outcome=outcome,
                failed_count=len(failed_targets),
                failure_reason=failure_reason_for(state),
            )

        evidence = pick_evidence(
            response.evidence,
            log_text=log_text,
            outcome=outcome,
            messages=state.executor_messages,
            screen_text=visible_screen_text(state),
        )
        return summary, evidence


# ---------------------------------------------------------------------------
# 结论层：代码判定，LLM 不参与
# ---------------------------------------------------------------------------


def step_limit_reached(state: State) -> bool:
    """这次尝试有没有撞到步数上限。

    Executor 撞上限时写的是 ``executor_done=True``，那个 True 的意思是
    「别再动了，收尾汇报」，**不是**「目标达成了」
    （``agents/executor/executor.py`` 的 step_limit 分支）。
    所以这里不能只看 ``executor_done`` —— 否则一次撞上限的尝试会被判成
    ``completed``，Cortex 就会以为这个子目标做成了。

    ⚠️ **默认不限**（``MAX_STEPS_PER_SUBGOAL is None``）⇒ 这个判定恒为 False
    ⇒ ``outcome`` 不会是 ``blocked``。``blocked`` 这条分支留着，是为了
    「有人把上限配回来」时行为不变（代码路径都在，只是默认不触发）。
    """
    if MAX_STEPS_PER_SUBGOAL is None:
        return False
    return state.executor_steps > MAX_STEPS_PER_SUBGOAL


def determine_outcome_for(state: State) -> Literal["completed", "failed", "blocked"]:
    """代码判定 outcome（规则无歧义，见 :func:`determine_outcome`）。"""
    step_limit = step_limit_reached(state)
    # 撞了上限就不是「达成」—— 那个分支里 executor_done 只表示「收尾」。
    #
    # ⚠️ **模型明确放弃**（`executor_failed`）同样不是「达成」——
    # 它是「做不到」。在这之前 `done=True` 会被**无条件**映射成 completed，
    # 而 executor 当时又没有「做不到」这个出口 ⇒ 「做不到」不可表达
    # ⇒ 模型只能一直试 ⇒ 原地打转 ⇒ 撞 recursion_limit 崩掉
    # （真机实测 2026-10-26，mobile-manager execution 880）。
    done = bool(state.executor_done) and not step_limit and not state.executor_failed
    return determine_outcome(
        done=done,
        failures=state.executor_failures,
        step_limit_reached=step_limit,
        gave_up=bool(state.executor_failed),
    )


def failure_reason_for(state: State) -> str | None:
    """模型自己说的「做不到的原因」（**逐字**，不改写）。

    不是「放弃」就返回 None。

    ⚠️ 它是**事实层**：这句话是模型的原话，Cortex / Orchestrator 读到的
    就是它自己说的那句 —— 不重写、不美化（「把失败说成完成」的老毛病
    在这里没有生存空间）。
    """
    if not state.executor_failed:
        return None
    for message in reversed(state.executor_messages or []):
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None


def _current_subgoal_id(state: State) -> str:
    """当前子目标的 ID（给遥测用，不含目标原文）。

    拿不到就返回 ``"unknown"`` —— 遥测不能因为缺个 ID 就把埋点搞挂。
    """
    subgoal = get_current_subgoal(state.subgoal_plan)
    return subgoal.id if subgoal is not None else "unknown"


def _retry_count(state: State) -> int:
    """这个子目标被**重发**过几次（换招重来的次数）。

    ⚠️ **不能用「失败事实条数」来近似** —— 那是错的，而且错在最常见的地方：
    重发有三条路径，其中**两条根本不产生失败事实**：

    1. **规则换招**（L3）：outcome 是 failed/blocked 时，cortex 用规则重新
       下发同一个子目标。
    2. **撞步数上限**：outcome 是 blocked（不是 failed），没有任何失败目标。
    3. **cortex 判「还没成」**：汇报说 completed，但证据不足，cortex 重新下发。

    1 和 2 都不写失败事实 ⇒ 用失败条数算出来的 `retry_count` **恒为 0** ——
    那是个「看着有数其实是常数」的假指标（真机实测：3 个子目标实际重发了
    5 次，`retry_count` 全程报 0）。

    正确的口径是**「这个子目标被下发过几次」减一**：
    下发记录写在 ``agents_thoughts`` 里（append-only，只增不减），
    由 cortex 写、由 :mod:`agents.cortex.issue_log` 统一计数 ——
    不用往 State 里加字段。
    """
    subgoal_id = _current_subgoal_id(state)
    if subgoal_id == "unknown":
        return 0
    return max(0, count_goal_issues(state.agents_thoughts, subgoal_id) - 1)


def blocked_reason_for(state: State) -> str | None:
    """blocked 的原因（事实，代码给）。不是 blocked 就返回 None。"""
    if not step_limit_reached(state):
        return None
    return (
        f"step limit of {MAX_STEPS_PER_SUBGOAL} reached; "
        "the attempt was stopped before the goal was reached"
    )


# ---------------------------------------------------------------------------
# 事实层：从 state 直搬，逐字
# ---------------------------------------------------------------------------


def build_failed_targets(failures: list[dict] | None) -> list[FailedTarget]:
    """事实账本 → 汇报里的失败清单（**逐字**，不改写）。

    ``executor_failures`` 是 append-only 的失败事实（``utils/failed_targets.py``
    的 ``append_failure``），每一条都是**独立的一次失败事件**。所以 ``count``
    恒为 1 —— ``FailedTarget.count`` 的语义是「连着失败了几次」，和这里不是一回事，
    混用会读错。

    注意这里**不是** ``state.failed_targets``：那个是熔断账本，成功即清、
    换键即整表替换（方案 §5.2），跨子目标的事实留不住。
    """
    items: list[FailedTarget] = []
    for failure in failures or []:
        index = failure.get("index")
        resource_id = failure.get("resource_id")
        label = failure.get("label") or ""
        key = failure.get("key") or make_key(index, resource_id, label)
        items.append(
            FailedTarget(
                key=key,
                index=index,
                resource_id=resource_id,
                label=label,
                count=1,
                reason=failure.get("reason") or "",
            )
        )
    return items


def build_actions_taken(messages: list[AnyMessage] | None) -> list[str]:
    """从局部历史里抽出执行过的动作（每个一行）。

    只取 AIMessage 上的 tool_calls —— 工具写回的是 ToolMessage（结果），
    动作和结果在汇报里是分开的两层。
    """
    actions: list[str] = []
    for message in messages or []:
        if not isinstance(message, AIMessage):
            continue
        for call in getattr(message, "tool_calls", None) or []:
            actions.append(f"{call.get('name')}({format_args(call.get('args') or {})})")
    return actions


def format_args(args: dict) -> str:
    """动作参数压成一行（太长就截断 —— 汇报不是日志全文）。"""
    try:
        text = json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(args)
    return text if len(text) <= MAX_ARGS_LEN else text[:MAX_ARGS_LEN] + "…"


def render_execution_log(messages: list[AnyMessage] | None) -> str:
    """把局部历史渲染成给人（和模型）看的执行日志。

    动作行和结果行分开：Reporter 要从**结果**里挑证据（§6.2.2），
    所以工具结果必须原样出现，不能被压掉。

    ⚠️ **多模态结果只取文字。** 截图工具的结果是 ``[文字块, 图片块]``，
    ``str(content)`` 会把几十 KB 的 base64 变成一行 Python 字面量 —— 汇报正文
    里多出一坨噪声，reporter 的 token 也白烧。图对汇报没用（汇报要的是
    「看到了什么」的事实），所以这里只留文字（见 ``utils/message_content.py``）。
    """
    lines: list[str] = []
    step = 0
    for message in messages or []:
        if isinstance(message, AIMessage):
            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                for call in tool_calls:
                    step += 1
                    name = call.get("name")
                    args = format_args(call.get("args") or {})
                    lines.append(f"[{step}] {name}({args})")
            else:
                content = content_to_text(message.content).strip()
                if content:
                    lines.append(f"[note] {content}")
        elif isinstance(message, ToolMessage):
            content = content_to_text(message.content).strip()
            marker = "error" if getattr(message, "status", "") == "error" else "result"
            lines.append(f"    <- {marker}: {content}")
    return "\n".join(lines)


def render_failures(failed_targets: list[FailedTarget]) -> str:
    """失败清单渲染成给模型看的几行（逐字，不改写理由）。"""
    if not failed_targets:
        return ""
    return "\n".join(
        f"- {describe(item.index, item.resource_id, item.label)}: {item.reason}"
        for item in failed_targets
    )


# ---------------------------------------------------------------------------
# 证据：LLM 挑，代码校验
# ---------------------------------------------------------------------------


def pick_evidence(
    raw: str | None,
    *,
    log_text: str,
    outcome: str,
    messages: list[AnyMessage] | None,
    screen_text: str | None = None,
) -> str | None:
    """挑证据：LLM 挑的那条必须**真的在日志里**，否则换成代码挑的最后一条结果。

    「不要编造」是提示词里的**软**规则；这里再补一道**硬**检查 ——
    模型改写了一句日志里没有的话，那它就不是证据。核对不过就退到代码挑的结果
    （最近一条工具结果），至少它是真事实。

    ``outcome == "completed"`` 时证据不能空（§6.4.1 对策 A：Cortex 要靠它
    独立判断「这个子目标真的完成了吗」），所以兜底给最近一条工具结果。

    🔴 **但光有工具结果不够**（见 :func:`visible_screen_text` 的根因说明）：
    「已成功点击 WLAN」只说明「我点了」，不说明「我看到了 WLAN 页面」——
    Cortex 拿它判不了，于是重发。所以证据由**两段**组成：
    动作结果 + **屏幕上的可见文字**（``screen_text``，代码直搬的事实）。
    """
    candidate = (raw or "").strip()
    if candidate:
        if _contains(log_text, candidate):
            return compose_evidence(candidate, screen_text)
        logger.warning(
            "[reporter] The model-provided evidence is not in the execution log; "
            "using the code-picked result instead"
        )

    if outcome == "completed":
        return compose_evidence(last_tool_result(messages), screen_text)
    return screen_text


def last_tool_result(messages: list[AnyMessage] | None) -> str | None:
    """最近一条非空的工具结果（代码挑的证据兜底）。"""
    for message in reversed(messages or []):
        if isinstance(message, ToolMessage):
            content = content_to_text(message.content).strip()
            if content:
                return content[:MAX_EVIDENCE_LEN]
    return None


# ---------------------------------------------------------------------------
# 证据的**来源**：动作结果不够，要「屏幕上因此看到了什么」
# ---------------------------------------------------------------------------
#
# 🔴 **这是「为什么会重发」的根因**（真机 2026-10-21）。
#
# 判「这个子目标真的完成了吗」需要的是**屏幕状态**；而执行日志里只有
# **动作结果**：
#
#     tap   → 「已成功点击元素 [3] WLAN」   ← 这是「我点了」，不是「我看到了」
#     swipe → 「滑动成功」                  ← 零信息
#
# ⇒ 汇报里从来没有「屏幕上因此出现了什么」，**Cortex 判不了**，于是它选了
# 「还没成，再来一次」。真机日志里 Cortex 的原话：
#
#     「报告的证据只有『滑动成功』这类操作结果，summary 也是代码生成的，
#       没有任何 WLAN 页面标题、开关状态或网络名称的原文，因此无法确认
#       已停留在 WLAN 页面。」
#
# ⇒ 而「屏幕上看到了什么」**早就在 state 里了**：
# ``state.latest_screen_elements`` / ``state.latest_ocr_results``。
#
# ⚠️ **为什么它就是「动作执行之后」那一屏**（这点必须对，否则证据是旧的）：
# 环是 ``executor → executor_tools → summarizer → contextor → executor`` ——
# 每执行一个动作都会回到 contextor **重新抓屏**。所以 executor 说 done 的
# 那一刻，state 里躺着的就是**最后一个动作执行之后**的屏。
#
# ⇒ 修法：**零成本、零 LLM、零额外抓屏** —— 只是把已经在手边的文字搬进汇报。


# 证据里「屏幕可见文字」最多留多长。
#
# 🔴 **2026-10-24：去掉了「前 N 行」的截断**（原来是
# ``MAX_SCREEN_EVIDENCE_LINES = 25``，只取元素表的**头 25 行**）。
#
# 真机实测（mobile-manager execution 853，Redmi Turbo 4）：小米应用商城
# 「番茄畅听」详情页的元素表有 **152 行**，主按钮「打开」在**第 141 行**
# ⇒ 前 25 行里根本没有它 ⇒ 汇报给 Cortex 的证据里没有「打开」
# ⇒ Cortex 读不到「按钮显示打开」，**正确地拒绝**了这次汇报（它原话：
# "it never shows the detail-page button reading 打开/已安装… The 安装 text
# visible belongs to the recommended app 番茄音乐"）⇒ 改下一个**难十倍**的
# 子目标（去全部应用列表找图标）⇒ 执行器掉进 MIUI 看点浮层陷阱
# ⇒ 原地打转 16 分钟、67.9 万 token。
#
# ⚠️ 那一屏实测：84 个带文字的元素、去重后 75 条、**全量只有 827 字符** ——
# **字符预算（900）本来就装得下，砍掉的其实是行数。** ⇒ 按字符定预算就够。
# ���在留 1600 是给更长的页留余量。
MAX_SCREEN_EVIDENCE_CHARS = 1600


def _screen_texts(state: State) -> list[str]:
    """这一屏所有**看得见且不重复**的文字，按屏序（从上到下）。

    ⚠️ 去重按**原文**：「打开」和「打开 」是两条不同的视觉事实，合并掉就是丢事实。
    """
    lines: list[str] = []
    for element in state.latest_screen_elements or []:
        text = (element.label or element.text or "").strip()
        if text and text not in lines:
            lines.append(text)

    if not lines:
        for ocr in state.latest_ocr_results or []:
            text = (ocr.text or "").strip()
            if text and text not in lines:
                lines.append(text)

    return lines


def _join_within_budget(lines: list[str], budget: int, sep: str = " | ") -> str:
    """把若干段文字拼起来；超预算就**保首尾**、中段省略。

    ⚠️ **砍尾巴是错的**（2026-10-24，execution 853）：主操作按钮几乎总在**屏底**
    （那一屏的「打开」在第 141/152 行）。砍掉尾巴 = 证据里没有主按钮 =
    supervisor 核实不了 = 它拒绝并换一个更难的目标 = 执行器掉进死胡同。

    ⇒ 宁可中间少几行，也**必须保住尾部**，并且**明说**省略了多少条
    （不装成「全都看到了」）。

    Args:
        lines: 要拼的片段（按顺序）。
        budget: 字符预算。
        sep: 片段之间的分隔符。
    """
    joined = sep.join(lines)
    if len(joined) <= budget:
        return joined

    def take(source: list[str], from_end: bool, room: int) -> list[str]:
        picked: list[str] = []
        used = 0
        for line in reversed(source) if from_end else source:
            cost = len(line) + len(sep)
            if used + cost > room:
                break
            picked.append(line)
            used += cost
        return list(reversed(picked)) if from_end else picked

    room = max(budget - 40, 1)  # 给省略标记留位置
    head = take(lines, False, room // 2)
    tail = take([ln for ln in lines if ln not in head], True, room - room // 2)
    if not head and not tail:
        # 一条就超预算（极端情况）⇒ 只能硬切，但至少保住尾巴
        return joined[-budget:]

    omitted = len(lines) - len(head) - len(tail)
    marker = f"{sep}…（中间 {omitted} 条省略）…{sep}"
    return f"{sep.join(head)}{marker}{sep.join(tail)}"


def visible_screen_text(state: State) -> str | None:
    """最后一屏（最后一个动作执行之后）**看得见的文字** —— 事实，代码直搬。

    优先用元素表里的 ``label``（UI 节点的可见文字），没有就用 OCR 的 ``text``。
    两边都空才返回 None（那说明这一屏确实没文字可报，比如纯图标界面）。

    🔴 **取「屏上有什么」，不取「前 N 行」**（2026-10-24，execution 853）：
    按行数截会把**屏底的主操作按钮**砍掉 —— supervisor 因此核实不了、
    拒绝汇报、换一个更难的目标，执行器就掉进死循环。详见
    :data:`MAX_SCREEN_EVIDENCE_CHARS` 上面那段。
    """
    lines = _screen_texts(state)
    if not lines:
        return None
    return _join_within_budget(lines, MAX_SCREEN_EVIDENCE_CHARS)


def compose_evidence(action_result: str | None, screen_text: str | None) -> str | None:
    """证据 = 动作结果 + 屏幕上的可见文字。

    两段都要：**动作结果**说「我做了什么」，**屏幕文字**说「因此看到了什么」。
    只给前者，Cortex 判不了（见上面的根因）。

    ⚠️ 超预算时**只砍动作结果那截**，屏幕文字那段一个字节都不动 ——
    主按钮在屏底（execution 853），砍它等于把「按钮显示打开」从证据里抹掉。
    """
    parts = [p for p in (action_result, screen_text) if p]
    if not parts:
        return None
    joined = "\n".join(parts)
    if len(joined) <= MAX_EVIDENCE_LEN:
        return joined
    room = MAX_EVIDENCE_LEN - len(screen_text or "")
    head = (action_result or "")[: max(room, 0)]
    return "\n".join([head, screen_text or ""]).strip()


def _normalize(text: str) -> str:
    """归一化空白和大小写 —— 用来做「证据是不是日志里那句」的核对。"""
    return " ".join(text.split()).lower()


def _contains(haystack: str, needle: str) -> bool:
    return _normalize(needle) in _normalize(haystack)


# ---------------------------------------------------------------------------
# prompt 组装
# ---------------------------------------------------------------------------


@cache
def load_reporter_prompt(language: PromptLanguage = PromptLanguage.EN) -> str:
    """Reporter 的 system prompt（**完全静态**，没有模板变量）。

    静态 = 跨子目标逐字节不变 = 前缀缓存能一直命中。

    ``language`` 进缓存键：中文版是另一个文件（``reporter-zh.md``），
    两种语言各自的静态块要各自缓存。
    """
    return load_prompt_template(REPORTER_PROMPT_PATH, language).render()


def build_reporter_messages(
    *,
    goal: str | None,
    outcome: str,
    log_text: str,
    failed_targets: list[FailedTarget],
    language: PromptLanguage = PromptLanguage.EN,
) -> list[BaseMessage]:
    """组装 Reporter 的 prompt。

    顺序：system（静态）→ 一条 human（子目标 + 已定的结论 + 执行日志 + 失败清单）。

    汇报每个子目标只发生一次，prompt 很小，不需要像 Executor 那样分块。
    结论（outcome）**先给**：模型不该自己猜结论，它只需要写人话、挑证据。
    """
    parts = [
        f"{pick('Subgoal attempted:', '尝试的子目标：', language=language)}\n"
        f"{goal or pick('(no goal was given)', '(没有给出目标)', language=language)}",
        f"{pick('Outcome already determined by code:', '代码已定好的结论：', language=language)}"
        f" {outcome}",
        pick("Actions and their results:", "执行过的动作及结果：", language=language)
        + "\n"
        + (log_text or pick("(no action was executed)", "(没有执行任何动作)", language=language)),
    ]
    failure_text = render_failures(failed_targets)
    if failure_text:
        parts.append(
            pick(
                "Failed targets (facts, do not reword them):",
                "失败目标（是事实，别改写）：",
                language=language,
            )
            + "\n"
            + failure_text
        )

    return [
        SystemMessage(content=load_reporter_prompt(language)),
        HumanMessage(content="\n\n".join(parts)),
    ]


def code_summary(*, outcome: str, failed_count: int, failure_reason: str | None = None) -> str:
    """纯代码摘要（不调 LLM）。

    两种用法：
    * **默认路径**（`ReporterNode(llm_free=True)`）—— 省一次 LLM 调用。
    * **降级路径** —— LLM 调不通时的兜底（汇报失败不该让整条任务挂掉：
      事实层（outcome / 失败清单）还在，上层看到的是「没完成 + 失败清单」，
      比没有汇报强）。

    措辞上不区分这两种情况：上层（Cortex / Orchestrator）读的是
    ``outcome`` 和事实字段，``summary`` 只是给人看的说明。

    ⚠️ ``failure_reason`` 是**模型自己说的「做不到的原因」**（逐字）。
    带上它，是因为「模型主动放弃」和「模型没放弃只是没做成」在上层看来
    长得一模一样（都是 outcome=failed）—— 不说清楚，监督者只能重新下发
    同一个目标，白白再烧一遍。
    """
    if outcome == "completed":
        return "子目标已完成（证据见下方：动作结果 + 屏幕上看到的文字）。"
    if outcome == "blocked":
        limit = MAX_STEPS_PER_SUBGOAL
        if limit is None:
            # 默认不限 ⇒ blocked 本来不可达；这里是「有人把上限配回来了」的兜底。
            return "子目标未完成，见失败清单。"
        return f"子目标未完成：触到步数上限（{limit} 步），见失败清单。"
    if failure_reason:
        return f"子目标做不到（执行器主动放弃）：{failure_reason}"
    if failed_count:
        return f"子目标未完成，见失败清单（{failed_count} 条失败事实）。"
    return "子目标未完成，未记录到失败事实。"
