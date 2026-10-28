"""Cortex：监督者。

改造前 Cortex 是整条链路上最贵的一环：**每轮**看屏、**每轮**决定下一个动作。
它自己渲染元素表、自己读 OCR、自己要截图、自己产出 JSON 动作 —— prompt 里
还背着工具 schema 和元素表渲染规则。

Executor 自主执行之后，这活儿下沉给了 Executor（它自己看屏、自己决定下一个
动作、自己执行）。Cortex 退化成**监督者**：

* **不看屏** —— 感知通道全部在 Executor 侧（元素表 / OCR / 截图）。
* **不产出动作** —— 动作由 Executor 决定。
* **每个子目标只判断一次** —— 上一个子目标达成了吗？下一个子目标是什么？

它的输出因此只剩三个字段（``agents/cortex/types.py``）：
``next_subgoal`` / ``subgoal_completed`` / ``completion_reason``。

四条**不能忘**的机制

1. **局部历史的清空责任在这里。** ``executor_messages`` 是「本子目标内」的
   历史（Executor 的回合 + 工具反馈）。换了目标就必须清空 —— 否则 Executor 会
   带着上一个目标的上下文去执行新目标，它自己会以为那些动作已经做过了。
   写 ``executor_goal`` 的每一次（首次下发、换招重发）都要清。

2. **汇报是给 Cortex 的一次性输入。** 消费掉就清空：留着的话，下一次进
   Cortex 会把同一份汇报当成新证据再判一次，可能把已经验收过的子目标再
   提名一次。

3. **完成判定要留给 Orchestrator 验收。** Cortex 只说「我判定它完成了」，
   提名写进 ``complete_subgoals_by_ids``；真正把子目标标成 SUCCESS 的是
   Orchestrator（``orchestrator.py:112``）。

4. **失败路径不一定问你（L3）。** 失败之后的「换招」是**规则**：同一个子目标
   换个路径再来一次。规则做这件事能省掉一次 LLM 调用（3~5 秒 + 1,000+ token），
   而子目标常常只有 1~2 步，那次调用是纯开销。

   ⚠️ **但规则的质量不如你，而且次数有上限**（``RULE_RETRY_BUDGET``）：
   规则用过之后一定回到你。所以「放弃」这个判断仍然只有你能做 ——
   只有你知道这个子目标本身是不是根本走不通（功能不存在 / 没权限 / 数据不存在）。
"""

import os
from functools import cache
from pathlib import Path
from typing import Any, Literal, NamedTuple

from jinja2 import Template
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from minitap.mobile_use.agents.cortex.issue_log import (
    RULE_RETRY_MARKER,
    count_goal_issues,
    count_rule_retries,
)
from minitap.mobile_use.agents.cortex.types import CortexOutput
from minitap.mobile_use.agents.executor.types import ExecutorReport
from minitap.mobile_use.agents.planner.types import Subgoal
from minitap.mobile_use.agents.planner.utils import (
    fail_current_subgoal,
    get_current_subgoal,
    render_subgoal_status,
)
from minitap.mobile_use.constants import EXECUTOR_MESSAGES_KEY, EXECUTOR_SCREENSHOT_KEY
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.prompt_language import PromptLanguage, load_prompt_template
from minitap.mobile_use.services.llm import get_llm, invoke_llm_with_timeout_message, with_fallback
from minitap.mobile_use.services.telemetry import telemetry
from minitap.mobile_use.tools.index import (
    EXECUTOR_WRAPPERS_TOOLS,
    VIDEO_RECORDING_WRAPPERS,
    format_tools_list,
)
from minitap.mobile_use.utils.decorators import wrap_with_callbacks
from minitap.mobile_use.utils.failed_targets import describe, render_failures, render_failure_facts
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.time import format_task_started_at
from minitap.mobile_use.utils.thoughts_compaction import (
    PLANNER_AUDIENCE,
    render_thoughts_history,
)

logger = get_logger(__name__)

CORTEX_PROMPT_ENV = "MOBILE_USE_CORTEX_PROMPT"

CORTEX_PROMPT_PATH = Path(__file__).parent / "cortex.md"

# 模型偶尔会用这些占位串表示“没有内容”，统一当作 None 处理。
EMPTY_STRING_TOKENS = frozenset({"{}", "[]", "null", "None", ""})


def _normalize_optional_str(value: str | None) -> str | None:
    if value is None or value.strip() in EMPTY_STRING_TOKENS:
        return None
    return value


@cache
def load_cortex_template(
    language: PromptLanguage = PromptLanguage.EN,
    override: str | None = None,
) -> Template:
    """Cortex 的 system prompt 模板（已编译）。

    **两套切换机制，优先级是「环境变量 > 语言开关」**：

    1. ``MOBILE_USE_CORTEX_PROMPT`` 可以指向另一个 .md（例如同目录的
       cortex.zh.md），用来做中/英文提示词的 A/B 实测 —— 同一份 context
       分别渲染两种语言，再各自真跑一遍，比较 token 与成功率。
       **显式指定了路径就以它为准**（那是「我要精确用这个文件」），
       指针写错不该让整条任务挂掉：只 warning 并回落到默认模板。
    2. 没设这个变量时，按 ``language`` 取：``en`` → ``cortex.md``，
       ``zh`` → ``cortex.zh.md``（没有则回落英文 + warning）。

    不设任何开关时读的还是同目录的 cortex.md，行为与从前逐字节一致，
    所以对生产是 no-op。

    ⚠️ **缓存键是 ``(语言, 覆盖路径)``，不是模板正文**：正文有 7KB，
    拿它当键每轮都要哈希一遍；而覆盖路径必须进键，否则「换个环境变量
    换份提示词」就不生效了。
    """
    if override:
        path = Path(override).expanduser()
        try:
            if path.is_file():
                return Template(path.read_text(encoding="utf-8"))
            logger.warning(
                f"[cortex] {CORTEX_PROMPT_ENV}={override!r} 不是可读文件，回落到默认 cortex.md"
            )
        except OSError as e:
            logger.warning(
                f"[cortex] 读取 {CORTEX_PROMPT_ENV}={override!r} 失败（{e}），回落到 cortex.md"
            )
    return load_prompt_template(CORTEX_PROMPT_PATH, language)


async def _invoke_cortex(
    llm: Runnable[Any, Any],
    messages: list[BaseMessage],
    ctx: MobileUseContext,
    used_fallback: bool = False,
) -> CortexOutput:
    """调用 LLM 并校验输出。

    空响应（既没有下一个子目标、也没有完成/放弃的理由）无法推进任务，直接抛错
    让 with_fallback 切到备用模型，避免白耗 step 直到 recursion_limit。
    """
    response = await invoke_llm_with_timeout_message(
        llm.ainvoke(messages),
        agent_name="cortex",
        task_id=ctx.trace_id,
        used_fallback=used_fallback,
    )
    if response is None:
        raise ValueError("Cortex LLM returned no response")

    response.next_subgoal = _normalize_optional_str(response.next_subgoal)
    response.completion_reason = _normalize_optional_str(response.completion_reason)

    if (
        response.next_subgoal is None
        and response.completion_reason is None
        and not response.subgoal_completed
    ):
        raise ValueError("Cortex LLM returned an empty response")
    return response


def render_executor_report(report: ExecutorReport | None) -> str:
    """把 Executor 的汇报渲染成给 Cortex 看的一块。

    分两层（方案 §6.3）：结论层（outcome + summary）和事实层
    （evidence / actions_taken / failed_targets / blocked_reason）。

    事实层**逐字搬运**，不重写：Reporter 只压缩散文，字段本身是代码搬过来的
    （``agents/executor/types.py``），Cortex 看到的就是真的发生了什么。
    """
    if report is None:
        return "Executor report: none yet - the Executor has not finished a subgoal attempt yet."

    lines = [
        "Executor report (last subgoal attempt):",
        f"outcome: {report.outcome}",
        f"summary: {report.summary or '(none)'}",
    ]
    if report.evidence:
        lines.append(f"evidence: {report.evidence}")
    if report.actions_taken:
        lines.append("actions taken:")
        lines.extend(f"- {action}" for action in report.actions_taken)
    if report.failed_targets:
        lines.append("failed targets:")
        lines.extend(
            f"- {describe(t.index, t.resource_id, t.label)} (failed {t.count}x) {t.reason}".rstrip()
            for t in report.failed_targets
        )
    if report.blocked_reason:
        lines.append(f"blocked reason: {report.blocked_reason}")
    if report.failure_reason:
        # 「做不到」的原因 —— **逐字**。这是「模型主动放弃」和「没做成」
        # 在上层看来唯一的区别，不说清楚，监督者只能把同一个目标再下发一遍。
        lines.append(f"failure reason: {report.failure_reason}")
    return "\n".join(lines)


class CortexDecision(NamedTuple):
    """Cortex 的一次决定。

    ``action`` 三选一：

    * ``issue`` —— 下发目标（首次下发，或失败后**换招重发**）
    * ``complete`` —— 上一个子目标达成了（提名给 Orchestrator 验收）
    * ``give_up`` —— 换招也没招了：这个子目标本身走不通（标记失败）
    """

    action: Literal["issue", "complete", "give_up"]
    goal: str | None
    reason: str


def decide_next_step(
    *,
    response: CortexOutput,
    report: ExecutorReport | None,
    current_subgoal: Subgoal | None,
) -> CortexDecision:
    """把 Cortex 的输出翻译成「下发什么 / 判完成 / 放弃」。

    纯逻辑：判断和 IO 分开，才能单测（不接设备、不调 LLM；只写日志）。

    编码约定（三个字段怎么组合成三种动作）：

    * 有汇报 + ``subgoal_completed`` → **complete**。
      没有汇报时这个字段一律忽略 —— 没人汇报过，就没有东西可以「完成」。
    * 有汇报 + 没判完成 + 给了 ``next_subgoal`` → **issue**（换招重发：
      换个思路达成**同一个**子目标）。
    * 有汇报 + 没判完成 + **没给** ``next_subgoal`` → **give_up**：
      模型认为换招也没招了 → 标记该子目标失败，让 Orchestrator 重新看计划。
      这一条是「不能无限重发」的兜底（方案 T8'）。
    * 没有汇报 → **issue**（第一次进 Cortex，下发第一个子目标）。
    """
    reason = response.completion_reason or (report.summary if report else "")

    if report is None:
        # 第一次进 Cortex：还没有任何尝试可供判断，只可能是「下发目标」。
        if response.subgoal_completed:
            logger.warning(
                "[cortex] Model claimed a completed subgoal but there is no "
                "executor report; ignoring the claim"
            )
        goal = response.next_subgoal
        if goal:
            return CortexDecision("issue", goal, reason)
        # 模型没给目标陈述：计划里就有当前子目标的描述，拿它兜底，
        # 总比让 Executor 空转（没有目标 = 什么都不做）强。
        if current_subgoal is not None:
            logger.warning(
                "[cortex] Model returned no goal statement; falling back to the "
                "active subgoal's description"
            )
            return CortexDecision("issue", current_subgoal.description, reason)
        raise ValueError("Cortex returned no goal statement and the plan has no active subgoal")

    if response.subgoal_completed:
        return CortexDecision("complete", None, reason)

    # 失败 / 阻塞：**默认换招重发** —— 同一个子目标，换个思路再来一次。
    if response.next_subgoal:
        return CortexDecision("issue", response.next_subgoal, reason)

    return CortexDecision("give_up", None, reason)


# ---------------------------------------------------------------------------
# L3：失败时的**规则化换招**（不调 LLM）
# ---------------------------------------------------------------------------
#
# 为什么失败路径可以不问 LLM
# --------------------------
# 失败之后的「换招」在语义上是个**规则**：同一个子目标，换一条路径再来一次。
# 规则完全能表达这件事（别重复上次的动作 / 换个入口 / 换个顺序 / 拆成更小的一步），
# 而 LLM 在这里要花 3~5 秒 + 1,000+ token，产出的大体是同一句话。
#
# 真机 A/B 实测（方案 §11.2.1）：每个子目标固定 4 次 LLM 调用
# （cortex 2 + reporter 1 + orchestrator 1），而子目标内常常只有 1~2 步 ——
# 失败路径上那次 cortex 调用是纯开销。
#
# ⚠️ **规则换招的质量肯定不如 LLM 生成的**。所以它有**明确的次数上限**：
# 用完 ``RULE_RETRY_BUDGET`` 次之后一律退回 LLM（LLM 还能判「放弃」）。
# 绝不能变成「永远不问 LLM」—— 那样 Cortex 的判断能力就没了，
# 「这个子目标本身是不是走不通」这个判断没人做了。

# 每个子目标允许用几次「规则换招」。用完就退回 LLM 换招。
RULE_RETRY_BUDGET = 1

# 「下发目标」这条 thought 的格式与计数原语放在 `issue_log` —— reporter 也要读
# 它（数「这个子目标被重发过几次」，上报 retry_count），放在这里会让 reporter
# 反向依赖整个 cortex 节点模块。
#
# 下面三个名字**继续从本模块导出**：既有调用点（L3 的规则换招、测试）都从
# `agents.cortex.cortex` 导入，改路径会波及它们。
__all__ = [
    "RULE_RETRY_BUDGET",
    "RULE_RETRY_MARKER",
    "count_goal_issues",
    "count_rule_retries",
]


def should_use_rule_retry(
    *,
    report: ExecutorReport | None,
    subgoal_id: str,
    thoughts: list[str],
) -> bool:
    """这次失败要不要走**规则换招**（而不是问 LLM）。

    条件（都要满足）：

    1. 有汇报，且 outcome 是 ``failed`` / ``blocked`` —— 只有失败路径才换招。
       「可能做成了」必须问 LLM 判证据（那才是真需要智能的地方）。
    2. 有当前子目标（没有描述就没法「换个路径做同一个事」）。
    3. 规则换招的次数还没用完。
    """
    if report is None or report.outcome not in ("failed", "blocked"):
        return False
    if not subgoal_id or subgoal_id == "(none)":
        return False
    return count_rule_retries(thoughts, subgoal_id) < RULE_RETRY_BUDGET


def build_rule_retry_goal(
    *,
    report: ExecutorReport,
    current_subgoal: Subgoal | None,
) -> str:
    """规则化的「换招」目标陈述。

    三段式：

    1. **目标** —— 原样用计划里的子目标描述。换招换的是**路径**，不是目标本身。
    2. **上次失败的事实** —— 试过什么 / 哪些目标已经失败 / 为什么被挡。
       逐字搬自 ``ExecutorReport``（那是代码直搬的），不重写、不脑补。
    3. **换招指令** —— 换一条路径，别重复上次的动作。
    """
    goal = (current_subgoal.description if current_subgoal else "").strip()
    if not goal:
        goal = report.summary.strip() or "Retry the subgoal."

    facts: list[str] = []
    if report.actions_taken:
        facts.append("already tried: " + "; ".join(report.actions_taken[-3:]))
    if report.failed_targets:
        facts.append(
            "targets that already failed: "
            + "; ".join(
                f"{describe(t.index, t.resource_id, t.label)} ({t.reason})".strip()
                for t in report.failed_targets
            )
        )
    if report.blocked_reason:
        facts.append(report.blocked_reason)
    if report.summary:
        facts.append(report.summary)

    lines = [goal, ""]
    if facts:
        lines.append("[previous attempt failed] " + "; ".join(facts))
    lines.append(
        "[retry] Take a different route this time - do not repeat the actions "
        "listed above. Try another entry point, another order, or a smaller "
        "first step."
    )
    return "\n".join(lines)


def decide_next_step_by_rule(
    *,
    report: ExecutorReport,
    current_subgoal: Subgoal | None,
) -> CortexDecision:
    """规则化换招的决策（纯函数，不调 LLM）。

    只会产出 ``issue``：换招 = 同一个子目标换个路径再来一次。

    「放弃」（``give_up``）是 LLM 的判断 —— 只有它知道这个子目标本身是不是
    根本走不通（功能不存在 / 没权限 / 数据不存在）。规则不替它做这个决定，
    所以规则换招的次数用完之后**一定**回到 LLM。
    """
    goal = build_rule_retry_goal(report=report, current_subgoal=current_subgoal)
    return CortexDecision("issue", goal, f"{RULE_RETRY_MARKER} (previous attempt failed)")


class CortexNode:
    def __init__(self, ctx: MobileUseContext):
        self.ctx = ctx

    @wrap_with_callbacks(
        before=lambda: logger.info("Starting Cortex Agent..."),
        on_success=lambda _: logger.success("Cortex Agent"),
        on_failure=lambda _: logger.error("Cortex Agent"),
    )
    async def __call__(self, state: State) -> dict[str, Any]:
        current_locked_app_package = (
            self.ctx.execution_setup.get_locked_app_package() if self.ctx.execution_setup else None
        )

        executor_wrappers = list(EXECUTOR_WRAPPERS_TOOLS)
        if self.ctx.video_recording_enabled:
            executor_wrappers.extend(VIDEO_RECORDING_WRAPPERS)

        # system prompt 整段静态：不再接 initial_goal / 计划 / 汇报 这类每轮
        # 变化的值。它因此是一个整体可复用的前缀，不会被任何单轮字段击穿。
        system_message = load_cortex_template(
            self.ctx.prompt_language,
            os.getenv(CORTEX_PROMPT_ENV),
        ).render(
            platform=self.ctx.device.mobile_platform.value,
            executor_tools_list=format_tools_list(ctx=self.ctx, wrappers=executor_wrappers),
            locked_app_package=current_locked_app_package,
            # perception 仍然传着（模板变量登记在 scripts/check_prompt_templates.py），
            # 但新 cortex.md 不再看屏，所以不再引用它 —— 感知在 Executor 侧。
            perception=self.ctx.perception,
        )

        # 消息按「首次变化时刻」从早到晚排序：恒定 → 只追加 → 每轮变。
        # 越靠前的块越稳定，能被前缀缓存覆盖的公共前缀就越长。
        messages: list[BaseMessage] = [
            SystemMessage(content=system_message),
        ]

        # 任务开始时间：系统提示词**之后**，单独一条消息。
        #
        # 为什么单独一条而不是塞进系统提示词：系统提示词必须**逐字节稳定**
        # （整段请求里最大的一块）。塞进去 ⇒ 变成「每任务唯一」⇒ 同一 persona
        # 的不同任务之间不再共享那段缓存。
        #
        # 为什么这条消息本身是安全的：任务开始时间**整场运行不变** ⇒ 它和它
        # 前面的系统提示词一样是前缀的一部分，永远命中。
        #
        # （文案硬编码在这里，与本文件既有的做法一致 —— 本文件的运行时消息
        #  一直是英文的，模板只负责系统提示词。）
        task_started_at = format_task_started_at(self.ctx.task_started_at)
        if task_started_at:
            messages.append(HumanMessage(content=f"Task started at: {task_started_at}"))

        messages.append(
            # 任务内恒定：设备基本信息、目标。这块逐字节不变，永远是前缀的一部分。
            HumanMessage(
                content="\n".join(
                    [
                        "Here are my device info:",
                        self.ctx.device.to_str().rstrip(),
                        f"Initial goal: {state.initial_goal}",
                    ]
                )
            )
        )

        # 这些 thought 绝不能塞成 assistant 回合。state.py:113 给每条加了 "[role] "
        # 前缀，模型会把它们当成 few-shot 照抄，于是无视 tool_choice 改用纯文本回答，
        # _parse() 再把这段中文当 JSON 解析 -> pydantic json_invalid -> 整个 goal 挂掉。
        # 压成单个 user 回合后，MiniMax-M3 的 tool_calls 实测恢复 2/2。
        # 见 v/probe_cortex_toolcall_ab.py、v/probe_cortex_history_len.py。
        #
        # 压缩：只压发给 LLM 的这份，state 里的原文不动（trace / TUI / steps_count /
        # outputter 都要全量）。压的是 cortex 的散文，executor 工具结果和
        # orchestrator 里程碑是**事实**，逐字保留。
        #
        # 关键：压缩是**位置无关**的（每条按同一条规则渲染），所以已经发出去的
        # 字节永不改变 -> 追加语义保留 -> 前缀缓存不因压缩而击穿。
        # 见 utils/thoughts_compaction.py 与 doc/thoughts-compaction-plan.md。
        thoughts = state.agents_thoughts
        if thoughts:
            compaction_enabled = self.ctx.perception.thoughts_compaction
            body = render_thoughts_history(
                thoughts,
                enabled=compaction_enabled,
                audience=PLANNER_AUDIENCE,
            )
            header = "Agent thoughts history (newest last):"
            # 头注说清**这里有什么** —— 模型不该以为看到的是全量流水。
            # （文案硬编码在这里，与本文件既有的做法一致。）
            notes: list[str] = []
            if compaction_enabled:
                notes.append(
                    "Older cortex reasoning is compressed to its first and last sentence."
                )
            notes.append(
                "Planning-level entries only: my own reasoning, orchestrator milestones, "
                "and Executor failures. Successful Executor steps are omitted — the "
                "Executor report tells you what actually happened."
            )
            header += "\n(" + " ".join(notes) + ")"
            messages.append(HumanMessage(content=header + "\n" + body))

        # 计划日志（子目标的 ID + 描述）排在 thoughts **之后**。
        #
        # 为什么不能放在 thoughts 前面
        # --------------------------
        # 计划日志是**只追加**的：replan 时尾部追加一段 "PLAN vN"，旧段一字不改。
        # 但「只追加」不等于「不击穿」—— 只要它排在 thoughts 前面，追加点之后的
        # 全部重算，而 thoughts 恰恰在它后面。
        #
        # 实测（2026-10-21，MiniMax M3.1-Flash 真 API，122 条 thoughts）：
        #   计划日志在 thoughts 前 -> replan 当轮 cached 2,329，满价重算 8,261
        #   计划日志在 thoughts 后 -> replan 当轮 cached 10,411，满价重算 183
        # 一次 replan 差 8,078 token，而这个差随 thoughts 变长而变大。
        #
        # 挪到 thoughts 之后的代价：thoughts 每轮增长，断点落在 thoughts 末尾，
        # 于是「计划日志 + 尾部块」每轮要满价重算一次（约 150 + 500 token）。
        # 摊到一次 replan 上（省 8,000+）是划算的，而且 replan 越多越划算。
        #
        # 语义没变：内容与之前逐字相同，只是从 head 块里拆出来独立成一条消息，
        # 「最后一段 PLAN 是当前计划」这句话依然成立。
        messages.append(
            HumanMessage(
                content="Subgoal plan (append-only history; "
                "the last PLAN section is the active one):\n"
                + (state.subgoal_plan_log or "(no plan)")
            )
        )

        # 完成情况是**索引**、且排在 thoughts 之后：它每轮都可能变（子目标完成 /
        # 推进 / replan），但按**计划序**排列、在固定槽位上原地替换，不会有新行插到
        # 行中间。放在 thoughts 后面，它怎么变都不会碰到 thoughts 那段唯一增长的前缀。
        # 固定成一条消息（无子目标时也发，值填 none），条数漂移本身也是前缀的一部分。
        messages.append(
            HumanMessage(content="Subgoal status: " + render_subgoal_status(state.subgoal_plan))
        )

        # 已失败的目标：模型下一轮最需要知道的一件事。
        #
        # 真机实测里，模型看到「目标不在屏上了」会误诊成「索引指错人了」，
        # 然后在同一个目标上反复打转。这条把事实摆出来：这些目标已经没了，
        # 别再用，换一个现在还在的元素。
        #
        # 固定发一条（空账本填 none）：条数本身也是前缀的一部分，偶尔才发会让
        # 后面所有块的位置漂移。
        messages.append(
            HumanMessage(
                content=render_failures(state.failed_targets) or "No target has failed recently."
            )
        )

        # Executor 的汇报：每子目标变一次，必须排在 thoughts 之后（它是「每轮变」
        # 的那一档）。汇报**不进** agents_thoughts 的散文通道 —— 那是 Cortex 自己
        # 的推理历史，Executor 的报告是外部事实，走独立的消息块（方案 §6.3）。
        messages.append(HumanMessage(content=render_executor_report(state.executor_report)))

        # 跨子目标的失败事实：只增不减，所以排在最后也不影响前缀稳定性。
        messages.append(
            HumanMessage(
                content=render_failure_facts(state.executor_failures)
                or "No failed target facts recorded yet."
            )
        )

        report = state.executor_report
        current_subgoal = get_current_subgoal(state.subgoal_plan)
        subgoal_id = current_subgoal.id if current_subgoal else "(none)"

        # ---- L3：失败路径先用**规则**换招，省掉一次 LLM 调用 ----
        #
        # 只有「可能做成了」才问 LLM（判证据需要智能）；失败时「换个路径再来」
        # 是个规则，规则的质量不如 LLM，所以有次数上限（见上面的说明）。
        #
        # ⚠️ LLM 客户端也一起跳过：规则路径连 cortex 的模型配置都不需要，
        # 少建一个用不上的客户端，也少一条「模型不可用就整个节点挂掉」的路。
        if report is not None and should_use_rule_retry(
            report=report,
            subgoal_id=subgoal_id,
            thoughts=state.agents_thoughts,
        ):
            decision = decide_next_step_by_rule(
                report=report,
                current_subgoal=current_subgoal,
            )
            decision_source = "rule"
            logger.info(
                f"[cortex] Subgoal {subgoal_id} did not succeed; "
                "re-issuing it with the rule-based retry (no LLM call)"
            )
        else:
            llm = get_llm(ctx=self.ctx, name="cortex", temperature=1).with_structured_output(
                CortexOutput
            )
            llm_fallback = get_llm(
                ctx=self.ctx, name="cortex", use_fallback=True, temperature=1
            ).with_structured_output(CortexOutput)

            async def ask_cortex() -> CortexOutput:
                return await with_fallback(
                    main_call=lambda: _invoke_cortex(llm, messages, self.ctx),
                    fallback_call=lambda: _invoke_cortex(
                        llm_fallback, messages, self.ctx, used_fallback=True
                    ),
                )

            async def ask_cortex_with_nudge() -> CortexOutput:
                """带 nudge 再问一次 —— 「你的输出读不出来，必须调用工具」。

                nudge 的内容和 executor 那份同款（`executor.py` 的「解析失败 →
                带 nudge 重试」），只是把工具名换成 CortexOutput、字段换成
                CortexOutput 的那几个。
                """
                nudge = HumanMessage(
                    content=(
                        "Your last answer could not be read. You must call the "
                        "CortexOutput tool with the fields (next_subgoal, "
                        "completion_reason, subgoal_completed). Do not write the "
                        "output as plain text — a plain-text answer is discarded."
                    )
                )
                return await with_fallback(
                    main_call=lambda: _invoke_cortex(llm, [*messages, nudge], self.ctx),
                    fallback_call=lambda: _invoke_cortex(
                        llm_fallback, [*messages, nudge], self.ctx, used_fallback=True
                    ),
                )

            # 🔴 **韧性**：解析失败不许杀死整个任务。
            #
            # 真机证据（2026-10-21，manager 链路）：MiniMax 偶发返回
            # 「finish_reason='tool_calls' 但内容是空的」⇒ `_parse` 抛
            # ValueError ⇒ **整个任务当场失败**（execution 直接 FAILED）。
            #
            # 为什么这很糟：
            # * `with_fallback` 在这里**救不了** —— 本机 override 里 fallback 和主模型
            #   是同一个模型，重试结果一样（`executor._invoke` 的注释里也写了这条）。
            # * 而 **executor 有三层韧性**（捕获异常 → 带 nudge 重试 → 降级成空动作），
            #   **cortex 却没有** ⇒ 同一个 provider 抖动，executor 扛得住、cortex 扛不住。
            #
            # ⇒ 照 executor 的形状补上：失败 → 带 nudge 重试一次 → 仍失败 →
            # **降级成「重发当前子目标」**。
            #
            # 为什么降级选「重发」而不是「判完成」或「放弃」：
            # * 判完成 = **误报成功**（事实层不能这么干）
            # * 放弃 = 一次 provider 抖动就否掉整个子目标，太狠
            # * 重发 = 「这个子目标再来一次」，**不引入新的事实断言**，
            #   而且和 §4.1 的契约一致（重发仍在同一个子目标的生命周期内）
            try:
                response: CortexOutput = await ask_cortex()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"[cortex] LLM call failed ({type(e).__name__}: {str(e)[:160]}); "
                    "retrying once with a nudge"
                )
                try:
                    response = await ask_cortex_with_nudge()
                except Exception as e2:  # noqa: BLE001
                    logger.warning(
                        f"[cortex] retry also failed ({type(e2).__name__}: {str(e2)[:160]}); "
                        "degrading to re-issuing the current subgoal"
                    )
                    if current_subgoal is None:
                        # 连「当前子目标」都没有 → 真的没法推进，只能放弃。
                        # 这条极罕见（计划里至少有一个子目标才会进 Cortex）。
                        raise
                    decision = CortexDecision(
                        "issue",
                        current_subgoal.description,
                        "cortex response could not be parsed; re-issuing the same subgoal",
                    )
                    decision_source = "degraded"
                else:
                    decision = decide_next_step(
                        response=response,
                        report=report,
                        current_subgoal=current_subgoal,
                    )
                    decision_source = "llm"
            else:
                decision = decide_next_step(
                    response=response,
                    report=report,
                    current_subgoal=current_subgoal,
                )
                decision_source = "llm"

        update: dict[str, Any] = {
            # latest_ui_hierarchy 没有索引契约，清掉保持工具侧现取
            # （contextor 建完元素表后也清；这里再清一次是兜底）。
            "latest_ui_hierarchy": None,
            # 汇报是给 Cortex 的**一次性**输入：消费掉就清空。留着的话下一次进
            # Cortex 会把同一份汇报当成新证据再判一次 —— 最坏的情况是把一个已经
            # 验收过的子目标再提名一次，计划就凭空跳过了一步。
            "executor_report": None,
        }
        agent_thoughts: list[str] = []

        if decision.action == "issue":
            assert decision.goal is not None  # decide_next_step 保证
            update["executor_goal"] = decision.goal
            # 🔴 **同目标重发不清局部历史**（契约 §4.1：局部历史只在
            # 「Cortex 下发本次任务」这个生命周期内 —— 换目标才丢）。
            #
            # 判据：**有没有汇报**。
            # * `report is None` ⇒ 这是**首次下发**（或上一个子目标已完成、
            #   要换目标了）⇒ 该清：Executor 不能带着上一个目标的上下文去
            #   执行新目标（它自己会以为那些动作已经做过了）。
            # * `report is not None` ⇒ **同一个子目标的换招重发**
            #   （失败/阻塞后换个思路再来一次）⇒ **不该清**。
            #
            # 旧实现**无条件清** ⇒ 重发时执行器忘了自己试过什么 ⇒ 在原地打转。
            # 真机证据（2026-10-21 第六轮）：x39h9m 的尝试序列是
            # `[(2, completed), (13, blocked), (1, completed)]` —— 第一次 2 步就
            # 做成了，重发后那次却瞎转 13 步，重发再重发又 1 步就成了。
            #
            # ⚠️ 保留历史会让它随重发次数增长。**增长有界的唯一保证是全局动作步
            # 预算**（`constants.DEFAULT_MAX_STEPS` → LangGraph 的
            # `recursion_limit`）—— 单子目标步数上限**默认不限**
            # （`MAX_STEPS_PER_SUBGOAL = None`，2026-10-22 起），所以这里
            # **没有**「每次尝试的步数上限」这层保护了。
            # （不做截断 —— 截断历史有把 AIMessage/tool_call 和 ToolMessage
            # 配对拆散的风险，那会变成「孤儿 tool_call → 400」，
            # 和致命 bug 3 同一类。）
            if report is None:
                update[EXECUTOR_MESSAGES_KEY] = [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
                # 当屏截图也一起清掉：换目标 ⇒ executor 失忆 ⇒ 上一张图对应的是
                # **上一个目标**的屏幕，留着会误导模型（「我看到的是这个」早就不是了）。
                update[EXECUTOR_SCREENSHOT_KEY] = None
            update["executor_steps"] = 0
            # 上一次的「做不到」标记要清掉 —— 它是**本子目标内**的信号，
            # 换目标之后它不成立（留着会让下一个子目标的汇报也判成 failed）。
            update["executor_failed"] = False
            # 进展指纹是「本子目标内」的，换目标就重新开始数
            # （和 `executor_messages` 同一个生命周期）。
            update["executor_progress"] = []
            # 没判完成就不该有提名：上一次尝试的提名（如果还没被 Orchestrator 消费）
            # 在这里作废，否则 Orchestrator 会验收一个 Cortex 没判完成的子目标。
            update["complete_subgoals_by_ids"] = []
            agent_thoughts.append(
                f"Issued goal for subgoal {subgoal_id}: {decision.goal}"
                + (f" ({decision.reason})" if decision.reason else "")
            )
        elif decision.action == "complete":
            # 提名给 Orchestrator 验收：真正把子目标标成 SUCCESS 的是它
            # （orchestrator.py:112），Cortex 只说「我判定它完成了」。
            update["complete_subgoals_by_ids"] = [subgoal_id] if current_subgoal else []
            agent_thoughts.append(
                f"Subgoal {subgoal_id} judged COMPLETE"
                + (f": {decision.reason}" if decision.reason else "")
            )
        else:
            # 换招也没招了：这个子目标本身走不通 → 标记失败。
            # 这一步是「计划走不通」唯一的处理者（方案 §6.5.1 的第三层）：
            # 标记成 FAILURE 之后 convergence 会判 one_of_them_is_failure → replan。
            if current_subgoal is not None:
                update["subgoal_plan"] = fail_current_subgoal(state.subgoal_plan)
            update["complete_subgoals_by_ids"] = []
            agent_thoughts.append(
                f"Subgoal {subgoal_id} given up"
                + (f": {decision.reason}" if decision.reason else "")
            )

        # Capture cortex decision telemetry (only non-sensitive flags)
        telemetry.capture_cortex_decision(
            task_id=self.ctx.trace_id,
            has_subgoal=decision.action == "issue",
            subgoal_completed=decision.action == "complete",
            completed_subgoals_count=len(update.get("complete_subgoals_by_ids", [])),
            # L3：这次决定是「模型判的」还是「规则换招」。两条路径的判断力不同，
            # 看板上必须能分开看，否则「cortex 判完成」的样本会被规则换招稀释。
            decision_source=decision_source,
        )

        if agent_thoughts:
            update["agents_thoughts"] = agent_thoughts

        return await state.asanitize_update(
            ctx=self.ctx,
            update=update,
            agent="cortex",
        )
