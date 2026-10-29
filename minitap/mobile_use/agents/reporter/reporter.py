"""汇报构建：把一次子目标尝试的执行日志压成一份 ``ExecutorReport``。

2026-10-29 起它**不是图节点**，而是一段纯代码，由 Cortex 在入口调用：
Executor 结束一次尝试（``executor_done=True``）之后，图直接回到 Cortex，
Cortex 先用这里构建汇报，再拿它做完成 / 换招 / 放弃的判断。

为什么不再是节点：

* 汇报的**唯一消费者**就是 Cortex。隔一个节点 = 每个子目标多一个 super-step、
  多一次「读 state → 写 state」的往返，而没有任何别处用到这份汇报。
* 汇报本来就免 LLM：``outcome`` 由代码判、事实字段由代码直搬、``summary``
  由代码生成、``evidence`` 由代码挑。节点里那套「可选 LLM 摘要」在生产里
  从来没有被打开过，删掉它之后 ``reporter`` 这个 LLM 角色也就不再需要。

汇报仍然分两层，理由不变。

结论层是 ``outcome``（做成了吗）、``summary``（人话）和 ``evidence``（证据）。
事实层是 ``actions_taken``、``failed_targets`` 和 ``blocked_reason``。

分层的意义是让「事实不丢」有结构性保证，而不是靠提示词写得好。

- ``outcome`` 由 :func:`determine_outcome` 判定，是纯代码的结果，不经 LLM。
  这里没有机会把失败说成完成。
- ``actions_taken``、``failed_targets``、``blocked_reason`` 从 state 里直搬，
  **逐字保留**。依据是 ``utils/thoughts_compaction.py`` 的硬规则「事实不能压」：
  失败事实是 Cortex 判断「换招还是放弃」的依据，压没了它就会重复派发同一个
  子目标。
"""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

from minitap.mobile_use.agents.cortex.issue_log import count_goal_issues
from minitap.mobile_use.agents.executor.executor import MAX_STEPS_PER_SUBGOAL
from minitap.mobile_use.agents.executor.types import ExecutorReport, determine_outcome
from minitap.mobile_use.agents.planner.utils import get_current_subgoal
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.services.telemetry import telemetry
from minitap.mobile_use.utils.failed_targets import FailedTarget, make_key
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.message_content import content_to_text

logger = get_logger(__name__)

# 证据最多留多长：工具结果里可能有整屏文本，太长会把 Cortex 的 prompt 撑大。
#
# 证据由两段组成，动作结果加屏幕上的可见文字。屏幕那一段最多 1600 字符
# （MAX_SCREEN_EVIDENCE_CHARS），动作结果那截也得留得下，所以上限按两段之和算，
# 超了就砍动作结果，不砍屏幕文字（见 compose_evidence）。
#
# Cortex 的 prompt 量级在 1,341 tok 左右，多出几百 tok 完全扛得住。
MAX_EVIDENCE_LEN = 2400

# 动作参数在汇报里最多留多长，理由同上。
MAX_ARGS_LEN = 200


def build_executor_report(state: State, ctx: MobileUseContext) -> ExecutorReport:
    """构建一次尝试的汇报（纯代码，不调 LLM）。

    Cortex 只在 ``executor_done`` 为真时调用它 —— 那时候 state 里的
    ``executor_messages``、``executor_failures``、``latest_screen_elements``
    正好是这次尝试的完整事实，不需要额外抓屏，也不需要额外调用。

    遥测在这里上报：``capture_reporter_report`` 记录结论分布，
    ``capture_subgoal_progress`` 是「steps_per_subgoal」的唯一来源 ——
    只有这里同时知道「这个子目标跑了多少步」和「结果如何」。
    """
    outcome = determine_outcome_for(state)
    failed_targets = build_failed_targets(state.executor_failures)
    actions_taken = build_actions_taken(state.executor_messages)
    failure_reason = failure_reason_for(state)
    screen_text = visible_screen_text(state)

    summary = code_summary(
        outcome=outcome,
        failed_count=len(failed_targets),
        failure_reason=failure_reason,
    )
    evidence = pick_evidence(
        None,
        log_text=render_execution_log(state.executor_messages),
        outcome=outcome,
        messages=state.executor_messages,
        screen_text=screen_text,
    )

    report = ExecutorReport(
        outcome=outcome,
        summary=summary,
        evidence=evidence,
        actions_taken=actions_taken,
        failed_targets=failed_targets,
        blocked_reason=blocked_reason_for(state),
        failure_reason=failure_reason,
    )

    telemetry.capture_reporter_report(
        task_id=ctx.trace_id,
        outcome=outcome,
        summary_len=len(summary),
        has_evidence=bool(evidence),
        failed_count=len(failed_targets),
    )
    telemetry.capture_subgoal_progress(
        task_id=ctx.trace_id,
        subgoal_id=_current_subgoal_id(state),
        steps=state.executor_steps,
        outcome=outcome,
        retry_count=_retry_count(state),
    )
    return report


# 结论层：代码判定，LLM 不参与


def step_limit_reached(state: State) -> bool:
    """这次尝试有没有撞到步数上限。

    Executor 撞上限时写的是 ``executor_done=True``，那个 True 的意思是
    「别再动了，收尾汇报」，不是「目标达成了」（见
    ``agents/executor/executor.py`` 的 step_limit 分支）。所以这里不能只看
    ``executor_done``，否则一次撞上限的尝试会被判成 ``completed``，Cortex 就会
    以为这个子目标做成了。

    默认不限步（``MAX_STEPS_PER_SUBGOAL is None``），所以这个判定恒为 False，
    ``outcome`` 不会是 ``blocked``。``blocked`` 这条分支留着，是为了有人把上限
    配回来时行为不变，代码路径都在，只是默认不触发。
    """
    if MAX_STEPS_PER_SUBGOAL is None:
        return False
    return state.executor_steps > MAX_STEPS_PER_SUBGOAL


def determine_outcome_for(state: State) -> Literal["completed", "failed", "blocked"]:
    """代码判定 outcome，规则无歧义，见 :func:`determine_outcome`。"""
    step_limit = step_limit_reached(state)
    # 撞了上限就不是「达成」，那个分支里 executor_done 只表示「收尾」。
    #
    # 模型明确放弃（executor_failed）同样不是「达成」，它是「做不到」。在这之前
    # done=True 会被无条件映射成 completed，而 executor 当时又没有「做不到」这个
    # 出口，所以「做不到」不可表达。模型只能一直试，于是原地打转，最后撞上
    # recursion_limit 崩掉。这是真机实测过的。
    done = bool(state.executor_done) and not step_limit and not state.executor_failed
    return determine_outcome(
        done=done,
        failures=state.executor_failures,
        step_limit_reached=step_limit,
        gave_up=bool(state.executor_failed),
    )


def failure_reason_for(state: State) -> str | None:
    """模型自己说的「做不到的原因」，逐字，不改写。

    不是「放弃」就返回 None。

    它是事实层。这句话是模型的原话，Cortex 读到的就是它自己说的
    那一句，不重写也不美化，所以「把失败说成完成」的老毛病在这里没有生存空间。
    """
    if not state.executor_failed:
        return None
    for message in reversed(state.executor_messages or []):
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None


def _current_subgoal_id(state: State) -> str:
    """当前子目标的 ID，给遥测用，不含目标原文。

    拿不到就返回 ``"unknown"``，遥测不能因为缺个 ID 就把埋点搞挂。
    """
    subgoal = get_current_subgoal(state.subgoal_plan)
    return subgoal.id if subgoal is not None else "unknown"


def _retry_count(state: State) -> int:
    """这个子目标被重发过几次，也就是换招重来的次数。

    不能用「失败事实条数」来近似，那会错在最常见的地方。重发有三条路径，其中
    两条根本不产生失败事实：

    - 规则换招：outcome 是 failed 或 blocked 时，cortex 用规则重新下发同一个
      子目标。
    - 撞步数上限：outcome 是 blocked 而不是 failed，没有任何失败目标。
    - cortex 判「还没成」：汇报说 completed，但证据不足，cortex 重新下发。

    前两条都不写失败事实，所以用失败条数算出来的 ``retry_count`` 恒为 0。那是
    一个看着有数其实是常数的假指标：真机实测里 3 个子目标实际重发了 5 次，
    ``retry_count`` 全程报 0。

    正确的口径是「这个子目标被下发过几次」减一。下发记录写在 ``agents_thoughts``
    里，append-only，只增不减，由 cortex 写、由 :mod:`agents.cortex.issue_log`
    统一计数，不用往 State 里加字段。
    """
    subgoal_id = _current_subgoal_id(state)
    if subgoal_id == "unknown":
        return 0
    return max(0, count_goal_issues(state.agents_thoughts, subgoal_id) - 1)


def blocked_reason_for(state: State) -> str | None:
    """blocked 的原因，是事实，由代码给。不是 blocked 就返回 None。"""
    if not step_limit_reached(state):
        return None
    return (
        f"step limit of {MAX_STEPS_PER_SUBGOAL} reached; "
        "the attempt was stopped before the goal was reached"
    )


# 事实层：从 state 直搬，逐字


def build_failed_targets(failures: list[dict] | None) -> list[FailedTarget]:
    """事实账本变成汇报里的失败清单，逐字，不改写。

    ``executor_failures`` 是 append-only 的失败事实（``utils/failed_targets.py``
    的 ``append_failure``），每一条都是独立的一次失败事件，所以 ``count`` 恒为 1。
    ``FailedTarget.count`` 的语义是「连着失败了几次」，和这里不是一回事，混用会
    读错。

    这里不是 ``state.failed_targets``。那个是熔断账本，成功即清、换键即整表
    替换，跨子目标的事实留不住。
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
    """从局部历史里抽出执行过的动作，每个一行。

    只取 AIMessage 上的 tool_calls。工具写回的是 ToolMessage，也就是结果，动作
    和结果在汇报里是分开的两层。
    """
    return [
        f"{call.get('name')}({format_args(call.get('args') or {})})"
        for message in messages or []
        if isinstance(message, AIMessage)
        for call in (getattr(message, "tool_calls", None) or [])
    ]


def format_args(args: dict) -> str:
    """动作参数压成一行，太长就截断，汇报不是日志全文。"""
    try:
        text = json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(args)
    return text if len(text) <= MAX_ARGS_LEN else text[:MAX_ARGS_LEN] + "…"


def render_execution_log(messages: list[AnyMessage] | None) -> str:
    """把局部历史渲染成执行日志。

    动作行和结果行分开。证据要从结果里挑，所以工具结果必须原样出现，不能被
    压掉。

    多模态结果只取文字。截图工具的结果是 ``[文字块, 图片块]``，
    ``str(content)`` 会把几十 KB 的 base64 变成一行 Python 字面量，证据里多出
    一坨噪声。图对汇报没用，汇报要的是「看到了什么」的事实，所以这里只留文字
    （见 ``utils/message_content.py``）。
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


# 证据：代码挑


def pick_evidence(
    raw: str | None,
    *,
    log_text: str,
    outcome: str,
    messages: list[AnyMessage] | None,
    screen_text: str | None = None,
) -> str | None:
    """挑证据。

    ``raw`` 是历史上「模型给出的证据」，现在没有调用方传它（汇报免 LLM），
    参数保留是为了 :func:`_contains` 那道硬校验还能被单独测到：凡是模型给的
    证据必须真的出现在日志里，改写过的就不算证据。

    ``outcome == "completed"`` 时证据不能空，因为 Cortex 要靠它独立判断「这个
    子目标真的完成了吗」，所以兜底给最近一条工具结果。

    光有工具结果仍然不够，理由见 :func:`visible_screen_text`。「已成功点击 WLAN」
    只说明「我点了」，不说明「我看到了 WLAN 页面」，Cortex 拿它判不了，于是重发。
    所以证据由两段组成，动作结果加上屏幕上的可见文字（``screen_text``，代码直搬
    的事实）。
    """
    candidate = (raw or "").strip()
    if candidate:
        if _contains(log_text, candidate):
            return compose_evidence(candidate, screen_text)
        logger.warning("[reporter] The provided evidence is not in the execution log; ignoring it")

    if outcome == "completed":
        return compose_evidence(last_tool_result(messages), screen_text)
    return screen_text


def last_tool_result(messages: list[AnyMessage] | None) -> str | None:
    """最近一条非空的工具结果，代码挑的证据兜底。"""
    for message in reversed(messages or []):
        if isinstance(message, ToolMessage):
            content = content_to_text(message.content).strip()
            if content:
                return content[:MAX_EVIDENCE_LEN]
    return None


# 证据的来源：动作结果不够，要「屏幕上因此看到了什么」
#
# 这是「为什么会重发」的根因，来自一次真机实测。判「这个子目标真的完成了吗」
# 需要的是屏幕状态，而执行日志里只有动作结果：「tap」的结果是「已成功点击元素
# [3] WLAN」，说的是「我点了」，不是「我看到了」；「swipe」的结果是「滑动成功」，
# 零信息。
#
# 于是汇报里从来没有「屏幕上因此出现了什么」，Cortex 判不了，就选了「还没成，
# 再来一次」。真机日志里 Cortex 的原话是：「报告的证据只有『滑动成功』这类操作
# 结果，summary 也是代码生成的，没有任何 WLAN 页面标题、开关状态或网络名称的
# 原文，因此无法确认已停留在 WLAN 页面。」
#
# 「屏幕上看到了什么」其实早就在 state 里了：``state.latest_screen_elements``
# 和 ``state.latest_ocr_results``。
#
# 它就是「动作执行之后」那一屏，这点必须成立，否则证据是旧的。环是 executor、
# executor_tools、contextor 再回到 executor，每执行一个动作都会回到
# contextor 重新抓屏。所以 executor 说 done 的那一刻，state 里躺着的就是最后
# 一个动作执行之后的屏。
#
# 修法的成本是零：不调 LLM，不额外抓屏，只是把已经在手边的文字搬进汇报。


# 证据里「屏幕可见文字」最多留多长。
#
# 预算按字符定，不按行数定。真机那一屏有 84 个带文字的元素，去重后 75 条，全量
# 只有 827 字符，按字符定预算本来就装得下，砍掉的其实是行数。
#
# 按行数截的代价很具体。小米应用商城「番茄畅听」详情页（Redmi Turbo 4）的元素表
# 有 152 行，主按钮「打开」在第 141 行。前 25 行里根本没有它，于是汇报给 Cortex
# 的证据里没有「打开」，Cortex 读不到「按钮显示打开」，正确地拒绝了这次汇报
# （它原话："it never shows the detail-page button reading 打开/已安装… The 安装
# text visible belongs to the recommended app 番茄音乐"），改下一个难十倍的子目标
# （去全部应用列表找图标），执行器掉进 MIUI 看点浮层陷阱，原地打转 16 分钟、
# 67.9 万 token。
#
# 现在留 1600 字符，是给更长的页留余量。
MAX_SCREEN_EVIDENCE_CHARS = 1600


def _screen_texts(state: State) -> list[str]:
    """这一屏所有看得见且不重复的文字，按屏序从上到下。

    去重按原文。「打开」和「打开 」是两条不同的视觉事实，合并掉就是丢事实。
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
    """把若干段文字拼起来，超预算就保首尾、中段省略。

    砍尾巴是错的。主操作按钮几乎总在屏底，真机上那一屏的「打开」在第 141/152 行。
    砍掉尾巴等于证据里没有主按钮，supervisor 核实不了，就拒绝并换一个更难的目标，
    执行器掉进死胡同。

    所以宁可中间少几行，也必须保住尾部，并且明说省略了多少条，不装成「全都看到
    了」。

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
        # 一条就超预算，这是极端情况，只能硬切，但至少保住尾巴
        return joined[-budget:]

    omitted = len(lines) - len(head) - len(tail)
    marker = f"{sep}…（中间 {omitted} 条省略）…{sep}"
    return f"{sep.join(head)}{marker}{sep.join(tail)}"


def visible_screen_text(state: State) -> str | None:
    """最后一屏（最后一个动作执行之后）看得见的文字，是事实，代码直搬。

    优先用元素表里的 ``label``（UI 节点的可见文字），没有就用 OCR 的 ``text``。
    两边都空才返回 None，那说明这一屏确实没文字可报，比如纯图标界面。

    取「屏上有什么」，不取「前 N 行」。按行数截会把屏底的主操作按钮砍掉，
    supervisor 因此核实不了、拒绝汇报、换一个更难的目标，执行器就掉进死循环。
    详见 :data:`MAX_SCREEN_EVIDENCE_CHARS` 上面那段。
    """
    lines = _screen_texts(state)
    if not lines:
        return None
    return _join_within_budget(lines, MAX_SCREEN_EVIDENCE_CHARS)


def compose_evidence(action_result: str | None, screen_text: str | None) -> str | None:
    """证据等于动作结果加屏幕上的可见文字。

    两段都要。动作结果说「我做了什么」，屏幕文字说「因此看到了什么」。只给前者，
    Cortex 判不了（见上面的根因）。

    超预算时只砍动作结果那截，屏幕文字那段一个字节都不动。主按钮在屏底，砍它
    等于把「按钮显示打开」从证据里抹掉。
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
    """归一化空白和大小写，用来核对「证据是不是日志里那句」。"""
    return " ".join(text.split()).lower()


def _contains(haystack: str, needle: str) -> bool:
    """核对 needle 的归一化形式是否出现在 haystack 里。"""
    return _normalize(needle) in _normalize(haystack)


def code_summary(*, outcome: str, failed_count: int, failure_reason: str | None = None) -> str:
    """纯代码摘要。

    ``summary`` 只影响「给人看的措辞」，不影响正确性：上层（Cortex）读的是
    ``outcome`` 和事实字段。

    ``failure_reason`` 是模型自己说的「做不到的原因」，逐字带上。模型主动放弃和
    模型没放弃只是没做成，在上层看来长得一模一样，都是 outcome=failed。不说清楚，
    Cortex 只能重新下发同一个目标，白白再烧一遍。
    """
    if outcome == "completed":
        return "子目标已完成（证据见下方：动作结果 + 屏幕上看到的文字）。"
    if outcome == "blocked":
        limit = MAX_STEPS_PER_SUBGOAL
        if limit is None:
            # 默认不限，blocked 本来不可达；这里是「有人把上限配回来了」的兜底。
            return "子目标未完成，见失败清单。"
        return f"子目标未完成：触到步数上限（{limit} 步），见失败清单。"
    if failure_reason:
        return f"子目标做不到（执行器主动放弃）：{failure_reason}"
    if failed_count:
        return f"子目标未完成，见失败清单（{failed_count} 条失败事实）。"
    return "子目标未完成，未记录到失败事实。"
