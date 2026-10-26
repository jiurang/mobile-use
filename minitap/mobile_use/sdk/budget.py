"""全局预算耗尽时的**优雅收尾**（不是崩溃）。

为什么要有它
------------
2026-10-26 真机实测（mobile-manager execution 880，任务「淘宝红包签到」）：
``GraphRecursionError: Recursion limit of 752 reached`` ——
45 分钟、198 次 LLM 调用、409 万 input token 之后，任务**崩**了，
只留下一句「递归上限到了」：

* 没说跑到哪了（停在第 3 个子目标）
* 没说完成了什么（前 2 个子目标其实做成了）
* 没说这是**预算**问题还是**任务本身**做不到

⇒ 撞上限不是「异常」，是**预算用完了**。它该产生一份**收尾报告**：
做了哪些、停在哪、还差什么、下次该怎么跑。

这个模块只做翻译：把「最后的状态」变成那份报告。
判断（是打转还是真需要那么多步）**不归它管** —— 那是「有没有进展」
那一层的职责（``agents/executor/progress.py``）。

为什么不是一个「限制」
--------------------
这是**降级**，不是加限制：预算还是那个预算，跑到哪儿还是跑到哪儿，
只是**到头了要有话说**，而不是抛一个谁都看不懂的异常。
"""

from __future__ import annotations

from typing import Any

from minitap.mobile_use.agents.planner.types import SubgoalStatus

#: 报告里最多列几个子目标（计划很长时不用全铺）。
MAX_LISTED_SUBGOALS = 20

#: 报告的最长字符数。
#:
#: 为什么要有上限：这份报告会**整个**进上层的错误信息里，而 Manager 的
#: ``executions.error_message`` 是 ``varchar(4000)`` —— 超了不是「被截断」，
#: 是**整条记录写不进去**。所以在源头截，别指望数据库兜底。
MAX_REPORT_CHARS = 2800


class BudgetExhaustedError(RuntimeError):
    """全局预算耗尽。

    ⚠️ **它不是「任务崩了」** —— 报告已经在 ``.report`` 里，
    任务实际做到哪儿、完成了什么，都在里面。

    它仍然会被抛出去，是为了让上层把这次运行标成**失败**而不是成功 ——
    任务目标确实没达成（这是事实，不能说成完成）。
    但「失败的原因」现在是一句人话 + 一份报告，不是
    ``Recursion limit of 752 reached without hitting a stop condition``。
    """

    def __init__(self, message: str, *, report: str = "") -> None:
        super().__init__(message)
        self.report = report


def _status_mark(status: Any) -> str:
    if status == SubgoalStatus.SUCCESS:
        return "✅ 已完成"
    if status == SubgoalStatus.FAILURE:
        return "❌ 判定失败"
    if status == SubgoalStatus.PENDING:
        return "⏳ 进行中"
    return "· 还没开始"


def build_budget_report(
    state: Any,
    *,
    max_steps: int,
    recursion_limit: int,
) -> str:
    """把「最后的状态」翻译成一份人能读的收尾报告。

    报告里只有**事实**（哪些子目标什么状态、停在哪个、跑了多少步），
    没有评价、没有建议性的结论 —— 「是打转还是真需要那么多步」这个判断
    不在这里做。
    """
    plan = list(getattr(state, "subgoal_plan", None) or [])
    current_goal = getattr(state, "executor_goal", None) or ""
    executor_steps = int(getattr(state, "executor_steps", 0) or 0)
    total_steps = int(getattr(state, "total_steps", 0) or 0)
    report = getattr(state, "executor_report", None)

    done = [s for s in plan if s.status == SubgoalStatus.SUCCESS]
    pending = [s for s in plan if s.status != SubgoalStatus.SUCCESS]

    lines: list[str] = []
    lines.append(
        f"预算用完了：本次上限是 {max_steps} 个动作步（换算成 {recursion_limit} 个图步），"
        f"实际跑了 {total_steps} 个动作步。"
    )
    lines.append(
        f"计划 {len(plan)} 个子目标，完成 {len(done)} 个，"
        f"剩下 {len(pending)} 个没完成。"
    )

    if plan:
        lines.append("")
        lines.append("子目标清单：")
        for subgoal in plan[:MAX_LISTED_SUBGOALS]:
            extra = ""
            if subgoal.completion_reason:
                extra = f" —— {subgoal.completion_reason}"
            lines.append(
                f"- {_status_mark(subgoal.status)} [ID:{subgoal.id}] "
                f"{subgoal.description}{extra}"
            )
        if len(plan) > MAX_LISTED_SUBGOALS:
            lines.append(f"- …（还有 {len(plan) - MAX_LISTED_SUBGOALS} 个没列）")

    if current_goal:
        lines.append("")
        lines.append(
            f"停在的子目标（已跑 {executor_steps} 步）：{current_goal}"
        )

    if report is not None and getattr(report, "summary", None):
        lines.append("")
        lines.append(f"最后一次汇报：{report.summary}")
        if getattr(report, "failure_reason", None):
            lines.append(f"（执行器说做不到：{report.failure_reason}）")

    lines.append("")
    lines.append(
        "这是**预算**用完，不是任务本身做不到 —— "
        "如果这个任务确实需要更多步，把任务配置里的「最大步数」调大再跑一次。"
    )
    report = "\n".join(lines)
    if len(report) > MAX_REPORT_CHARS:
        report = report[: MAX_REPORT_CHARS - 30] + "\n…（报告过长，已截断）"
    return report


__all__ = [
    "BudgetExhaustedError",
    "MAX_LISTED_SUBGOALS",
    "MAX_REPORT_CHARS",
    "build_budget_report",
]
