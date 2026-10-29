"""Executor 自主执行的数据契约。

这个文件定义两个模型：一个收模型的话，一个给上层看。

:class:`ExecutorOutput` 是 Executor 的 LLM 每轮吐的东西。它不含 16 个设备工具的
JSON schema：Executor 只绑定这一个结构化输出 schema，工具的用法靠 ``prompts/system/40-tools.md``
里的文本工具表教（365 token，静态可缓存）。绑定全量 schema 会让每轮多背 4,008
token，而那份 schema 不进缓存键，所以每轮都按满价计费（实测，见
``docs/智能体协作v2.md``）。

:class:`ExecutorReport` 是 Reporter 产出、给 Cortex 和 Orchestrator 看的东西。
``outcome`` 由代码判定，不经 LLM（见 :func:`determine_outcome`）；
``failed_targets`` 与 ``evidence`` 由代码直搬，LLM 只写 ``summary``。

动作本身不单独建模，用 ``dict`` 承载：它要原样喂给 dispatcher，多一层模型只会
多一层失配的机会。

只有一种形态。模型现在看不见截图，也就不需要「按需决定给不给它看图」的第二个
schema，解析层对着 :class:`ExecutorOutput` 判 ``isinstance`` 就是对着真类判，
不会再出现「现场造出来的同名类不等于真类」的情况（真机实测 2026-10-22 23:19）。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from minitap.mobile_use.utils.failed_targets import FailedTarget


# 下面这个类的 docstring 是模型看得见的工具描述，不是给人看的注释。
# convert_to_openai_tool 会把 cls.__doc__ 塞进 JSON schema 的 description，
# 而这个 schema 每轮都发给模型，改这段 docstring 就是改提示词。
# 它和字段描述一样属于「动了就会改变模型行为」的契约，改之前先想清楚。
#
# 这条约束是实测踩出来的：改写 docstring 之后字段部分没变，发给模型的工具描述
# 变了。
#
# 「这个类怎么用、为什么这么用」这类给人看的信息写在上面的注释里，docstring 只
# 留给模型。
class ExecutorOutput(BaseModel):
    """Executor 每一轮的 LLM 输出。

    ``actions`` 用 ``list[dict]`` 而不是更严的模型，是**故意的**：

    * 每个 dict 的形状是 ``{"action": "<工具名>", "<参数名>": <值>, ...}``，
      和 ``agents/cortex/dispatcher.py`` 期待的形状**逐字一致**
      （dispatcher 把所有非 ``action`` 的 key 平铺成 tool args）。
    * schema 保持成 ``{"type": "array", "items": {"type": "object"}}``，
      模型按文本工具表填，**不需要**下发 16 个工具的 JSON schema。
    * 形状不合法时由 dispatcher 挡下（返回 None），**拒绝理由要回灌给
      Executor**（见 ``doc/executor-autonomy-plan.md`` §3.1.1 坑 2），
      退回 LLM 重来会死循环。
    """

    actions: list[dict] = Field(
        default_factory=list,
        description=(
            "The actions to execute now. Each is an object like "
            '{"action": "tap", "target": {"index": 7}}. '
            "Always a list, even when there is exactly one action."
        ),
    )
    agent_thought: str = Field(
        default="",
        description=(
            "Short: first the result of the previous step (checked against what "
            "you want), then why this action. Do NOT put the goal statement here."
        ),
    )
    done: bool = Field(
        default=False,
        description="True when the goal is reached. Then actions is empty.",
    )
    done_reason: str | None = Field(
        default=None,
        description="Why the goal is considered reached. Required when done=true.",
    )
    failed: bool = Field(
        default=False,
        description=(
            "True when this goal CANNOT be achieved and you are giving up on it. "
            "Then actions is empty. Use this instead of retrying forever: "
            "report what you tried and why it did not work."
        ),
    )
    failed_reason: str | None = Field(
        default=None,
        description="Why the goal cannot be achieved. Required when failed=true.",
    )


class ExecutorReport(BaseModel):
    """一个子目标尝试的汇报。

    除 ``summary`` 外全部由代码填充，LLM 不得改写，依据是
    ``utils/thoughts_compaction.py`` 的硬规则：事实不能压。
    """

    outcome: Literal["completed", "failed", "blocked"] = Field(
        description="Whether the subgoal was reached. Decided by code, not by the LLM."
    )
    summary: str = Field(
        default="",
        description="1-2 sentences describing what was done and what the result was.",
    )
    evidence: str | None = Field(
        default=None,
        description=(
            "The concrete result that proves the subgoal was reached "
            "(e.g. the text that was read off the screen). "
            "Required when outcome == 'completed'."
        ),
    )
    actions_taken: list[str] = Field(
        default_factory=list,
        description="The actions performed, in order. One line each.",
    )
    failed_targets: list[FailedTarget] = Field(
        default_factory=list,
        description=(
            "The targets that failed during this attempt. Copied verbatim from "
            "the executor failure ledger; never reworded."
        ),
    )
    blocked_reason: str | None = Field(
        default=None,
        description="Why the attempt was blocked (step limit / circuit breaker).",
    )
    failure_reason: str | None = Field(
        default=None,
        description=(
            "Why the executor gave up on this subgoal (it said failed=true). "
            "Copied verbatim; never reworded."
        ),
    )


def determine_outcome(
    *,
    done: bool,
    failures: list[dict] | None = None,
    step_limit_reached: bool = False,
    gave_up: bool = False,
) -> Literal["completed", "failed", "blocked"]:
    """由代码判定 outcome，规则无歧义（见 ``docs/智能体协作v2.md``）。

    ``done=True`` 判 ``completed``；``gave_up=True`` 判 ``failed``，对应模型明确
    说「做不到」；步数超限或熔断判 ``blocked``；其余一切（没达成）判 ``failed``。

    ``failures`` 不参与判定：「有失败事实」和「什么都没发生」都判 ``failed``。
    失败几条、失败在哪属于事实层，由 :class:`ExecutorReport.failed_targets`
    单独承载，逐字搬运，见 ``agents/reporter/reporter.py``。

    不拿失败条数去细分 outcome，是因为 Cortex 拿到的是「成了 / 没成」，它据此
    决定换招还是放弃；失败清单是给它判断「换哪招」用的，不是用来把 outcome 拆
    成 completed-with-warnings 之类的第三种。判定规则一旦依赖条数，「失败 0 条
    但没达成」和「失败 3 条」就会落到不同分支，而这两种都是「没成」，规则必须
    无歧义。

    这条函数是「事实不丢」的结构性保证：``outcome`` 不经 LLM，所以 Reporter
    没有机会把失败说成完成。

    ``gave_up`` 必须排在 ``done`` 前面，两者语义相反（「做成了」和「做不到」）。
    模型曾经只能表达前者，「做不到」在契约里不可表达，于是原地打转，最后撞
    ``recursion_limit`` 崩掉（真机实测 2026-10-26：卡了 70 步、45 分钟、
    409 万 input token）。
    """
    if gave_up:
        return "failed"
    if done:
        return "completed"
    if step_limit_reached:
        return "blocked"
    return "failed"
