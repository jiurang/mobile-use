"""Executor 自主执行的数据契约。

三个模型，三个角色
-----------------

* :class:`ExecutorOutput` —— **Executor 的 LLM 每轮吐的东西**。
  注意它**不含** 16 个设备工具的 JSON schema：Executor 用
  ``_ToolCallStructuredRunnable`` 只绑定**这一个** schema，
  工具的用法靠 ``executor.md`` 里的**文本工具表**教（365 tok，静态可缓存）。
  绑定全量 schema 会让每轮多背 4,008 token 且不进缓存键（实测，见
  ``doc/executor-autonomy-plan.md`` §3.1）。

* :class:`ExecutorReport` —— **Reporter 产出、给 Cortex 和 Orchestrator 看的东西**。
  ``outcome`` 由**代码**判定（见 :func:`determine_outcome`），不经 LLM；
  ``failed_targets`` / ``evidence`` 也由代码直搬。LLM 只写 ``summary``。

* :class:`ExecutorAction` —— 不单独用了，动作用 ``dict`` 承载（见
  :class:`ExecutorOutput.actions` 的说明）。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from minitap.mobile_use.utils.failed_targets import FailedTarget


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
            "One short sentence: why this action. Max 1 sentence. "
            "Do NOT put the goal statement here."
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
    need_screenshot: bool = Field(
        default=False,
        description=(
            "True when the screen description you were given cannot tell your "
            "targets apart (you need color, icons, images, or to know what covers "
            "what). The screenshot of the current screen is then attached to your "
            "NEXT turn, and the action you gave in this turn is NOT executed - you "
            "decide again with the picture in front of you."
        ),
    )
    # 🔴 **2026-10-22 改：`need_screenshot` 回来了，但换了一种用法。**
    #
    # 原来（改造前）：模型说「我要图」→ 同轮补一张 → **再问一次 LLM**。
    # 那次二问是整整一次 LLM 调用（真机实测 executor 单次决策中位 5,276ms），
    # 换来的只是「模型偶尔能看见颜色/图标/遮挡」—— 代价和收益不成比例。
    #
    # 现在：模型说「我要图」→ **下一轮**的 prompt 里带上当屏截图。
    # 循环本来就要继续下一轮 ⇒ **零额外 LLM 调用**。
    #
    # 为什么下一轮给的图一定是对的：
    # - 图和元素表是**同一次抓屏**的产物（`agents/contextor/contextor.py` 在
    #   同一个节点里抓屏 + 建表），所以图和 `[i]` 编号指的是同一屏；
    # - 图是 **contextor 刚抓的那一帧**，不是重抓的 —— 重抓会多一次 adb
    #   往返，还可能拿到不同帧，编号和画面对不上。
    #
    # 通道开关在 `PerceptionConfig.screenshot`（Manager 侧）：
    # - `"always"`     → 每轮都带图，模型不必举手；
    # - `"on_demand"`  → **默认**：模型举手才有图；
    # - `"off"`        → 永远没有图，**提示词和 schema 里都不留图的痕迹**。


def executor_output_schema(*, can_see_screenshot: bool) -> type[BaseModel]:
    """按「模型能不能看图」返回 Executor 的输出 schema。

    不能看图时**连字段都不给模型** —— 提示词里没有图相关的词，schema 里
    也不该留 `need_screenshot`：模型看不到的东西，不该让它以为能举手要。

    ⚠️ **工具名必须还是 ``ExecutorOutput``**：模型被教的是「你只能调
    ``ExecutorOutput``」，解析层也拿这个名字对账（喊错工具名会被拒 ——
    见 ``services/llm.py`` 的 ``_ToolCallStructuredRunnable._parse``）。
    """
    if can_see_screenshot:
        return ExecutorOutput
    fields: dict[str, Any] = {
        name: (info.annotation, info)
        for name, info in ExecutorOutput.model_fields.items()
        if name != "need_screenshot"
    }
    return create_model("ExecutorOutput", **fields)


class ExecutorReport(BaseModel):
    """一个子目标尝试的汇报。

    **除 ``summary`` 外全部由代码填充**，LLM 不得改写
    （依据 ``utils/thoughts_compaction.py`` 的硬规则：事实不能压）。
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


def determine_outcome(
    *,
    done: bool,
    failures: list[dict] | None = None,
    step_limit_reached: bool = False,
) -> Literal["completed", "failed", "blocked"]:
    """**代码**判定 outcome，规则无歧义（见方案 §6.2.1）。

    ``done=True``                → completed
    步数超限 / 熔断              → blocked
    有失败且未 done              → failed
    其余（什么都没发生）        → failed

    这条函数是「事实不丢」的结构性保证：``outcome`` **不经 LLM**，
    所以 Reporter 没有机会把失败说成完成。
    """
    if done:
        return "completed"
    if step_limit_reached:
        return "blocked"
    if failures:
        return "failed"
    return "failed"
