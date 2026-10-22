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

from typing import Literal

from pydantic import BaseModel, Field

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
    # 🔴 **2026-10-22 删掉了 `need_screenshot`。**
    #
    # Executor 不再有「要图」这个能力：截图通道整条关掉。
    #
    # 为什么关
    # --------
    # 1. **二问的代价是整整一次 LLM 调用**（真机实测 executor 单次决策中位
    #    5,276ms）—— 收益只是「模型偶尔看不见颜色/图标/遮挡」。
    # 2. 截图进 prompt 是**唯一永远不可缓存**的那一块（base64 长度随画面变，
    #    前缀缓存直接失效）—— 实测它占了整个易变尾部的 92%。
    # 3. `state.latest_screenshot` 的**唯一消费者**就是这条二问路径
    #    （`agents/executor/executor.py` 的 `_capture_screenshot`）——
    #    关掉之后截图整条路就没有消费者了。
    #
    # 代价
    # ----
    # 模型从此**看不到画面**，只能靠 UI 元素表 + OCR。颜色、图标、图片内容、
    # 遮挡关系这些它读不出来。**这是个能力削减，不是纯优化** —— 真机跑完
    # 要看「步数 / 失败目标数」有没有涨。
    #
    # 模型如果仍然吐出 `need_screenshot`（旧习惯 / 提示词残留），pydantic
    # 会**忽略**这个多余的键（extra=ignore），不会报错、也不会触发任何
    # 额外动作。


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
