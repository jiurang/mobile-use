"""Executor 自主执行的数据契约。

两个模型，两个角色
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

动作本身不单独建模：用 ``dict`` 承载（见 :class:`ExecutorOutput.actions`
的说明）—— 它要原样喂给 dispatcher，多一层模型只会多一层失配的机会。

还有一层 :class:`ExecutorOutputBase`
----------------------------------------
``ExecutorOutput`` 和「模型不能看图时那个 schema」是**兄弟类**，共用
:class:`ExecutorOutputBase`。两种形态都成立的代码依赖基类 —— 依赖
``ExecutorOutput`` 会在「不能看图」那条路上把**正确**的模型回包判成错误
（真机实测 2026-10-22 23:19）。详见 :func:`executor_output_schema`。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, create_model

from minitap.mobile_use.utils.failed_targets import FailedTarget


class ExecutorOutputBase(BaseModel):
    """Executor 输出里**两种形态都有**的字段。

    为什么要有这个基类
    ------------------
    模型**不能看图**时，schema 里不能有 ``need_screenshot``（不给模型看它
    不存在的选项）。实现上只能另造一个类 —— 而那个类和 :class:`ExecutorOutput`
    是**兄弟，不是父子**（见 :func:`executor_output_schema`）。

    ⇒ 凡是「两种形态都成立」的代码，只能依赖这个基类。依赖
    :class:`ExecutorOutput` 就会踩到真机实测过的那一刀：
    ``isinstance(response, ExecutorOutput)`` 在「不能看图」那条路上**恒为
    False**，模型的回包完全正确，动作却被整批丢掉
    （真机实测 2026-10-22 23:19，Redmi Turbo 4）。

    字段一个都不能少 —— 少一个，两种形态的行为就不一致了。

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


# ⚠️ **下面这个类的 docstring 是「模型看得见的工具描述」，不是给人看的注释。**
#
# ``convert_to_openai_tool(schema)`` 会把 ``cls.__doc__`` 塞进 JSON schema 的
# ``description``，而这个 schema 每轮都发给模型 ⇒ **改这段 docstring = 改提示词**。
# 它和字段描述一样属于「动了就会改变模型行为」的契约，改之前先想清楚。
#
# 这条约束是实测踩出来的：本轮重构把 docstring 改写了，A/B 一比就发现
# 发给模型的工具描述变了（字段部分没变，只有 description 变了）。
#
# 而「这个类和基类是什么关系」这类**给人看**的信息，**别写进 docstring** ——
# 写在上面的注释里。基类的 docstring 不发给模型，可以随便写。
class ExecutorOutput(ExecutorOutputBase):
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


def executor_output_schema(*, can_see_screenshot: bool) -> type[ExecutorOutputBase]:
    """按「模型能不能看图」返回 Executor 的输出 schema。

    不能看图时**连字段都不给模型** —— 提示词里没有图相关的词，schema 里
    也不该留 `need_screenshot`：模型看不到的东西，不该让它以为能举手要。

    返回的那个类是 :class:`ExecutorOutputBase` 的**子类**（名字仍然叫
    ``ExecutorOutput``），所以它和 :class:`ExecutorOutput` 共享基类、
    共享那四个字段，**只少一个** ``need_screenshot``。

    ⚠️ **工具名必须还是 ``ExecutorOutput``**：模型被教的是「你只能调
    ``ExecutorOutput``」，解析层也拿这个名字对账（喊错工具名会被拒 ——
    见 ``services/llm.py`` 的 ``_ToolCallStructuredRunnable._parse``）。
    工具名来自类名（``convert_to_openai_tool(schema)["function"]["name"]``），
    所以类名不能改。
    """
    if can_see_screenshot:
        return ExecutorOutput
    # 只给基类、**不重传字段** —— 字段（含 description / default /
    # default_factory）由基类继承而来，结果与「把字段重抄一遍」**逐字相同**，
    # 但少一层容易抄漏、抄错的中间物。
    return create_model("ExecutorOutput", __base__=ExecutorOutputBase)


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

    ``done=True``            → ``completed``
    步数超限 / 熔断          → ``blocked``
    其余一切（没达成）       → ``failed``

    ⚠️ **``failures`` 不参与判定** —— 「有失败事实」和「什么都没发生」
    都判 ``failed``。失败**几条**、失败在**哪**，属于**事实层**，由
    :class:`ExecutorReport.failed_targets` 单独承载（逐字搬运，见
    ``agents/reporter/reporter.py``）。

    为什么不拿失败条数去细分 outcome：Cortex 拿到的是「成了 / 没成」，
    它据此决定换招还是放弃；失败清单是给**它判断「换哪招」**用的，
    不是用来把 outcome 拆成 completed-with-warnings 之类的第三种。
    判定规则一旦依赖条数，「失败 0 条但没达成」和「失败 3 条」就会落到
    不同分支 —— 而这两种都是「没成」，规则必须无歧义。

    这条函数是「事实不丢」的结构性保证：``outcome`` **不经 LLM**，
    所以 Reporter 没有机会把失败说成完成。
    """
    if done:
        return "completed"
    if step_limit_reached:
        return "blocked"
    return "failed"
