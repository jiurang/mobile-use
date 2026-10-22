"""「目标下发记录」的格式与计数 —— **cortex 写，cortex 和 reporter 都要读**。

为什么单独一个模块
------------------
这条记录写在 ``state.agents_thoughts`` 里，格式是::

    [cortex] Issued goal for subgoal <subgoal_id>: <目标陈述>

写它的是 **cortex**（每次下发目标时追加一条），读它的有两个：

* **cortex 自己** —— 数「这个子目标被下发过几次」和「规则换招用了几次」
  （规则换招有次数上限，用完要退回 LLM）。
* **reporter** —— 数「这个子目标被**重发**过几次」，用来上报
  ``retry_count``（方案 §11.6 的 ``steps_per_subgoal`` 指标的一部分）。

放在 ``cortex.cortex`` 里会让 reporter 反向依赖整个 cortex 节点模块
（连带 services/llm、tools 那一串 import）。放这里，两个方向都只依赖
一个几行的纯函数模块。

为什么用 thoughts 而不是往 State 里加字段
----------------------------------------
``agents_thoughts`` 是 **append-only** 的，任务生命周期内只增不减 ——
正好当计数器用，**不用往 State 里加字段**（State 是多方共改的地盘，
加字段要过契约）。
"""

from __future__ import annotations

# 「下发目标」这条 thought 的固定前缀（用来数这个子目标被下发过几次）
ISSUED_GOAL_PREFIX = "Issued goal for subgoal "

# 规则换招在 thought 里的标记 —— 用来数「规则已经用过几次」。
#
# 刻意用一个**只由规则路径写出来的**字符串做标记：这样数出来的就是
# 「规则换招用了几次」，而不是「下发了几次」（后者会把 LLM 自己的换招也算进去，
# 额度会莫名其妙地被消耗掉）。
RULE_RETRY_MARKER = "rule-based retry"

# state.asanitize_update 给 agent 写的 thought 加的前缀
_AGENT_PREFIX = "[cortex] "


def _issue_marker(subgoal_id: str) -> str:
    """数下发次数用的前缀匹配串。"""
    return f"{_AGENT_PREFIX}{ISSUED_GOAL_PREFIX}{subgoal_id}:"


def count_goal_issues(thoughts: list[str], subgoal_id: str) -> int:
    """这个子目标被**下发过几次**目标。

    数的是 Cortex 自己写进 ``agents_thoughts`` 的那条记录
    （``Issued goal for subgoal <id>: <goal>``，``state.py`` 会加 ``[cortex] `` 前缀）。
    ``agents_thoughts`` 是 append-only 的，所以这个数在任务生命周期内只增不减 ——
    正好当「换招次数」用，**不需要往 State 里加字段**。
    """
    if not subgoal_id:
        return 0
    marker = _issue_marker(subgoal_id)
    return sum(1 for thought in thoughts if thought.startswith(marker))


def count_rule_retries(thoughts: list[str], subgoal_id: str) -> int:
    """这个子目标已经用过几次**规则换招**。

    只数带 :data:`RULE_RETRY_MARKER` 的下发记录 —— LLM 自己发的换招不算，
    否则额度会被模型的下发次数吃掉，规则换招就永远轮不上。
    """
    if not subgoal_id:
        return 0
    marker = _issue_marker(subgoal_id)
    return sum(
        1
        for thought in thoughts
        if thought.startswith(marker) and RULE_RETRY_MARKER in thought
    )
