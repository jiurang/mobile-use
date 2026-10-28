"""「目标下发记录」的格式与计数。

cortex 写这条记录，cortex 和 reporter 都读它。

记录写在 ``state.agents_thoughts`` 里，格式是：

    [cortex] Issued goal for subgoal <subgoal_id>: <目标陈述>

读它的地方有两个。cortex 要数「这个子目标被下发过几次」和「规则换招用了几次」
—— 规则换招有次数上限，用完要退回 LLM。reporter 要数「这个子目标被重发过
几次」，用来上报 ``retry_count``。

单独成模块是因为依赖方向。放进 ``cortex.cortex`` 会让 reporter 反向依赖整个
cortex 节点，连带 services/llm 和 tools 那一串 import。放在这里，两个方向
都只依赖一个几行的纯函数模块。

用 thoughts 而不是往 State 里加字段，是因为 ``agents_thoughts`` 是 append-only
的，任务生命周期内只增不减，正好当计数器用。State 是多方共改的地盘，
加字段要过契约。
"""

from __future__ import annotations

# 「下发目标」这条 thought 的固定前缀，用来数这个子目标被下发过几次。
ISSUED_GOAL_PREFIX = "Issued goal for subgoal "

# 规则换招在 thought 里的标记，用来数「规则已经用过几次」。
#
# 刻意用一个只有规则路径会写出来的字符串做标记：这样数出来的才是「规则换招
# 用了几次」，而不是「下发了几次」。后者会把 LLM 自己的换招也算进去，额度会
# 莫名其妙地被消耗掉。
RULE_RETRY_MARKER = "rule-based retry"

# state.asanitize_update 给 agent 写的 thought 加的前缀
_AGENT_PREFIX = "[cortex] "


def _issue_marker(subgoal_id: str) -> str:
    """数下发次数用的前缀匹配串。"""
    return f"{_AGENT_PREFIX}{ISSUED_GOAL_PREFIX}{subgoal_id}:"


def count_goal_issues(thoughts: list[str], subgoal_id: str) -> int:
    """这个子目标被下发过几次目标。

    数的是 cortex 自己写进 ``agents_thoughts`` 的那条记录
    （``Issued goal for subgoal <id>: <goal>``，``state.py`` 会加 ``[cortex] ``
    前缀）。``agents_thoughts`` 是 append-only 的，所以这个数在任务生命周期内
    只增不减，正好当「换招次数」用，不需要往 State 里加字段。
    """
    if not subgoal_id:
        return 0
    marker = _issue_marker(subgoal_id)
    return sum(1 for thought in thoughts if thought.startswith(marker))


def count_rule_retries(thoughts: list[str], subgoal_id: str) -> int:
    """这个子目标已经用过几次规则换招。

    只数带 :data:`RULE_RETRY_MARKER` 的下发记录。LLM 自己发的换招不算，
    否则额度会被模型的下发次数吃掉，规则换招就永远轮不上。
    """
    if not subgoal_id:
        return 0
    marker = _issue_marker(subgoal_id)
    return sum(
        1 for thought in thoughts if thought.startswith(marker) and RULE_RETRY_MARKER in thought
    )
