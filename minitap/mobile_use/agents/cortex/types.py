"""Cortex 的输出契约。

Cortex 是监督者：Executor 负责看屏和动作，Cortex 在一个子目标结束时判断一次。
感知和动作都在 Executor 侧（它自己看屏、自己决定下一个动作），所以这份契约里
没有逐轮的动作、动作的理由、完成的子目标 id 列表，也没有「我要图」的请求。
输出因此小了一个数量级。

它只说三件事：

* ``next_subgoal`` —— 下一个子目标的目标陈述，自然语言，不是动作脚本。
* ``subgoal_completed`` —— 上一个子目标是否真的达成了。
* ``completion_reason`` —— 凭什么这么判。

刻意没有的字段：

* ``needs_replan`` —— 重规划由 PlanManager 的规则（重发预算超限）或 Cortex
  自己的 ``give_up`` 触发，不在本输出里。两处都能判 replan 就会出现两个互相
  覆盖的判断源。
* ``decisions`` / ``decisions_reason`` / ``goals_completion_reason`` ——
  逐轮的动作和长推理不进 Cortex 的输出。
"""

from pydantic import BaseModel, Field


class CortexOutput(BaseModel):
    next_subgoal: str | None = Field(
        default=None,
        description=(
            "The goal statement for the next subgoal, as one natural-language "
            "sentence describing what 'done' looks like (e.g. 'read the Wi-Fi "
            "network name currently connected and report it exactly as shown'). "
            "It is a target, NOT an action script: do not name tools, elements or "
            "indices. Leave it empty when the subgoal is done, or when you are "
            "giving up on it."
        ),
    )
    subgoal_completed: bool = Field(
        default=False,
        description=(
            "True only when the last attempt really reached the subgoal AND you "
            "have evidence for it (the report's evidence/summary shows the thing "
            "that was asked for). A step towards the goal is not a completed goal."
        ),
    )
    completion_reason: str | None = Field(
        default=None,
        description=(
            "1-3 sentences: what the evidence shows, or why the subgoal is not "
            "done, or why you are giving up on it. Required unless the subgoal "
            "is completed with clear evidence."
        ),
    )
