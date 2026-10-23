"""Cortex 的输出契约。

改造前 Cortex 每轮要产出：一份动作 JSON（``decisions``）、动作的理由
（``decisions_reason``）、完成的子目标 id 列表、还有「我要图」的请求。
那是「**每轮看屏、逐个动作决策**」的形态。

Executor 自主执行之后，感知和动作全下沉给了 Executor（它自己看屏、自己决定
下一个动作）。Cortex 退化成**监督者**：一个子目标结束时判断一次，只说三件事：

* ``next_subgoal`` —— 下一个子目标的**目标陈述**（自然语言，不是动作脚本）
* ``subgoal_completed`` —— 上一个子目标是否真的达成了
* ``completion_reason`` —— 凭什么这么判

刻意**没有**的字段：

* ``needs_replan`` —— 重规划的判断权只在 Orchestrator（方案 §6.5.1）。
  两处都能判 replan 就会出现两个互相覆盖的判断源。
* ``decisions`` / ``decisions_reason`` / ``goals_completion_reason`` ——
  不再逐轮写动作和长推理，输出因此小了一个数量级。
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
