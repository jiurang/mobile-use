"""Orchestrator 的输出契约。

谁在验收
--------
Cortex 只说「我判定这个子目标完成了」，**真正把子目标标成 SUCCESS 的是
Orchestrator**（``orchestrator.py`` 里调 ``complete_subgoals_by_ids``）。
这是「完成判定要留给 Orchestrator 验收」那条契约的落点 ——
两个地方都能判完成，就会出现两个互相覆盖的判断源。

⚠️ **字段名 ``needs_replaning`` 的拼写是「故意的」，别改。**
它出现在给模型看的 JSON schema 里（``needs_replaning`` / ``needs_replanning``），
改名 = 改提示词 = 改模型行为，不是重构。
"""

from pydantic import BaseModel, Field


class OrchestratorOutput(BaseModel):
    completed_subgoal_ids: list[str] = Field(
        default_factory=list,
        description="IDs of subgoals that can now be marked as complete",
    )
    needs_replaning: bool = Field(
        description="Whether the orchestrator needs to replan the subgoal plan",
    )
    reason: str
