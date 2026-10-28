"""计划的数据形状：子目标、子目标状态，以及 LLM 交回来的解析结果。

``PlannerOutput`` 是模型返回的原始形状，``Subgoal`` 是计划在 State 里的样子。
两者之间的转换在 :mod:`agents.planner.planner` 的 ``_build_subgoals`` 里。

``Subgoal.__str__`` 的输出会进 orchestrator 的 thought，也就进了模型提示词。
它结尾的状态符号是给模型看的速记，一个字都不能动：改动这个格式等于改模型
看到的东西。
"""

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel


class PlannerSubgoalOutput(BaseModel):
    description: str


class PlannerOutput(BaseModel):
    subgoals: list[PlannerSubgoalOutput]


class SubgoalStatus(Enum):
    NOT_STARTED = "NOT_STARTED"
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class Subgoal(BaseModel):
    """计划里的一个子目标。

    状态由 :mod:`agents.planner.utils` 里的几个函数原地改写，别处只读。
    """

    id: Annotated[str, "Unique identifier of the subgoal"]
    description: Annotated[str, "Description of the subgoal"]
    completion_reason: Annotated[
        str | None, "Reason why the subgoal was completed (failure or success)"
    ] = None
    status: SubgoalStatus
    started_at: Annotated[datetime | None, "When the subgoal started"] = None
    ended_at: Annotated[datetime | None, "When the subgoal ended"] = None

    def __str__(self) -> str:
        """一行计划：ID、描述、状态符号，有完成原因就带上原因。

        这个格式会进 orchestrator 的 thought，因而进模型提示词，改动即改变
        模型行为。
        """
        status_emoji = "❓"
        match self.status:
            case SubgoalStatus.SUCCESS:
                status_emoji = "✅"
            case SubgoalStatus.FAILURE:
                status_emoji = "❌"
            case SubgoalStatus.PENDING:
                status_emoji = "⏳"
            case SubgoalStatus.NOT_STARTED:
                status_emoji = "(not started yet)"

        output = f"- [ID:{self.id}]: {self.description} : {status_emoji}."
        if self.completion_reason:
            output += f" Completion reason: {self.completion_reason}"
        return output

    def __repr__(self) -> str:
        return str(self)
