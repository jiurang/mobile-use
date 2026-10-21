from pydantic import BaseModel, Field


class CortexOutput(BaseModel):
    decisions: str | None = Field(
        default=None, description="The decisions to be made. A stringified JSON object"
    )
    decisions_reason: str | None = Field(default=None, description="The reason for the decisions")
    goals_completion_reason: str | None = Field(
        default=None,
        description="The reason for the goals completion, if there are any goals to be completed.",
    )
    complete_subgoals_by_ids: list[str] = Field(
        default_factory=list, description="List of subgoal IDs to complete"
    )
    need_screenshot: bool = Field(
        default=False,
        description=(
            "Set to true ONLY when the screen element list and the OCR text are "
            "NOT enough to decide what to do - e.g. the target is only "
            "distinguishable by color, an icon, an image, or it is obscured. "
            "You will then be given a screenshot and asked again. "
            "Leave it false whenever the element list already identifies the "
            "target: the screenshot costs a lot and is never cached."
        ),
    )
