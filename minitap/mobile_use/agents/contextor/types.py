"""Contextor 的两个输出契约。

``ContextorOutput`` 是模型在「前台 app 不是锁定 app」时给出的判断：要不要把锁定
app 重新拉起来。``AppLockVerificationOutput`` 是这次校验的结果，附一句给模型看的
说明。

说明末尾的 emoji 是既有约定，逐字保留。
"""

from typing import Literal

from pydantic import BaseModel, Field


class ContextorOutput(BaseModel):
    """Output schema for the Contextor agent decision."""

    should_relaunch_app: bool = Field(..., description="Whether to relaunch the locked app")
    reasoning: str = Field(
        ..., description="Explanation of why we should or should not relaunch the app"
    )


class AppLockVerificationOutput(BaseModel):
    """应用锁校验的结果。"""

    package_name: str = Field(..., description="Package name of the app that was verified")
    reasoning: str | None = Field(default=None, description="Reasoning for the decision")
    status: Literal["already_in_foreground", "relaunched", "allowed_deviation", "error"] = Field(
        ..., description="Status of the decision"
    )

    def to_optional_message(self) -> str | None:
        """给模型看的一句话说明。

        已经在前台时返回 None，没有需要告诉模型的事。其余状态返回一句话，通常
        以模型的 reasoning 开头；校验失败时改用一句固定说明。
        """
        msg = f"App {self.package_name}"
        if self.reasoning:
            msg = f"{self.reasoning} {msg}"
        match self.status:
            case "already_in_foreground":
                return None
            case "relaunched":
                msg += " was successfully relaunched ✅"
            case "allowed_deviation":
                msg += " was allowed deviation ⚠️"
            case "error":
                msg = f"Could not verify app lock for {self.package_name}."

        return msg
