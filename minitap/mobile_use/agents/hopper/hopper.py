"""Hopper：从一大坨文本里**捞出**指定的那一份数据。

和 Executor 那条链路无关 —— 它不碰设备、不驱动界面，只做一件事：把
``data`` 和「要什么」的 ``request`` 一起交给模型，让模型按结构化输出把
**那一份**数据挑出来。

所以它的 ``temperature=0``（要的是确定性，不是发挥），且失败没有降级路径
（没有产出就没有产出，调用方自己决定怎么办）。
"""

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.prompt_language import load_prompt_template, pick
from minitap.mobile_use.services.llm import get_llm, invoke_llm_with_timeout_message, with_fallback
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


class HopperOutput(BaseModel):
    found: bool = Field(description="True if the requested data was found, False otherwise.")
    output: str | None = Field(description="The extracted data if found, null otherwise.")
    reason: str = Field(
        description="A short explanation of what you looked for"
        + " and how you decided what to extract."
    )


async def hopper(
    ctx: MobileUseContext,
    request: str,
    data: str,
) -> HopperOutput:
    logger.info("Starting Hopper Agent")
    system_message = load_prompt_template(
        Path(__file__).parent.joinpath("hopper.md"),
        ctx.prompt_language,
    ).render()
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(
            content=(
                f"{request}\n"
                + pick(
                    "Here is the data you must dig:",
                    "下面是你要从中挖掘的数据：",
                    language=ctx.prompt_language,
                )
                + f"\n{data}"
            )
        ),
    ]

    llm = get_llm(ctx=ctx, name="hopper", is_utils=True, temperature=0).with_structured_output(
        HopperOutput
    )
    llm_fallback = get_llm(
        ctx=ctx, name="hopper", is_utils=True, use_fallback=True, temperature=0
    ).with_structured_output(HopperOutput)
    response: HopperOutput = await with_fallback(
        main_call=lambda: invoke_llm_with_timeout_message(
            llm.ainvoke(messages), agent_name="hopper", task_id=ctx.trace_id
        ),
        fallback_call=lambda: invoke_llm_with_timeout_message(
            llm_fallback.ainvoke(messages),
            agent_name="hopper",
            task_id=ctx.trace_id,
            used_fallback=True,
        ),
    )  # type: ignore
    return response
