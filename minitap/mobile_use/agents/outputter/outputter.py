"""Outputter：把整条链路的思考过程压成一份结构化产出。

它是链路的最后一跳，产出物直接给调用方（SDK / TUI），不再回流给任何 agent。

有两条路径：

* 结构化（``output_config.structured_output`` 有值）时走
  ``with_structured_output(schema)``，产出按 schema 校验过的对象。``schema``
  可以是 dict（原生 JSON schema），也可以是一个 ``BaseModel``。
* 纯文本时模型直接写内容，按 JSON 解析。

``output_config.output_description`` 不为空时会多发一条 HumanMessage，把描述
当成追加指令。只要结构化响应对象自带 ``content`` 字段，返回的就是模型写的
JSON（``json.loads(response.content)``），不是 ``model_dump()``；两条路径的数据
来源不一样，调用方拿到的东西也不一样。
"""

import json
from pathlib import Path
from typing import Any, cast

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from minitap.mobile_use.config import OutputConfig
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.prompt_language import load_prompt_template, pick
from minitap.mobile_use.services.llm import get_llm, invoke_llm_with_timeout_message, with_fallback
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


async def outputter(
    ctx: MobileUseContext, output_config: OutputConfig, graph_output: State
) -> dict[str, Any]:
    """把整条链路的思考过程压成一份结构化产出。"""
    logger.info("Starting Outputter Agent")

    system_message = pick(
        "You are a helpful assistant tasked with generating "
        "the final structured output of a multi-agent reasoning process.",
        "你是一个助手，负责把多智能体推理过程的最终结果整理成结构化输出。",
        language=ctx.prompt_language,
    )
    human_message = load_prompt_template(
        Path(__file__).parent.joinpath("prompts", "en", "human.md"),
        ctx.prompt_language,
    ).render(
        initial_goal=graph_output.initial_goal,
        agents_thoughts=graph_output.agents_thoughts,
        structured_output=output_config.structured_output,
        output_description=output_config.output_description,
    )

    messages: list[BaseMessage] = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_message),
    ]

    if output_config.output_description:
        messages.append(HumanMessage(content=output_config.output_description))

    llm = get_llm(ctx=ctx, name="outputter", is_utils=True, temperature=1)
    llm_fallback = get_llm(
        ctx=ctx, name="outputter", is_utils=True, use_fallback=True, temperature=1
    )

    # structured_output 有三种写法：一个 pydantic 类、一个 pydantic 实例，或者
    # 一份原生 JSON schema（dict）。三种都要落到同一个 with_structured_output
    # 调用上。
    schema: dict[str, Any] | type[BaseModel] | None = None
    so = output_config.structured_output
    if so:
        if isinstance(so, dict):
            schema = so
        elif isinstance(so, BaseModel):
            schema = type(so)
        elif isinstance(so, type) and issubclass(so, BaseModel):
            schema = so

    # 带 schema 的 runnable 返回 pydantic 实例或 dict，不带的返回消息；两种形状
    # langchain 只标成「dict 或 BaseModel」，这里放宽成 Any，靠下面的 isinstance
    # 分支收窄。
    structured_llm: Runnable[LanguageModelInput, Any] = llm
    structured_llm_fallback: Runnable[LanguageModelInput, Any] = llm_fallback
    if schema is not None:
        structured_llm = llm.with_structured_output(schema)
        structured_llm_fallback = llm_fallback.with_structured_output(schema)

    response = await with_fallback(
        main_call=lambda: invoke_llm_with_timeout_message(
            structured_llm.ainvoke(messages),
            agent_name="outputter",
            task_id=ctx.trace_id,
        ),
        fallback_call=lambda: invoke_llm_with_timeout_message(
            structured_llm_fallback.ainvoke(messages),
            agent_name="outputter",
            task_id=ctx.trace_id,
            used_fallback=True,
        ),
    )

    # 这两条分支上 content 就是模型写的那段 JSON 文本。langchain 的标注允许它
    # 还是 list（多模态内容块），但结构化输出走的是纯文本，这里收窄成 str。
    if isinstance(response, BaseModel):
        if output_config.output_description and hasattr(response, "content"):
            return json.loads(cast("str", response.content))
        return response.model_dump()
    if hasattr(response, "content"):
        return json.loads(cast("str", response.content))
    logger.info("Found unknown response type: " + str(type(response)))
    return response
