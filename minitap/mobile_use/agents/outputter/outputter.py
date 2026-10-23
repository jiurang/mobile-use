"""Outputter：把整条链路的思考过程压成**一份结构化产出**。

它是链路的**最后一跳**，产出物直接给调用方（SDK / TUI），不再回流给任何
agent。

两条路径
--------

* **结构化**（``output_config.structured_output`` 有值）⇒ 走
  ``with_structured_output(schema)``，产出按 schema 校验过的对象。
  ``schema`` 可以是 dict（原生 JSON schema）或一个 ``BaseModel``。
* **纯文本** ⇒ 模型直接写内容，按 JSON 解析。

⚠️ ``output_config.output_description`` 不为空时会**多发一条 HumanMessage**
（把描述当成追加指令）。同时，只要结构化响应对象**自带** ``content`` 字段，
返回的就是**模型写的 JSON**（``json.loads(response.content)``）而不是
``model_dump()`` —— 两条路径的数据来源不一样，调用方拿到的东西也不同。
"""

import json
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from minitap.mobile_use.config import OutputConfig
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.prompt_language import load_prompt_template, pick
from minitap.mobile_use.services.llm import get_llm, invoke_llm_with_timeout_message, with_fallback
from minitap.mobile_use.utils.conversations import is_ai_message
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


async def outputter(
    ctx: MobileUseContext, output_config: OutputConfig, graph_output: State
) -> dict:
    logger.info("Starting Outputter Agent")
    last_message = graph_output.messages[-1] if graph_output.messages else None

    system_message = pick(
        "You are a helpful assistant tasked with generating "
        "the final structured output of a multi-agent reasoning process.",
        "你是一个助手，负责把多智能体推理过程的最终结果整理成结构化输出。",
        language=ctx.prompt_language,
    )
    human_message = load_prompt_template(
        Path(__file__).parent.joinpath("human.md"),
        ctx.prompt_language,
    ).render(
        initial_goal=graph_output.initial_goal,
        agents_thoughts=graph_output.agents_thoughts,
        structured_output=output_config.structured_output,
        output_description=output_config.output_description,
        last_ai_message=last_message.content
        if last_message and is_ai_message(message=last_message)
        else None,
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
    structured_llm = llm
    structured_llm_fallback = llm_fallback

    if output_config.structured_output:
        schema: dict | type[BaseModel] | None = None
        so = output_config.structured_output

        if isinstance(so, dict):
            schema = so
        elif isinstance(so, BaseModel):
            schema = type(so)
        elif isinstance(so, type) and issubclass(so, BaseModel):
            schema = so

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
    )  # type: ignore
    if isinstance(response, BaseModel):
        if output_config.output_description and hasattr(response, "content"):
            response = json.loads(response.content)  # type: ignore
            return response
        return response.model_dump()
    elif hasattr(response, "content"):
        return json.loads(response.content)  # type: ignore
    else:
        logger.info("Found unknown response type: " + str(type(response)))
    return response
