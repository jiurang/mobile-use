"""Summarizer：给全局消息历史瘦身。

它只管 ``state.messages``（跨子目标的整条对话历史），不碰 Executor 的局部历史
（``state.executor_messages``）。局部历史是「事实通道」，
``utils/thoughts_compaction.py`` 的纪律（事实不能压）在那里生效。

剪的规则是从最老的消息开始删，但不从一条 ``ToolMessage`` / ``HumanMessage``
中间切开。切开了会留下「孤儿 tool_call」（assistant 的 tool_call 没有对应的
ToolMessage），OpenAI 兼容端点直接 400。所以一旦遇到 ``ToolMessage`` /
``HumanMessage``，就把它和它之前的所有消息一起删（倒序遍历时置位
``start_removal``）。

这个节点不调 LLM。「摘要」是节点名的历史包袱，它做的就是裁剪。
"""

from typing import Any

from langchain_core.messages import (
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)

from minitap.mobile_use.constants import MAX_MESSAGES_IN_HISTORY
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.graph.state import State


class SummarizerNode:
    """裁剪全局消息历史：只删最老的一段，不改任何一条消息的内容。"""

    def __init__(self, ctx: MobileUseContext) -> None:
        self.ctx = ctx

    async def __call__(self, state: State) -> dict[str, Any]:
        if len(state.messages) <= MAX_MESSAGES_IN_HISTORY:
            return {}

        nb_removal_candidates = len(state.messages) - MAX_MESSAGES_IN_HISTORY

        remove_messages: list[RemoveMessage] = []
        start_removal = False

        for msg in reversed(state.messages[:nb_removal_candidates]):
            if isinstance(msg, ToolMessage | HumanMessage):
                start_removal = True
            if start_removal and msg.id:
                remove_messages.append(RemoveMessage(id=msg.id))

        return await state.asanitize_update(
            ctx=self.ctx,
            update={
                "messages": remove_messages,
            },
        )
