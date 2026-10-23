"""Summarizer：给**全局**消息历史瘦身。

只管 ``state.messages``（跨子目标的整条对话历史），**不碰** Executor 的局部
历史（``state.executor_messages``）—— 局部历史是「事实通道」，
``utils/thoughts_compaction.py`` 的纪律（事实不能压）在那里生效。

剪的规则
--------
从**最老**的消息开始删，但**不从一条 ``ToolMessage`` / ``HumanMessage``
中间切开** —— 切开了会留下「孤儿 tool_call」（assistant 的 tool_call 没有
对应的 ToolMessage），OpenAI 兼容端点直接 400。

⇒ 所以一旦遇到 ``ToolMessage`` / ``HumanMessage``，就把它**和它之前的所有
消息**一起删（倒序遍历时置位 ``start_removal``）。

⚠️ 这个节点**不调 LLM** —— 「摘要」是节点名的历史包袱，它做的就是裁剪。
"""

from langchain_core.messages import (
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)

from minitap.mobile_use.constants import MAX_MESSAGES_IN_HISTORY
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.graph.state import State


class SummarizerNode:
    def __init__(self, ctx: MobileUseContext):
        self.ctx = ctx

    async def __call__(self, state: State):
        if len(state.messages) <= MAX_MESSAGES_IN_HISTORY:
            return {}

        nb_removal_candidates = len(state.messages) - MAX_MESSAGES_IN_HISTORY

        remove_messages = []
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
