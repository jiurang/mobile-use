from typing import TypeGuard

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


def is_ai_message(message: BaseMessage) -> TypeGuard[AIMessage]:
    return isinstance(message, AIMessage)


def is_human_message(message: BaseMessage) -> TypeGuard[HumanMessage]:
    return isinstance(message, HumanMessage)


def is_tool_message(message: BaseMessage) -> TypeGuard[ToolMessage]:
    return isinstance(message, ToolMessage)


def is_tool_for_name(tool_message: ToolMessage, name: str) -> bool:
    return tool_message.name == name
