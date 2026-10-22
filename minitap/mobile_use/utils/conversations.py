from typing import Any, TypeGuard

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


def is_ai_message(message: BaseMessage) -> TypeGuard[AIMessage]:
    return isinstance(message, AIMessage)


def is_human_message(message: BaseMessage) -> TypeGuard[HumanMessage]:
    return isinstance(message, HumanMessage)


def is_tool_message(message: BaseMessage) -> TypeGuard[ToolMessage]:
    return isinstance(message, ToolMessage)


def is_tool_for_name(tool_message: ToolMessage, name: str) -> bool:
    return tool_message.name == name


def get_screenshot_message_for_llm(screenshot_base64: str, caption: str | None = None):
    """把 base64 截图包成一条多模态消息。

    ``caption`` 是一句**给人话**的说明，放在图片前面。模型看到一张没有解释的
    图片时会自己猜它对应哪一屏 —— 而这里它可能对应**上一轮**的屏（Executor
    举手要图、动作执行后屏已经变了）。一句话把这件事说清楚，比让它猜便宜。

    ⚠️ **caption 里的文字会进 prompt** —— 别写屏幕上的内容（用户数据）。
    """
    prefix = "" if screenshot_base64.startswith("data:image") else "data:image/jpeg;base64,"
    # 类型写成 `list[str | dict[str, Any]]`：langchain 的 content 参数就是这个
    # 联合类型，`list` 是不变的（invariant），注解得和它**完全一致**才过得去
    # pyright（写成 `list[dict]` 会被判成 `dict[Unknown, Unknown]` 而报错）。
    content: list[str | dict[str, Any]] = []
    if caption:
        content.append({"type": "text", "text": caption})
    content.append(
        {
            "type": "image_url",
            "image_url": {"url": f"{prefix}{screenshot_base64}"},
        }
    )
    return HumanMessage(content=content)
