"""消息内容的两种形态：**纯文本** 与 **多模态块列表**。

为什么要有这个模块
------------------
LangChain 的 ``Message.content`` 有两种形态：

* ``str`` —— 绝大多数消息是这样；
* ``list[str | dict]`` —— 多模态消息（文字块 + 图片块）。

⚠️ **两种形态不能混着处理。** 以前的代码一律写 ``str(message.content or "")``，
在纯文本消息上完全正确；一旦某条消息带了图片块，这行就会把整个列表
``str()`` 掉 —— 变成一个 Python 列表字面量（``[{'type': 'image_url', ...}]``），
里面塞着几十 KB 的 base64。

三个受害点（都已经被这个工具踩到）：

1. ``agents/executor/prompt.py`` 的工具结果截断 —— 它会把图片消息**整个换成
   一个字符串**，图就没了，模型拿到的是一坨 base64 噪声；
2. ``agents/reporter/reporter.py`` 的执行日志 —— 汇报正文里会多出几十 KB 的
   base64，既污染报告又白烧 reporter 的 token；
3. ``agents/executor/tool_node.py`` 的错误提取 —— 失败时把列表塞进日志。

⇒ 凡是「把消息内容当文字看」的地方，都必须先过这里的
:func:`content_to_text`；凡是「要保留图」的地方，用 :func:`has_image_block`
判断后再决定怎么处理。

⚠️ **2026-10-27：截图通道已整条移除**（含按需工具 ``take_screenshot``），
模型现在收不到任何图。本模块的三个函数**仍然保留** —— 它们防的是
「任何多模态内容」，不只是截图；哪天某个工具又往消息里塞图片块，
这些坑不该再踩一遍。

**图片块长什么样**
------------------
图片块用 **OpenAI 原生形态**::

    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...."}}

（langchain_core 的 ``convert_to_openai_messages`` 原样透传这种块，
见 ``langchain_core/messages/utils.py`` 里 ``block.get("type") == "image_url"``
那一支。）

但为了不把话说死，下面两个判定同时认三种写法：

* ``image_url``（OpenAI 原生）：``{"type": "image_url", "image_url": {...}}``
* ``image``（Anthropic / 标准块）：``{"type": "image", "source": {...}}``
* ``file``（标准文件块，截图偶尔也这么发）：``{"type": "file", ...}``
"""

from __future__ import annotations

from typing import Any

#: 会被当成「图片」的块类型。
_IMAGE_BLOCK_TYPES = frozenset({"image_url", "image", "file"})


def content_to_text(content: Any) -> str:
    """把消息内容压成纯文本。**图片块被跳过**（不是变成 base64 噪声）。

    * ``str`` → 原样返回；
    * ``list`` → 把 ``text`` 块 / 裸字符串拼起来，其余块跳过；
    * ``None`` → 空串。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
            continue
        # 图片 / 文件块：这里**不**要它的数据（那是几十 KB 的 base64）。
        # 真要读图，模型自己看得见。
    return "\n".join(parts)


def has_image_block(content: Any) -> bool:
    """这条消息里有没有图片块。"""
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, dict) and block.get("type") in _IMAGE_BLOCK_TYPES:
            return True
    return False


def with_text_replaced(content: Any, text: str) -> str | list[Any]:
    """把内容里的**文字**换成 ``text``，**图片块原样保留**。

    截断工具结果时用：截的是文字，图不能跟着没 —— 那是模型要的东西本身。

    * ``str`` → 返回新的字符串；
    * ``list`` → 文字部分合并成一个 text 块，其余块（图片/文件）**顺序不变**地留下。
    """
    if not isinstance(content, list):
        return text

    blocks: list[Any] = []
    text_written = False
    for block in content:
        is_text = isinstance(block, str) or (
            isinstance(block, dict) and block.get("type") == "text"
        )
        if is_text:
            if not text_written:
                blocks.append({"type": "text", "text": text})
                text_written = True
            continue
        blocks.append(block)
    if not text_written:
        blocks.insert(0, {"type": "text", "text": text})
    return blocks
