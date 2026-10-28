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

**图片块长什么样**
------------------
本仓库发出去的图片块是 **OpenAI 原生形态**::

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

from minitap.mobile_use.prompt_language import PromptLanguage, pick

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


#: 图像通道的**标题** —— 图所在那条消息的第一块。
#:
#: 图和元素表 / OCR 是**两条不同的通道**：文字通道给编号（``[i]``），图上没有
#: 编号。这个标题就是把两条通道隔开的那道线 —— 标题之下是像素，不是另一份
#: 屏幕描述。
#:
#: ⚠️ 必须**逐字固定** —— 它每轮都出现在 prompt 的同一个位置，
#: 一旦改成随轮次变化的内容，就等于把易变块推回了可缓存的前缀里。
#:
#: 英文在前、中文在后（``pick`` 的约定）：两种语言并排写出来，对照关系在眼前。
SCREENSHOT_CHANNEL_LABEL_EN = "Image perception"
SCREENSHOT_CHANNEL_LABEL_ZH = "图片感知"


def screenshot_channel_label(*, language: PromptLanguage) -> str:
    """图像通道的标题（按提示词语言取一份）。"""
    return pick(SCREENSHOT_CHANNEL_LABEL_EN, SCREENSHOT_CHANNEL_LABEL_ZH, language=language)


def build_screenshot_blocks(image_b64: str) -> list[dict[str, Any]]:
    """当屏截图的内容块：**只有图**（OpenAI 原生 ``image_url`` 形态）。

    ⚠️ **这里没有图注。** 「图是补充、寻址走元素表的行号、一屏一图就够」
    这套契约写在**静态提示词**里（``agents/executor/executor.md``）—— 逐轮重说
    一遍只是重复，还把一段易变文字塞进历史。模型每轮都读得到静态提示词。

    这一块要挂在 prompt 的**最后一条消息**里 —— 元素表旁边。
    元素表每轮都变 ⇒ 那一带本来就是**易变区** ⇒ 图换不换、删不删，
    都**不影响**前面那段可缓存的前缀。

    ⚠️ 反过来，图挂在局部历史里就是被钉在历史中段 ⇒ 下一次拍图「顶替」它时
    那个位置的字节变了 ⇒ **前缀从历史中段起整段作废**
    （真 API 实测，MiniMax-M3.1-Flash：图片块进 provider 的缓存键）。
    """
    return [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}]


def build_screenshot_message_content(
    image_b64: str, *, language: PromptLanguage
) -> list[dict[str, Any]]:
    """图像通道那条消息的内容：**标题 + 图**。

    独立成一条消息、排在 prompt 最末 —— 元素表 / OCR 那边是文字通道，
    这里是像素通道，两条通道之间隔着这个标题。

    ⚠️ 标题**逐字固定**（见 :data:`SCREENSHOT_CHANNEL_LABEL_EN`）—— 它每轮
    都出现在同一个位置，改成随轮次变化的内容会把易变块推进可缓存的前缀。
    """
    return [
        {"type": "text", "text": screenshot_channel_label(language=language)},
        *build_screenshot_blocks(image_b64),
    ]
