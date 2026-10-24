"""截图工具：把当前屏幕拍成一张图，交给模型看。

它是什么位置
------------
**UI 元素表与 OCR 的补充，不是第三条感知通道。**

这个仓库的感知有两条主通道（``context.PerceptionConfig``）：

* ``ui_hierarchy`` —— UI 层级树 → 元素表（能寻址：``Target.index``）
* ``ocr`` —— 截图 OCR 文本（能读字，但读不出图标 / 颜色 / 布局）

这两条都是**文字**通道，都有各自的盲区：图标、插图、颜色、被弹窗盖住的东西、
「这个按钮到底长什么样」—— 元素表里只有一行 ``[7] "搜索"``，模型不知道它长什么样。

本工具补的就是这一块：**模型自己喊一声要看图，框架就把当屏那张图给它**。

⚠️ **它是补充，不是替代。**

* 寻址仍然走元素表：``[i]`` 编号是**寻址契约**，图上没有编号。模型看到图之后
  要点某个东西，还是得报元素表里的行号。
* 每屏一图就够：屏幕一变图就过期，而每张图要花掉约 1,200 input token
  （真机实测，MiniMax-M3.1：640x1422 的图 = 1,175 token；原图 1220x2712
  = 2,378 token；压到 320x711 = 314 token。
  ⚠️ 别用 ``utils/image_compress.py`` 里那张 tile 公式表 —— 那是
  OpenAI/Gemini 口径，MiniMax 的图片计价高 2 倍多，见
  ``scripts/probe_image_token_cost.py``）。
* 图**不替代** OCR：OCR 认出来的字比人眼读图更准、更便宜。

为什么做成「工具」而不是 2026-10-23 下线掉的那条图片通道
---------------------------------------------------------
下线的那条是**每轮通道**（``perception.screenshot`` = always/on_demand/off）：
模型在输出 schema 里举一次手（``need_screenshot``），框架**扣下这一轮的动作**、
回灌一条 HumanMessage 走 retry 边，下一轮的 prompt 末尾才附当屏截图。

做成工具之后：

* 模型要图的动作**照常执行**（截图不动屏幕），不占一次「扣动作 + 重来」；
* 不需要 schema 里多一个字段（``need_screenshot`` 那种「举手」字段会让模型
  学会为它撒谎）；
* 图挂在**工具结果**上，模型看得见「这是我刚要的那张图」，归属清楚。

⚠️ **图和元素表是同一屏。** 工具在 contextor 抓完屏之后才跑，中间没有任何动作
改动过屏幕 ⇒ 图与元素表描述的是同一屏，编号契约不受影响。

provider 支持面（2026-10-24 真打 MiniMax-M3.1-Flash-Preview 实测）
-----------------------------------------------------------------
工具结果里带图（``role=tool`` 的 content 是 ``[text, image_url]``）是**可以**的：
探针 ``scripts/probe_minimax_vision.py`` 三组实测 ——

    A. 纯文本 user 消息          HTTP 200，prompt_tokens=210
    B. user 消息带图             HTTP 200，**模型读出了图里的字**，338
    C. tool 消息带图（工具结果）  HTTP 200，**模型读出了图里的字**，404

⇒ 「模型看不见图」不是 provider 的限制，是当时**没人给模型看**。
"""

from __future__ import annotations

import asyncio
import base64
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from minitap.mobile_use.constants import EXECUTOR_MESSAGES_KEY
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.controllers.unified_controller import UnifiedMobileController
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.tools.tool_wrapper import ToolWrapper
from minitap.mobile_use.utils.image_compress import compress_screenshot_base64
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.message_content import has_image_block

logger = get_logger(__name__)

#: 图的 JPEG 质量。与 ``utils/image_compress.py`` 的默认一致（q50 实测可读）。
SCREENSHOT_JPEG_QUALITY = 50

#: 旧截图被顶掉时留在历史里的一句话。
#:
#: 为什么留一句话而不是直接删掉那条消息：删 ToolMessage 会留下**孤儿 tool_call**
#: （assistant 那边的 tool_calls 还在），OpenAI 兼容端点直接 400
#: （见 ``agents/summarizer/summarizer.py`` 的模块文档）。所以是**替换内容**，
#: 不是删除消息。
SUPERSEDED_NOTE = "(this screenshot was superseded by a newer one)"


async def capture_screen_base64(ctx: MobileUseContext) -> str:
    """抓一张当屏截图，返回**原始分辨率**的 base64（不带 data URI 前缀）。

    两条路，快的那条优先：

    * **快路**：``ctx.ui_adb_client.get_screenshot_base64()`` ——
      uiautomator 的 ``device.screenshot()``，真机实测 **166ms**，只截图不动层级树。
    * **通用路**（iOS / 云设备 / 快路不可用）：控制器那条
      ``take_screenshot()``。Android 上它会顺带 dump 一次层级树（多花
      134~206ms，动画屏最多 800ms），但结果一样。

    快路失败**不抛**，直接回落通用路 —— 截图是补充通道，不该因为一条快路
    挂了就整个动作失败。
    """
    client = getattr(ctx, "ui_adb_client", None)
    getter = getattr(client, "get_screenshot_base64", None)
    if getter is not None:
        try:
            image = await asyncio.to_thread(getter)
            if image:
                return image
            logger.warning("[take_screenshot] fast capture returned nothing; falling back")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[take_screenshot] fast capture failed ({e}); falling back")

    return await UnifiedMobileController(ctx).take_screenshot()


def build_screenshot_content(image_b64: str) -> list[str | dict[str, Any]]:
    """工具结果的 content：**一段说明 + 一张图**。

    说明是给模型看的，必须把三件事讲清楚：

    1. 图和元素表是**同一屏**（不是另一屏）；
    2. 图是**补充**，寻址仍然走元素表的 ``[i]``；
    3. 一屏一图就够（token 成本 + 过期）。

    ⚠️ 说明里**不要写屏幕上的内容**（那是用户数据），只讲图本身的性质。
    """
    text = (
        "Screenshot of the current screen - the same screen the element table "
        "describes. It does not touch the screen.\n\n"
        "The picture SUPPLEMENTS the element table and the OCR text, it does not "
        "replace them:\n"
        "- Use it for what those two cannot show: icons, images, colors, layout, "
        "whether a popup covers something.\n"
        "- Keep using the element table for WHAT TO TAP: the [i] row numbers stay "
        "the addressing contract, this picture has no numbers on it.\n"
        "- One picture per screen is enough: it goes stale as soon as the screen "
        "changes, and each one costs about 1,200 input tokens."
    )
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
    ]


def _superseded_screenshots(state: State) -> list[Any]:
    """历史里**带着图**的那些工具结果（要顶掉的那几张）。

    只保留**最新一张**图是有理由的，不只是省钱：

    * 旧图对应的是**旧屏** —— 屏幕变了它就是错的，留着会误导模型
      （「我看到的是这个」其实早就不是了）；
    * 每张图约 1,200 input token，而局部历史**每轮都要整段重发** ⇒ 攒图等于
      每轮多烧钱。

    顶掉的方式是**替换内容**（留一句话），不是删消息 —— 删 ToolMessage 会留下
    孤儿 tool_call，端点 400（见 ``agents/summarizer/summarizer.py``）。
    """
    return [
        message
        for message in (getattr(state, "executor_messages", None) or [])
        if has_image_block(getattr(message, "content", None))
    ]


async def _error_command(
    ctx: MobileUseContext,
    state: State,
    *,
    agent_thought: str,
    tool_call_id: str,
    reason: str,
) -> Command:
    """失败分支的**统一出口**：一条 error ToolMessage + 一条 Command。

    息屏和抓图失败是同一件事的两面 ——「拿不到一张有用的图」⇒ 都回灌一条
    说人话的失败结果，都**不**把整个动作判成失败重试（它是补充通道）。
    两边共用这一个出口，避免两段一模一样的 Command 构造各自漂移。
    """
    agent_outcome = take_screenshot_wrapper.on_failure_fn(reason)
    tool_message = ToolMessage(
        tool_call_id=tool_call_id,
        content=agent_outcome,
        additional_kwargs={"error": agent_outcome},
        status="error",
    )
    return Command(
        update=await state.asanitize_update(
            ctx=ctx,
            update={
                "agents_thoughts": [agent_thought, agent_outcome],
                EXECUTOR_MESSAGES_KEY: [tool_message],
            },
            agent="executor",
        ),
    )


def get_take_screenshot_tool(ctx: MobileUseContext):
    @tool
    async def take_screenshot(
        agent_thought: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[State, InjectedState],
    ) -> Command:
        """Take a picture of the current screen and look at it.

        Use this ONLY when the element table and the OCR text do not tell you
        enough - for example: you cannot tell which of two similar rows is the
        button, the target is an icon with no text, a popup covers the thing
        you need, or you need to check a color or a layout.

        The picture is a SUPPLEMENT to the element table and the OCR text, not
        a replacement: it shows what those two cannot (icons, images, colors,
        layout). Keep addressing elements by their element-table row number.

        It does not touch the screen, and one picture per screen is enough -
        it goes stale as soon as the screen changes.
        """
        controller = UnifiedMobileController(ctx)

        # 屏幕灭着的时候截出来是一张全黑图 —— 发给模型只是白花 1,200 token。
        # 如实告诉它「先唤醒」，比给它一张黑图有用得多。
        power_state = await controller.get_screen_power_state()
        if power_state == "off":
            return await _error_command(
                ctx,
                state,
                agent_thought=agent_thought,
                tool_call_id=tool_call_id,
                reason=(
                    "the screen is off, so the picture would be black - call "
                    "wake_screen first, then take the picture again"
                ),
            )

        try:
            raw = await capture_screen_base64(ctx)
            # PIL 的解码 / 缩放 / JPEG 编码是**纯 CPU 活**（真机 1220x2712 约
            # 100~300ms）。直接调会**堵住事件循环** —— 同一条循环上还有
            # perception 与 LLM 请求在跑。丢线程里。
            image_b64 = await asyncio.to_thread(
                compress_screenshot_base64,
                raw,
                quality=SCREENSHOT_JPEG_QUALITY,
            )
        except Exception as e:  # noqa: BLE001
            # 截图失败**不**该把整个动作判成失败重试 —— 它是补充通道，
            # 拿不到就如实说，让模型退回元素表 / OCR 继续。
            return await _error_command(
                ctx,
                state,
                agent_thought=agent_thought,
                tool_call_id=tool_call_id,
                reason=f"{type(e).__name__}: {e}",
            )

        # 报的是 **JPEG 的真实字节数**，不是 base64 串长 —— base64 比原文大
        # ~33%，拿它当 KB 报给模型，那个数字是错的（模型会拿它做推理）。
        jpeg_bytes = len(base64.b64decode(image_b64))
        agent_outcome = take_screenshot_wrapper.on_success_fn(jpeg_bytes)
        tool_message = ToolMessage(
            tool_call_id=tool_call_id,
            content=build_screenshot_content(image_b64),
            status="success",
        )

        # 旧图顶掉：只留最新一张。**替换内容**而不是删消息（孤儿 tool_call 会 400）。
        superseded = [
            message.model_copy(update={"content": SUPERSEDED_NOTE})
            for message in _superseded_screenshots(state)
            if getattr(message, "id", None)
        ]

        return Command(
            update=await state.asanitize_update(
                ctx=ctx,
                update={
                    "agents_thoughts": [agent_thought, agent_outcome],
                    EXECUTOR_MESSAGES_KEY: [*superseded, tool_message],
                },
                agent="executor",
            ),
        )

    return take_screenshot


take_screenshot_wrapper = ToolWrapper(
    tool_fn_getter=get_take_screenshot_tool,
    on_success_fn=lambda size_bytes: (
        f"Screenshot attached ({size_bytes // 1024} KB JPEG). It supplements the "
        "element table and the OCR text."
    ),
    on_failure_fn=lambda reason: f"Failed to take a screenshot: {reason}.",
)
