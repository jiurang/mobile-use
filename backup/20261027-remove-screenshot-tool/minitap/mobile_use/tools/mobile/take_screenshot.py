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
* 图在**prompt 末尾**（元素表旁边），模型看得见「这是我刚要的那张图」。

⚠️ **图和元素表是同一屏。** 工具在 contextor 抓完屏之后才跑，中间没有任何动作
改动过屏幕 ⇒ 图与元素表描述的是同一屏，编号契约不受影响。

图为什么不挂���工具结果上
------------------------
**图片块会进 provider 的前缀缓存键。** 真 API 实测（2026-10-26，
MiniMax-M3.1-Flash，``api.minimaxi.com``；探针
``mobile-manager/scripts/probe_cache_image_role.py``）：把请求**中段**的图片
删掉、或换成另一张，命中量立刻掉到「图之前」的位置（8,639 → 7,193 /
7,207 token）。

⇒ 图挂在局部历史里 = 被钉在历史中段；下一次拍图要「顶替」它时那个位置的字节
变了 ⇒ **前缀从历史中段起整段作废**。真机实测（mobile-manager execution 893）：
14 次顶替让 47,181 个本可命中的 token 变成满价重算，缓存率从 82% 掉到 41%。

⇒ 图另存（``EXECUTOR_SCREENSHOT_KEY``），装配 prompt 时放到**最后一条消息**
（元素表旁边 —— 那本来就是每轮都变的易变区）⇒ 局部历史一个字节都不动，
前缀永远稳定。

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

from minitap.mobile_use.constants import EXECUTOR_MESSAGES_KEY, EXECUTOR_SCREENSHOT_KEY
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.controllers.unified_controller import UnifiedMobileController
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.prompt_language import PromptLanguage, pick
from minitap.mobile_use.tools.tool_wrapper import ToolWrapper
from minitap.mobile_use.utils.image_compress import compress_screenshot_base64
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.message_content import screenshot_channel_label

logger = get_logger(__name__)

#: 图的 JPEG 质量。与 ``utils/image_compress.py`` 的默认一致（q50 实测可读）。
SCREENSHOT_JPEG_QUALITY = 50


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


def build_screenshot_content(
    *, language: PromptLanguage, jpeg_bytes: int
) -> list[str | dict[str, Any]]:
    """工具结果的 content：**一行事实**（图不在这里，图注也不在）。

    ⚠️ **图为什么不挂在这条工具结果上**
    --------------------------------
    图片块**会进 provider 的前缀缓存键**（真 API 实测，MiniMax-M3.1-Flash：
    把请求中段的图片删掉或换一张，命中量立刻掉到「图之前」的位置）。

    ⇒ 图挂在局部历史里 = 被钉在历史中段；下一次拍图要「顶替」它时那个位置的
    字节就变了 ⇒ **前缀从历史中段起整段作废**。
    真机实测（mobile-manager execution 893）：14 次顶替让 47,181 个本可命中的
    token 变成满价重算，缓存率从 82% 掉到 41%。

    ⇒ 改成：工具结果只留**文字事实**，图另存（``EXECUTOR_SCREENSHOT_KEY``），
    装配 prompt 时作为**独立的一条消息**（标题「图片感知」）放在最末。
    局部历史一个字节都不动 ⇒ 前缀永远稳定。

    ⚠️ **为什么也没有图注**
    ----------------------
    「图是补充、寻址走元素表的 ``[i]``、一屏一图就够」这套契约写在**静态
    提示词**里（``agents/executor/executor.md``）—— 模型每轮都读得到。逐轮
    重说一遍只是重复，还把一段易变文字塞进局部历史。

    ⇒ 工具结果只回答两件事：**拍到了没有 / 多大**；图在哪儿由这一行指路。
    """
    label = screenshot_channel_label(language=language)
    text = pick(
        f"Screenshot taken: image/jpeg, {jpeg_bytes:,} bytes. The picture is in "
        f'the "{label}" block at the end of this request.',
        f"已截图：image/jpeg，{jpeg_bytes:,} 字节。图像在本次请求末尾的「{label}」段。",
        language=language,
    )
    return [{"type": "text", "text": text}]


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

        The picture itself arrives in the next request, as the last message,
        under the heading "Image perception" (「图片感知」) - a channel of its
        own, separate from the element table and the OCR text.
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
        # 工具结果**只有文字** —— 图不进局部历史（见 build_screenshot_content 的
        # 注释：图片块进 provider 的缓存键，写进历史会把前缀打断）。
        tool_message = ToolMessage(
            tool_call_id=tool_call_id,
            content=build_screenshot_content(
                language=ctx.prompt_language,
                jpeg_bytes=jpeg_bytes,
            ),
            status="success",
        )

        return Command(
            update=await state.asanitize_update(
                ctx=ctx,
                update={
                    "agents_thoughts": [agent_thought, agent_outcome],
                    # 历史里只**追加**这条工具结果 —— 一个字节都不改写。
                    EXECUTOR_MESSAGES_KEY: [tool_message],
                    # 图另存：只留最新一张（覆盖），装配 prompt 时放到末尾。
                    EXECUTOR_SCREENSHOT_KEY: image_b64,
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
