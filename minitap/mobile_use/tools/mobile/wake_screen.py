"""唤醒屏幕的工具。

为什么需要它
------------
屏幕一黑（自动息屏超时、按电源键），会发生三件事：

1. 截图是**全黑**的；
2. uiautomator 的层级树**不是空的** —— 2026-10-21 真机实测（Redmi Turbo 4 /
   Android 16）熄屏后仍然返回 **19 个元素**，全是 systemui 的通知面板/状态栏
   残留。也就是说「层级树为空」**不是**熄屏的判据（它被实测证伪过）；
3. 于是 agent 看不到任何业务内容 —— 它会以为「屏上什么都没有」，于是在黑屏上
   反复尝试、原地打转。

真机实测（2026-10-21，Redmi Turbo 4 / Android 16）：手机的息屏超时是
30 分钟，一个任务跑到 30 分钟后就开始在黑屏上打转，**步数从 30 涨到 51 步
还没跑完**。这个症状（步数暴涨、没有任何报错）极易被误判成「代码回归」。

现在的做法是：harness **读出**屏幕是不是灭的，**告诉模型**，模型调这个工具
把屏幕点亮。

它做什么
--------
- ``KEYCODE_WAKEUP`` 点亮屏幕（息屏时普通的 tap / swipe 是**没用的** ——
  事件发给了一个没显示的屏幕）。
- ``wm dismiss-keyguard`` 划掉无密码锁屏。**有 PIN 划不掉**，这时只能靠人解锁；
  工具会如实回报，不会假装成功。

**Android 专用**：iOS 侧没有可用的唤醒接口（见 `ios_controller.py`），
调用会明确回报失败。
"""

from typing import Annotated

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


def get_wake_screen_tool(ctx: MobileUseContext):
    @tool
    async def wake_screen(
        agent_thought: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[State, InjectedState],
    ) -> Command:
        """Wake up the device screen and dismiss the keyless lock screen.

        Use this ONLY when the screen is reported as off/blank: you cannot see
        anything while the screen is off, so every action you take would be
        blind. After waking up, the next turn will show you the real screen.

        Android only. A PIN-protected screen cannot be dismissed
        programmatically.
        """
        controller = UnifiedMobileController(ctx)
        woke = await controller.wake_up()
        # 唤醒之后**再读一次**状态：有 PIN 锁屏时划不掉，屏幕仍然是灭的，
        # 这时必须如实回报，不能让模型以为已经亮了。
        state_after = await controller.get_screen_power_state()
        if state_after == "on":
            agent_outcome = wake_screen_wrapper.on_success_fn()
        elif state_after == "off":
            agent_outcome = wake_screen_wrapper.on_failure_fn(
                "the screen is still off (a PIN lock screen cannot be dismissed "
                "programmatically - a human has to unlock it)"
            )
        else:
            agent_outcome = wake_screen_wrapper.on_failure_fn(
                "the screen power state could not be read after waking up"
                if not woke
                else "the screen power state is unknown after waking up"
            )

        tool_message = ToolMessage(
            tool_call_id=tool_call_id,
            content=agent_outcome,
            additional_kwargs={} if state_after == "on" else {"error": agent_outcome},
            status="success" if state_after == "on" else "error",
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

    return wake_screen


wake_screen_wrapper = ToolWrapper(
    tool_fn_getter=get_wake_screen_tool,
    on_success_fn=lambda: "The screen is on now. Read the screen again before acting.",
    on_failure_fn=lambda reason: f"Failed to wake up the screen: {reason}.",
)
