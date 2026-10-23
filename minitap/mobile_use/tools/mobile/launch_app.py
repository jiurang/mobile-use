from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from minitap.mobile_use.agents.hopper.hopper import HopperOutput, hopper
from minitap.mobile_use.constants import EXECUTOR_MESSAGES_KEY
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.controllers.platform_specific_commands_controller import list_packages_async
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.tools.tool_wrapper import ToolWrapper
from minitap.mobile_use.utils.app_launch_utils import launch_app_with_retries
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


async def find_package(ctx: MobileUseContext, app_name: str) -> str | None:
    """
    Finds the package name for a given application name.
    Returns None if package not found or on error.
    """
    all_packages = await list_packages_async(ctx=ctx)
    try:
        hopper_output: HopperOutput = await hopper(
            ctx=ctx,
            request=f"I'm looking for the package name of the following app: '{app_name}'",
            data=all_packages,
        )
        if not hopper_output.found:
            return None
        return hopper_output.output
    except Exception as e:
        logger.error(f"Failed to find package for '{app_name}': {e}")
        return None


def get_launch_app_tool(ctx: MobileUseContext):
    @tool
    async def launch_app(
        app_name: str,
        agent_thought: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[State, InjectedState],
    ) -> Command:
        """
        Finds and launches an application on the device using its natural language name.
        """
        package_name = await find_package(ctx=ctx, app_name=app_name)

        if not package_name:
            tool_message = ToolMessage(
                tool_call_id=tool_call_id,
                content=launch_app_wrapper.on_failure_fn(app_name, "Package not found."),
                status="error",
            )
        else:
            # ``detail`` 的含义随 ``ok`` 变（见 launch_app_with_retries 的文档）：
            # 失败时它是**原因**，成功时它是**保留意见**（目前只有「没能确认切到
            # 前台」一种）。**两种都要让模型看见** —— 少了它，模型只能把
            # 「启动成了但没确认」当成「失败了」，然后重发、然后重验、然后转圈。
            ok, detail = await launch_app_with_retries(ctx=ctx, app_package=package_name)
            if ok:
                content = launch_app_wrapper.on_success_fn(app_name)
                if detail:
                    content = f"{content} {detail}"
            else:
                content = launch_app_wrapper.on_failure_fn(app_name, detail or "")
            tool_message = ToolMessage(
                tool_call_id=tool_call_id,
                content=content,
                additional_kwargs={} if ok else {"error": detail},
                status="success" if ok else "error",
            )

        return Command(
            update=await state.asanitize_update(
                ctx=ctx,
                update={
                    "agents_thoughts": [agent_thought, tool_message.content],
                    EXECUTOR_MESSAGES_KEY: [tool_message],
                },
                agent="executor",
            ),
        )

    return launch_app


launch_app_wrapper = ToolWrapper(
    tool_fn_getter=get_launch_app_tool,
    on_success_fn=lambda app_name: f"App '{app_name}' launched successfully.",
    on_failure_fn=lambda app_name, error: f"Failed to launch app '{app_name}': {error}",
)
