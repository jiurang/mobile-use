"""Executor 的工具执行节点：串行执行，一个失败就中止后面的。

上游 ``ToolNode`` 并行跑所有 tool call，也没有「一个失败就中止后面」的公开钩子，
而 Executor 依赖这两条语义：

- 串行。元素表的编号只在那一屏上成立。一轮执行多个动作，第二个动作的
  ``Target.index`` 指向的还是第一个动作执行前的屏，屏幕已经被改过了。
- 一个失败就中止。前一个动作失败了，后面的动作多半建立在错误的前提上。

这两条语义的承载者不在 LLM 里，在 ``ExecutorToolNode.__func`` 里面；prompt 里
写的是软约束，这里是硬兜底。

这个类依赖 langgraph-prebuilt 的私有 API（``_parse_input`` / ``_extract_state`` /
``_run_one`` / ``_arun_one`` / ``_combine_tool_outputs``）。它们的签名在 minor
版本之间会变，所以 ``pyproject.toml`` 里把 langgraph 锁在 ``>=1.2.0,<1.3.0``。
升级前先 diff 上游的 ``ToolNode._func``，并跑
``tests/mobile_use/test_executor_tool_node.py``。

顺带做的一件事：给「动过屏幕」的工具结果打上动作时刻（``last_action_at``），
让 contextor 能判断「要不要先等屏幕沉降」。打点放在这里，而不是 13 个工具文件
里各写一遍，因为工具们各有各的返回形状，在执行完工具之后统一处理，才是唯一
能保证「每个动屏动作都被记到」的位置。
"""

import asyncio
import copy
import time
from typing import Any, override

from langchain_core.messages import AnyMessage, ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel

from minitap.mobile_use.services.telemetry import telemetry
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.message_content import content_to_text

logger = get_logger(__name__)

# 会改变屏幕的工具。只有这些工具之后才需要等屏幕沉降（见
# ``PerceptionConfig.settle_delay_ms``）：等笔记、读笔记、刻意等待都不算。
#
# ``wait_for_delay`` 不在列表里：等待本身就是在给屏幕时间，额外再等一遍是双倍。
DEVICE_CHANGING_TOOLS = frozenset(
    {
        "tap",
        "long_press_on",
        "swipe",
        "swipe_coordinates",
        "swipe_percentages",
        "press_key",
        "back",
        "launch_app",
        "stop_app",
        "open_link",
        "focus_and_input_text",
        "focus_and_clear_text",
        "erase_one_char",
    }
)


def _stamp_action(output: Any, tool_name: str) -> None:
    """给「动过屏幕」的工具结果打上动作时刻。

    打时间戳这件事放在这里，而不是 13 个工具文件里各写一遍：工具们各有各的
    返回形状（Command / ToolMessage / 列表），在执行完工具之后统一处理，
    才是唯一能保证「每个动屏动作都被记到」的位置。
    """
    if tool_name not in DEVICE_CHANGING_TOOLS:
        return

    now = time.time()
    candidates = output if isinstance(output, list) else [output]
    stamped = False
    for item in candidates:
        if isinstance(item, Command) and isinstance(item.update, dict):
            item.update["last_action_at"] = now
            stamped = True
    if not stamped:
        logger.debug(f"[executor] {tool_name} 未产生 state update，动作时刻没记上")


class ExecutorToolNode(ToolNode):
    """逐个执行 tool call，一个出错就中止剩下的。

    上游 ToolNode 没有「串行 + 出错即中止」的公开钩子（它并行跑所有 call），
    所以这里覆写 ``_func`` / ``_afunc``，并依赖 ToolNode 的私有方法
    （``_parse_input``、``_extract_state``、``_run_one``、``_arun_one``、
    ``_combine_tool_outputs``）。这些方法的签名在 minor 版本之间会变，
    所以 pyproject.toml 里对 langgraph 做了 minor 版本上限。升版本前先 diff
    上游的 ``ToolNode._func``，并跑 tests/mobile_use/test_executor_tool_node.py。
    """

    def __init__(self, tools, messages_key: str, trace_id: str | None = None) -> None:
        super().__init__(tools=tools, messages_key=messages_key)
        self._trace_id = trace_id

    @override
    async def _afunc(
        self,
        input: list[AnyMessage] | dict[str, Any] | BaseModel,
        config: RunnableConfig,
        runtime: Runtime,
    ) -> Any:
        return await self.__func(is_async=True, input=input, config=config, runtime=runtime)

    @override
    def _func(
        self,
        input: list[AnyMessage] | dict[str, Any] | BaseModel,
        config: RunnableConfig,
        runtime: Runtime,
    ) -> Any:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.__func(is_async=False, input=input, config=config, runtime=runtime)
        )

    def _build_tool_runtime(
        self,
        call: ToolCall,
        input: list[AnyMessage] | dict[str, Any] | BaseModel,
        config: RunnableConfig,
        runtime: Runtime,
    ) -> Any:
        state = self._extract_state(input, config)
        return ToolRuntime(
            state=state,
            tool_call_id=call["id"],
            config=config,
            context=runtime.context,
            store=runtime.store,
            stream_writer=runtime.stream_writer,
            tools=list(self.tools_by_name.values()),
            execution_info=runtime.execution_info,
            server_info=runtime.server_info,
        )

    async def __func(
        self,
        is_async: bool,
        input: list[AnyMessage] | dict[str, Any] | BaseModel,
        config: RunnableConfig,
        runtime: Runtime,
    ) -> Any:
        tool_calls, input_type = self._parse_input(input)
        outputs: list[Command | ToolMessage | list[Command | ToolMessage]] = []
        failed = False
        for call in tool_calls:
            tool_runtime = self._build_tool_runtime(call, input, config, runtime)
            if failed:
                output = self._get_erroneous_command(
                    call=call,
                    message="Aborted: a previous tool call failed!",
                )
            else:
                if is_async:
                    output = await self._arun_one(call, input_type, tool_runtime)
                else:
                    output = self._run_one(call, input_type, tool_runtime)
                # None 表示工具返回了预期之外的类型，当成失败处理。
                failed = self._has_tool_call_failed(call, output)
                if failed is None:
                    output = self._get_erroneous_command(
                        call=call,
                        message=f"Unexpected tool output type: {type(output)}",
                    )
                    failed = True
                else:
                    # 动过屏幕的动作要记时刻：contextor 靠它决定「要不要先等屏幕沉降」
                    _stamp_action(output, call["name"])

            call_without_state = copy.deepcopy(call)
            if "args" in call_without_state and "state" in call_without_state["args"]:
                del call_without_state["args"]["state"]
            if failed:
                error_msg = ""
                try:
                    if isinstance(output, ToolMessage):
                        # 多模态内容（截图那种）要压成文本再记日志：
                        # 直接 str() 一个块列表会打出几十 KB 的 base64 字面量。
                        error_msg = content_to_text(output.content)
                    elif isinstance(output, Command):
                        error_msg = content_to_text(self._get_tool_message(output).content)
                    elif tool_message := self._find_tool_message(output):
                        error_msg = content_to_text(tool_message.content)
                except Exception as e:
                    # 提取错误详情失败不能盖掉真正的错误：这里只是拿不到
                    # 更详细的说明，工具本身已经失败了（`failed` 为真）。
                    # 记一条 debug 保住现场，否则「提取失败」和「工具失败」
                    # 在日志里长得一模一样。
                    logger.debug(f"[executor] Could not extract error details: {e}")
                    error_msg = "Could not extract error details"

                logger.info(f"❌ Tool call failed: {call_without_state}")
                logger.info(f"   Error: {error_msg}")

                if self._trace_id:
                    telemetry.capture_executor_action(
                        task_id=self._trace_id,
                        tool_name=call["name"],
                        success=False,
                        error=str(error_msg)[:500] if error_msg else None,
                    )
            else:
                logger.info("✅ Tool call succeeded: " + str(call_without_state))

                if self._trace_id:
                    telemetry.capture_executor_action(
                        task_id=self._trace_id,
                        tool_name=call["name"],
                        success=True,
                    )

            outputs.append(output)
        return self._combine_tool_outputs(outputs, input_type)  # type: ignore

    def _has_tool_call_failed(
        self,
        call: ToolCall,
        output: ToolMessage | Command | list[Command | ToolMessage],
    ) -> bool | None:
        # langgraph-prebuilt >= 1.1 lets a tool return a list of ToolMessage/Command, with
        # exactly one ToolMessage for the call (others are plain state updates): judge on it.
        if isinstance(output, list):
            tool_message = self._find_tool_message(output)
            return None if tool_message is None else tool_message.status == "error"
        if isinstance(output, ToolMessage):
            return output.status == "error"
        if isinstance(output, Command):
            output_msg = self._get_tool_message(output)
            return output_msg.status == "error"
        return None

    def _find_tool_message(self, outputs: list[Command | ToolMessage]) -> ToolMessage | None:
        for item in outputs:
            if isinstance(item, ToolMessage):
                return item
            if isinstance(item, Command) and isinstance(item.update, dict):
                msg = item.update.get(self._messages_key)
                if isinstance(msg, ToolMessage):
                    return msg
                if isinstance(msg, list) and msg and isinstance(msg[-1], ToolMessage):
                    return msg[-1]
        return None

    def _get_erroneous_command(self, call: ToolCall, message: str) -> Command:
        tool_message = ToolMessage(
            name=call["name"], tool_call_id=call["id"], content=message, status="error"
        )
        return Command(update={self._messages_key: [tool_message]})

    def _get_tool_message(self, cmd: Command) -> ToolMessage:
        if isinstance(cmd.update, dict):
            msg = cmd.update.get(self._messages_key)
            if isinstance(msg, list):
                if len(msg) == 0:
                    raise ValueError("No messages found in command update")
                if not isinstance(msg[-1], ToolMessage):
                    raise ValueError("Last message in command update is not a tool message")
                return msg[-1]
            elif isinstance(msg, ToolMessage):
                return msg
            elif msg is None:
                raise ValueError(f"Missing '{self._messages_key}' in command update")
            raise ValueError(f"Unexpected message type in command update: {type(msg)}")
        raise ValueError("Command update is not a dict")
