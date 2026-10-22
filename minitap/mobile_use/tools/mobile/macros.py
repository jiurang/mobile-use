"""宏（脚本）相关的三个工具：查、存、跑。

为什么需要它们
--------------
「每一步都让模型想」很贵：真机实测 executor 一次决策中位 5~8s，而一串
**已经决定过**的动作（本子目标刚做完的那一串）根本不用再想一遍。

所以：

* ``save_macro`` —— 把**刚做完的**这一串动作存成宏（下次同样的活直接跑）
* ``find_macros`` —— 找现成的宏
* ``run_macro`` —— 跑一个宏（**零次 LLM 调用**）

**步骤由代码提取，模型只写语义**
--------------------------------
``save_macro`` 只让模型给「叫什么」和「什么时候用」两句话；**步骤是代码从
执行器自己的历史里提取的**（含执行时解析出来的 resource_id 和上下文指纹）。

理由：步骤是**事实** —— 哪个动作、按什么选择器点的、点了成没成。让模型复述
一遍，它会漏步、会改顺序、会编出没执行过的动作；而宏是**要照着执行**的东西，
编错了就是在真机上乱点。

这和 Reporter 同一套分工：outcome/evidence 由代码填，LLM 只写 summary。
"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from minitap.mobile_use.constants import EXECUTOR_MESSAGES_KEY
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.macros.compiler import steps_from_history
from minitap.mobile_use.macros.store import MacroStore, MacroStoreError
from minitap.mobile_use.tools.tool_wrapper import ToolWrapper
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


def get_find_macros_tool(ctx: MobileUseContext) -> BaseTool:
    @tool
    async def find_macros(
        agent_thought: str,
        query: str = "",
        app_package: str = "",
        tool_call_id: Annotated[str, InjectedToolCallId] = None,  # type: ignore
        state: Annotated[State, InjectedState] = None,  # type: ignore
    ):
        """
        Find reusable macros (recorded action sequences) in the script library.

        A macro is a sequence of actions that already worked once. If one fits
        the current task, running it is far cheaper than deciding every step
        again. Search before starting a long repetitive flow.
        """
        store = MacroStore()
        try:
            candidates = store.find(query or "", app_package or None, 5)
        except MacroStoreError as exc:
            agent_outcome = find_macros_wrapper.on_failure_fn(str(exc))
            status = "error"
        else:
            if candidates:
                lines = [c.summary() for c in candidates]
                agent_outcome = find_macros_wrapper.on_success_fn("\n".join(lines))
            else:
                agent_outcome = find_macros_wrapper.on_success_fn("")
            status = "success"

        tool_message = ToolMessage(
            tool_call_id=tool_call_id,
            content=agent_outcome,
            status=status,
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

    return find_macros


def get_save_macro_tool(ctx: MobileUseContext) -> BaseTool:
    @tool
    async def save_macro(
        agent_thought: str,
        name: str,
        when_to_use: str,
        tool_call_id: Annotated[str, InjectedToolCallId] = None,  # type: ignore
        state: Annotated[State, InjectedState] = None,  # type: ignore
    ):
        """
        Save the actions you just performed as a reusable macro.

        You give it a name and when it applies; the steps are taken from what
        you actually did. Do this once a flow has worked end to end, so the
        same flow can be replayed later without deciding every step again.
        """
        steps = steps_from_history(list(state.executor_messages or []))
        if not steps:
            agent_outcome = save_macro_wrapper.on_failure_fn(
                "no replayable step in this attempt "
                "(steps need a stable resource_id; index-only taps are not recorded)"
            )
            status = "error"
        else:
            app_package = state.focused_app_info or ""
            payload = {
                "name": name,
                "when_to_use": when_to_use,
                "app_package": app_package,
                "device_serial": ctx.device.device_id,
                "steps": steps,
            }
            try:
                receipt = MacroStore().save(payload)
            except MacroStoreError as exc:
                agent_outcome = save_macro_wrapper.on_failure_fn(str(exc))
                status = "error"
            else:
                agent_outcome = save_macro_wrapper.on_success_fn(
                    receipt.get("id"), name, len(steps)
                )
                status = "success"

        tool_message = ToolMessage(
            tool_call_id=tool_call_id,
            content=agent_outcome,
            status=status,
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

    return save_macro


def get_run_macro_tool(ctx: MobileUseContext) -> BaseTool:
    @tool
    async def run_macro(
        agent_thought: str,
        macro_id: int,
        tool_call_id: Annotated[str, InjectedToolCallId] = None,  # type: ignore
        state: Annotated[State, InjectedState] = None,  # type: ignore
    ):
        """
        Run a macro from the script library.

        The macro is replayed step by step without asking the model anything,
        so it is much cheaper than doing the same flow by hand. It stops at the
        first step it cannot complete and tells you which one and why.
        """
        from minitap.mobile_use.macros.device_env import make_device_env
        from minitap.mobile_use.macros.model import Macro
        from minitap.mobile_use.macros.runner import run_macro as run_macro_steps

        store = MacroStore()
        try:
            raw = store.get(int(macro_id))
        except MacroStoreError as exc:
            agent_outcome = run_macro_wrapper.on_failure_fn(str(exc))
            status = "error"
        else:
            try:
                macro = Macro.from_dict(raw)
            except ValueError as exc:
                agent_outcome = run_macro_wrapper.on_failure_fn(f"bad macro: {exc}")
                status = "error"
            else:
                env = await make_device_env(ctx)
                result = await run_macro_steps(env, macro)
                store.record_usage(int(macro_id), result.ok)
                agent_outcome = run_macro_wrapper.on_success_fn(
                    result.summary(), result.degradations
                )
                status = "success" if result.ok else "error"

        tool_message = ToolMessage(
            tool_call_id=tool_call_id,
            content=agent_outcome,
            status=status,
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

    return run_macro


find_macros_wrapper = ToolWrapper(
    tool_fn_getter=get_find_macros_tool,
    on_success_fn=lambda lines: (
        f"Found {len(lines.splitlines())} macro(s):\n{lines}" if lines else "No macro matched."
    ),
    on_failure_fn=lambda reason: f"Macro search failed: {reason}",
)

save_macro_wrapper = ToolWrapper(
    tool_fn_getter=get_save_macro_tool,
    on_success_fn=lambda macro_id, name, count: (
        f"Saved macro '{name}' (id={macro_id}, {count} steps) to the script library."
    ),
    on_failure_fn=lambda reason: f"Could not save the macro: {reason}",
)

run_macro_wrapper = ToolWrapper(
    tool_fn_getter=get_run_macro_tool,
    on_success_fn=lambda summary, degradations: (
        summary + (f" Degradations: {json.dumps(degradations, ensure_ascii=False)}"
                   if degradations else "")
    ),
    on_failure_fn=lambda reason: f"Macro run failed: {reason}",
)
