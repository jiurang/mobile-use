"""从「做过的一串动作」里提取出宏的步骤。

**同一个提取器，两个用场**：

* 离线：把 manager 里某次执行的 trace 编成宏（``scripts/compile_macro.py``）。
* 在线：Executor 刚做完一个子目标，把**自己历史里实际执行过的动作**存成宏
  （``tools/mobile/save_macro.py``）。

为什么在线也用代码提取，而不是让模型把步骤写出来
------------------------------------------------
步骤是**事实**：哪个动作、按什么选择器点的、点了成没成。让模型复述一遍，
它会漏步、会改顺序、会编出没执行过的动作 —— 而宏是**要照着执行**的东西，
编错了就是在真机上乱点。

所以分工和 Reporter 一样（outcome/evidence 由代码填，LLM 只写 summary）：

* **步骤**由代码从历史里提取（含执行时解析出的 resource_id 和上下文指纹）
* **叫什么、什么时候用**由模型写（这是语义，代码写不出来）

消息的形状
----------
两种都要认：

* LangChain 的 ``AIMessage`` / ``ToolMessage``（执行器手里的本地历史）
* manager 存的 dict（``{"role": ..., "blocks": [...], "toolCalls": [...]}``，trace 里）
"""

from __future__ import annotations

import json
import re
from typing import Any

#: 工具结果文字里的「解析出来的 resource_id」。
RESOURCE_RE = re.compile(r"resource_id='([^']+)'")
#: 工具结果文字里的「按 index/坐标点的」（没有 id ⇒ 跨运行不可靠）。
INDEX_RE = re.compile(r"index=(\d+)")
#: 工具结果文字里的坐标。
COORD_RE = re.compile(r"coordinates \((\d+),\s*(\d+)\)")

#: executor 的工具名 → 宏里的工具名（宏是**设备侧**那套名字）。
TOOL_ALIASES = {
    "tap": "tap",
    "long_press_on": "long_press",
    "swipe_percentages": "swipe",
    "swipe_coordinates": "swipe",
    "focus_and_input_text": "input_text",
    "focus_and_clear_text": "clear_text",
    "erase_one_char": "erase_one_char",
    "press_key": "press_key",
    "back": "back",
    "wait_for_delay": "wait",
    "launch_app": "launch_app",
    "stop_app": "stop_app",
}


def _role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "")
    kind = getattr(message, "type", None) or ""
    if kind in ("ai", "AIMessage"):
        return "assistant"
    if kind in ("tool", "ToolMessage"):
        return "tool"
    if kind in ("human", "HumanMessage"):
        return "user"
    return str(kind)


def _tool_calls(message: Any) -> list[dict[str, Any]]:
    """一条消息里的工具调用（两种形状都认）。"""
    if isinstance(message, dict):
        calls = message.get("toolCalls") or message.get("tool_calls") or []
        out = []
        for call in calls:
            if isinstance(call, dict):
                out.append({"name": call.get("name"), "args": call.get("args") or {}})
        return out
    out = []
    for call in getattr(message, "tool_calls", None) or []:
        args = getattr(call, "args", None)
        if args is None and isinstance(call, dict):
            args = call.get("args")
        name = getattr(call, "name", None)
        if name is None and isinstance(call, dict):
            name = call.get("name")
        out.append({"name": name, "args": args or {}})
    return out


def _text(message: Any) -> str:
    """一条消息的正文（两种形状都认）。"""
    if isinstance(message, dict):
        parts = []
        for block in message.get("blocks") or []:
            if block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        if parts:
            return " ".join(parts)
        return str(message.get("content") or "")
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return " ".join(parts)
    return ""


def _actions_of(args: Any) -> dict[str, Any] | None:
    """从工具调用的 args 里取那一个动作。

    两种形状：

    * 执行器本地历史（**改写后**）：``{"actions": [{"action": "tap", ...}]}``
    * manager trace 里的模型原始输出：``{"action": "tap", ...}``
    """
    if not isinstance(args, dict):
        return None
    actions = args.get("actions")
    if isinstance(actions, list) and actions:
        first = actions[0]
        return dict(first) if isinstance(first, dict) else None
    if "action" in args:
        return dict(args)
    return None


def _to_device_step(action: dict[str, Any], result_text: str) -> dict[str, Any] | None:
    """一个动作 → 宏里的一步（设备侧工具名 + 稳定选择器 + 降级链）。"""
    name = str(action.get("action") or "")
    tool = TOOL_ALIASES.get(name)
    if tool is None:
        return None  # scratchpad 那几个不录

    # ---- 稳定选择器：先看动作自带的，再看执行结果里解析出来的 ----
    # ⚠️ 顺序不能反：模型报的是 ``index``，**完整的 id 是执行时解析出来的**，
    # 只在结果文字里。反过来写会把「本来能录的点击」当成「没有稳定选择器」丢掉。
    resource_id: str | None = None
    target = action.get("target")
    if isinstance(target, dict):
        candidate = target.get("resource_id")
        if isinstance(candidate, str) and candidate:
            resource_id = candidate
    if resource_id is None:
        found = RESOURCE_RE.search(result_text)
        if found:
            resource_id = found.group(1)

    args: dict[str, Any] = {}
    if resource_id:
        args["target"] = {"resource_id": resource_id}

    if name == "swipe_percentages":
        args.update(
            {
                "use_percentages": True,
                "start_x_percent": action.get("start_x_percent", 50),
                "start_y_percent": action.get("start_y_percent", 70),
                "end_x_percent": action.get("end_x_percent", 50),
                "end_y_percent": action.get("end_y_percent", 35),
            }
        )
    elif name == "swipe_coordinates":
        args.update(
            {
                "use_percentages": False,
                "start_x": action.get("start_x", 0),
                "start_y": action.get("start_y", 0),
                "end_x": action.get("end_x", 0),
                "end_y": action.get("end_y", 0),
            }
        )
    elif name == "focus_and_input_text":
        args["text"] = action.get("text", "")
    elif name == "wait_for_delay":
        args["seconds"] = round(float(action.get("time_in_ms", 1000)) / 1000.0, 3)
    elif name == "press_key":
        args["key"] = action.get("key") or "back"

    if tool in ("tap", "long_press", "input_text", "clear_text") and "target" not in args:
        # 没有稳定选择器 ⇒ 这一步跨运行不可靠，**不录**。
        # 宁可宏少一步，也不要它在别的屏上乱点。
        return None

    step: dict[str, Any] = {"tool": tool, "args": args}
    fallback: dict[str, Any] = {}
    if resource_id:
        fallback["selector"] = resource_id
    if "ctx:" in result_text:
        from minitap.mobile_use.utils.screen_fingerprint import parse_context

        context = parse_context(result_text)
        if context:
            fallback["context"] = context
    if fallback:
        step["fallback"] = fallback
    return step


def steps_from_history(messages: list[Any]) -> list[dict[str, Any]]:
    """从一段历史（执行器的本地历史，或 trace 里的一轮）提取宏的步骤。

    配对方式：assistant 的工具调用 → 紧随其后的 tool 消息（执行结果）。
    动作的**稳定选择器**和**上下文指纹**都来自执行结果文字。
    """
    steps: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for message in messages or []:
        role = _role(message)
        if role == "assistant":
            for call in _tool_calls(message):
                action = _actions_of(call.get("args"))
                if action is not None:
                    pending = action
        elif role == "tool" and pending is not None:
            step = _to_device_step(pending, _text(message))
            if step is not None:
                steps.append(step)
            pending = None
    return steps


def steps_to_json(steps: list[dict[str, Any]]) -> str:
    return json.dumps(steps, ensure_ascii=False)
