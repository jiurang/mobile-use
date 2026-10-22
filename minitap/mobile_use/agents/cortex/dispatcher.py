"""把 Cortex 的 decisions JSON **确定性地**翻译成 tool_calls，跳过 Executor 的 LLM 往返。

为什么要做这个
--------------
一条动作在 Cortex 的输出里已经是一个完备、无歧义的动作描述了：

    [{"action": "tap", "target": {"resource_id": "com.x:id/send", ...}}]

`action` 字段**本身就是工具名**。所以 Executor 那次 LLM 调用做的事情是
「JSON → tool_calls 的机械转写」——它不重新决策，Cortex 已经决策完了
（executor.md 第一条规则就是 "Don't reason about strategy - just execute what
Cortex decided"）。这一轮往返是纯开销：一次完整的 prefill + 一次完整的生成。

**Executor 里真正承载语义的那部分不在 LLM 里**：
- 「按返回顺序串行执行」+「一个失败就中止后面的」实现在
  `ExecutorToolNode.__func`（串行 for 循环 + failed 标志），
  不是在 Executor 的 prompt 里。跳过 LLM 不会丢这两条语义。
- `agent_thought` 由 Cortex 的 `decisions_reason` 提供，信息没有丢。

失败模式：直连是**快路径**，不是唯一路径
--------------------------------------
`dispatch_decisions` 任何一步不确定就返回 `None`，调用方退回原来的
Executor LLM。也就是说最坏情况就是今天的行为（多一次往返），
绝不会因为 Cortex 的 JSON 写法不标准就把整轮搞挂。
"""

from __future__ import annotations

import json
import re
import uuid

from langchain_core.messages import ToolCall
from langchain_core.tools import BaseTool

from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

# Cortex 可能把工具名放在这几个 key 里（prompt 里用的是 "action"）
_NAME_KEYS = ("action", "tool", "tool_name", "name")

# 由 LangChain 注入、不出现在 tool.args 里、也不该由模型提供的参数
_INJECTED_ARGS = frozenset({"tool_call_id", "state", "store", "runtime", "config"})


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()

def _extract_json_array(text: str) -> str | None:
    """取出第一个括号配平的 JSON 数组。

    Cortex 的 `decisions` 本身已经是个 JSON 数组**字符串**，但模型经常在
    外面裹一层解释文字（"好的，我来…"）。用括号配平扫描而不是正则：
    动作对象里可能有嵌套数组和转义引号，贪婪正则会截断。
    """
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_actions(decisions: str | None) -> list[dict] | None:
    """把 decisions 字符串解析成动作列表。解析不出来就返回 None。"""
    if not decisions or not decisions.strip():
        return None

    candidates = [_strip_fences(decisions)]
    extracted = _extract_json_array(candidates[0])
    if extracted and extracted != candidates[0]:
        candidates.append(extracted)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
            return parsed
    return None


def _required_args(tool: BaseTool) -> set[str]:
    """模型可见的必填参数（排除注入参数）。

    不能直接用 `tool.args["required"]`：langchain 的 `@tool` 实测不填它。
    也不能用 `tool.args_schema.model_validate()`：注入参数在那个 model 里
    也是必填，会永远报错。只能从 model_fields 里排除掉注入名。
    """
    schema = tool.args_schema
    model_fields = getattr(schema, "model_fields", None)
    if not model_fields:
        return set()
    return {
        name
        for name, field in model_fields.items()
        if field.is_required() and name not in _INJECTED_ARGS
    }


def _looks_valid(action: dict, tool: BaseTool, thought: str) -> dict | None:
    """校验单个动作能否安全直发；不行返回 None。

    校验两件事，方向相反：
    - 必填参数齐不齐（缺了就交给 Executor LLM 去补）
    - 有没有模型压根不认识的参数名（Cortex 幻觉出来的字段）

    另外顺手拦一类最常见的形态错误：声明为对象的参数（比如 `target`）
    传了字符串或数字 —— 这种在 ToolNode 里报错太晚，直接退回 LLM 更划算。
    """
    kwargs, _reason = _looks_valid_with_reason(action, tool, thought)
    return kwargs


def _looks_valid_with_reason(
    action: dict, tool: BaseTool, thought: str
) -> tuple[dict | None, str | None]:
    """和 `_looks_valid` 一样，但**同时返回给模型看的拒绝理由**。

    理由要能直接塞进下一轮 prompt —— 模型得知道**自己错在哪**，
    光说「不合法」它只会换个方式再犯一次。
    """
    kwargs = {k: v for k, v in action.items() if k not in _NAME_KEYS}
    kwargs["agent_thought"] = kwargs.get("agent_thought") or thought

    visible = tool.args
    for key in kwargs:
        if key not in visible:
            logger.info(f"Dispatch: unknown arg '{key}' for tool '{tool.name}'")
            return None, (
                f"'{key}' is not a valid argument for '{tool.name}'; "
                f"its arguments are: {', '.join(sorted(visible))}"
            )

    missing = _required_args(tool) - set(kwargs)
    if missing:
        logger.info(f"Dispatch: tool '{tool.name}' missing {sorted(missing)}")
        return None, (
            f"'{tool.name}' is missing required argument(s): {', '.join(sorted(missing))}"
        )

    for key, value in kwargs.items():
        prop = visible.get(key) or {}
        if prop.get("type") == "object" and not isinstance(value, dict):
            logger.info(f"Dispatch: arg '{key}' of '{tool.name}' should be an object")
            return None, (
                f"'{key}' of '{tool.name}' must be an object, "
                f"e.g. {key}: {{\"index\": 7}}"
            )

    return kwargs, None


def dispatch_decisions(
    decisions: str | None,
    default_thought: str | None,
    tools: list[BaseTool],
) -> list[ToolCall] | None:
    """Cortex decisions → tool_calls。

    Args:
        decisions: Cortex 的 `structured_decisions`（JSON 数组字符串）。
        default_thought: 兜底的 agent_thought（用 Cortex 的 decisions_reason）。
        tools: 当前会话可用的工具。

    Returns:
        成功时返回按原顺序排列的 tool_calls；**任何一步不确定都返回 None**，
        调用方必须退回 Executor 的 LLM 路径。空列表也返回 None
        （没有动作可发，和「解析失败」在调用方的处理方式是一样的）。
    """
    calls, _reason = dispatch_decisions_with_reason(decisions, default_thought, tools)
    return calls


def dispatch_decisions_with_reason(
    decisions: str | None,
    default_thought: str | None,
    tools: list[BaseTool],
) -> tuple[list[ToolCall] | None, str | None]:
    """和 `dispatch_decisions` 一样，但**同时返回拒绝理由**。

    为什么需要它
    ------------
    旧架构里 dispatcher 失败 → 退回 Executor 的 LLM **重新转写**同一批
    decisions，那是有意义的（转写器重试可能就写对了）。

    新架构里 Executor 的 LLM **就是决策者**（它自己决定动作），
    所以「退回 LLM」退回来的还是同一个决策者 —— 它大概率输出**同样的坏动作**，
    因为它没看到自己错在哪。这会变成死循环。

    ⇒ 正确的做法是把**拒绝理由**回灌给 Executor，让它带着
    「你刚才说的 `teleport` 不是个工具」重新决策。

    Returns:
        `(tool_calls, reason)`。成功时 reason 为 None；
        失败时 tool_calls 为 None、reason 是给模型看的一句话。
    """
    actions = parse_actions(decisions)
    if not actions:
        # ⚠️ **空 actions 和「解析失败」是两回事**，回灌给模型的理由必须不同：
        #   * 空 actions —— 模型**没决定任何动作**（它可能卡住了 / 上一轮失败了）。
        #     说「格式错」会让它去改格式，而它本来就没格式问题。
        #   * 解析失败 —— 模型输出的东西根本不是 JSON 数组。
        if decisions is not None and decisions.strip() in ("[]", "[ ]"):
            return None, (
                "you returned an empty action list — decide one concrete action to take, "
                "or set done=true if the goal is already reached"
            )
        return None, (
            "the decisions could not be parsed as a JSON array of actions; "
            "see the Tools table for the expected shape"
        )

    thought = (default_thought or "").strip()
    if not thought:
        # agent_thought 是所有工具的必填参数，没有它发不出去。
        logger.info("Dispatch: no agent_thought available (decisions_reason empty)")
        return None, "agent_thought is required for every action"

    by_name = {tool.name: tool for tool in tools}
    calls: list[ToolCall] = []
    for index, action in enumerate(actions):
        tool_name = next(
            (action[key] for key in _NAME_KEYS if isinstance(action.get(key), str)),
            None,
        )
        tool = by_name.get(tool_name) if tool_name else None
        if tool is None:
            logger.info(f"Dispatch: unknown tool {tool_name!r} (action #{index})")
            return None, (
                f"'{tool_name}' is not a known tool; "
                f"available tools are: {', '.join(sorted(by_name))}"
            )

        kwargs, reason = _looks_valid_with_reason(action, tool, thought)
        if kwargs is None:
            return None, reason

        calls.append(
            ToolCall(name=tool.name, args=kwargs, id=f"dispatch_{uuid.uuid4().hex[:12]}")
        )

    logger.info(
        f"Dispatch: direct execution of {len(calls)} action(s) "
        f"[{', '.join(c['name'] for c in calls)}] — skipping executor LLM"
    )
    return calls, None
