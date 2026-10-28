"""Executor 的 prompt 组装。

消息顺序是这里最重要的一件事：每轮变的东西一律放末尾。

provider 的缓存是前缀匹配。任何每轮变的内容放在中间，它之后的全部字节都要重算。
屏幕元素表每轮都变（每轮重抓屏），所以它必须排在局部历史之后。实测的差别很大：
元素表排在历史前面，缓存命中率递减到 48%；排在历史后面，命中率递增到 93%。

文案全部由模板宏提供。``executor.md`` / ``executor-zh.md`` 是宏库：正文在
``system_message`` 宏里，其余每个运行时消息块（目标 / 失败 / 设备 / 元素表 /
OCR / 反馈）也各是一个宏。本模块只负责按顺序把宏的渲染结果放进消息列表，不再
自带任何提示词散文。唯一的例外是元素行 / OCR 行，它们的编号契约要和工具侧的
反解共用，不能拆两份。

不要 bind_tools 那 16 个设备工具。它们的 JSON schema 有 4,008 token，而且不参与
缓存键（实测），所以每轮都按满价计费。Executor 的工具用法靠 ``executor.md``
里的文本工具表教，365 token，静态，可缓存。

屏幕描述有两种形态，寻址字段不一样。``executor.md`` 是静态的，必须同时讲清两种
形态；「这一屏是哪一种」只有运行时知道（Android 有统一元素表，iOS / 云设备只有
legacy OCR 块）。有元素表时模型填 ``Target.index``，没元素表时模型只能填
``Target.ocr_index``。判据在 ``has_element_table``，它和工具侧
``tools/utils.resolve_target`` 看的是同一个 ``state.latest_screen_elements``。
两边判据一旦不一致，模型填的字段和工具认的字段就对不上，表现是每个动作都解析
失败。legacy OCR 分支因此必须自带一段「这里用 ``ocr_index``」的说明：静态提示词
只教 ``index`` 时，模型在 iOS / 云设备上每个动作都会被工具拒掉
（``index=N is not a row in the screen element table (no table)``）。

两条感知通道由 Manager 侧开关。``ctx.perception`` 有两个开关（``ui_hierarchy`` /
``ocr``），Manager 建 agent 时决定（``sdk/agent.py:722``），本模块只负责按开关裁
prompt。UI 元素表关掉就不建表，走 legacy OCR 分支（``ocr_index``），判据仍然是
``has_element_table``，提示侧和工具侧共用同一个。OCR 关掉就不给 legacy OCR 块，
不是「关了还偷偷给」。

屏幕截图不进模型。prompt 里没有任何图，模型只有两条文字通道：UI 元素表和
OCR 块。

工具结果截断。局部历史是 prompt 里唯一随步数线性增长的一段，而工具结果可能
很长（读笔记 = 整条笔记回显、输入文本 = 原文回显）。所以发给 LLM 的那份局部历史
里，过长的成功结果会被截成「首句 … 末句」，上限 120 token。错误结果逐字保留：
报错那部分模型要靠它判断「为什么失败」，压了就等于让它盲猜。

只截 Executor 局部历史这一路。Reporter 走的是 ``state.executor_messages``
（原对象），守「事实不能压」的纪律（``utils/thoughts_compaction.py`` 的模块
文档），一个字都不动。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.prompt_language import PromptLanguage, load_prompt_template
from minitap.mobile_use.tools.index import (
    EXECUTOR_WRAPPERS_TOOLS,
    VIDEO_RECORDING_WRAPPERS,
    format_tools_list,
)
from minitap.mobile_use.utils.failed_targets import describe_target
from minitap.mobile_use.utils.message_content import (
    content_to_text,
    has_image_block,
    with_text_replaced,
)
from minitap.mobile_use.utils.ocr_view import render_ocr_rows
from minitap.mobile_use.utils.screen_elements import ScreenElement, render_screen_rows
from minitap.mobile_use.utils.thoughts_compaction import split_sentences
from minitap.mobile_use.utils.time import format_task_started_at

EXECUTOR_PROMPT_PATH = Path(__file__).parent / "executor.md"


def _blocks(language: PromptLanguage) -> Any:
    """模板里的宏集合：``system_message`` 加各运行时消息块。

    返回类型标成 ``Any``：Jinja 的 ``TemplateModule`` 把宏挂在实例 ``__dict__``
    上（``context.get_exported()``），静态类型里根本没有这些属性名，不标注就
    全是 ``reportAttributeAccessIssue``。宏名是否真的存在由
    ``tests/test_prompt_templates.py`` 校验，静态类型帮不上忙。

    所有会进 prompt 的文案都住在 ``.md`` 里，代码只负责挑数据：系统消息、目标、
    失败事实、设备状态、元素表表头、OCR 说明、反馈语。唯一的例外是元素行 /
    OCR 行本身，它们由共用渲染器生成，因为编号契约是「渲染」和「反解」两端共用的，
    不能拆成两份。

    ``load_prompt_template`` 按 ``(路径, 语言)`` 缓存编译结果，
    ``Template.module`` 也是缓存属性，所以同一份模板只编译一次。
    """
    return load_prompt_template(EXECUTOR_PROMPT_PATH, language).module


def render_executor_system(ctx: MobileUseContext) -> str:
    """渲染 Executor 的 system prompt（静态部分）。

    ``system_message`` 宏的参数：``platform`` / ``executor_tools_list`` /
    ``perception``。

    目标不在这个宏里。目标是每子目标变的，放进 system prompt 会让整个 system 块
    在换子目标时全部重算。目标走独立的消息块（``goal_message``，见
    ``build_executor_messages`` 的第 2 项），子目标内它是恒定的。

    ``perception`` 传的是通道开关状态（``ctx.perception``）：两条通道开着哪几条，
    在一次运行里是恒定的，所以它落在 system prompt 里不会伤缓存，但模型必须知道，
    否则它会按一个被关掉的通道寻址。

    语言跟着 ``ctx.prompt_language`` 走（``en`` 对应 ``executor.md``，``zh`` 对应
    ``executor-zh.md``）。
    """
    return _blocks(ctx.prompt_language).system_message(
        platform=ctx.device.mobile_platform.value,
        executor_tools_list=format_tools_list(
            ctx=ctx,
            wrappers=list(_executor_wrappers(ctx)),
        ),
        perception=ctx.perception,
    )


def _executor_wrappers(ctx: MobileUseContext) -> list[Any]:
    """Executor 这次能用的工具集合：固定的执行工具，加上按需的视频录制工具。"""
    wrappers = list(EXECUTOR_WRAPPERS_TOOLS)
    if ctx.video_recording_enabled:
        wrappers.extend(VIDEO_RECORDING_WRAPPERS)
    return wrappers


def render_goal_message(goal: str, *, language: PromptLanguage) -> str:
    """目标陈述块（子目标内恒定，排在局部历史之前才能被缓存覆盖）。"""
    return _blocks(language).goal_message(goal=goal)


def render_failures_message(failures: list[dict[str, Any]], *, language: PromptLanguage) -> str:
    """跨子目标的失败事实块；空账本返回空串（调用方据此不发消息）。

    每条事实的描述由 ``describe_target`` 拼，不带 index：索引每轮重编号，写进给
    模型看的失败事实里会让它去点一个完全无关的元素（见 ``utils/failed_targets.py``
    的 ``describe_target``）。

    Reporter 那一路仍然用带 index 的 ``describe``（事实层、逐字搬运），两边故意
    不一样：给 Executor 的是「模型的避坑清单」，给 Reporter 的是「事实原文」。
    """
    facts = [
        {
            "desc": describe_target(
                label=item.get("label"),
                resource_id=item.get("resource_id"),
            ),
            "reason": item.get("reason") or "",
        }
        for item in (failures or [])
    ]
    if not facts:
        return ""
    return _blocks(language).failures_message(failures=facts)


def render_device_state(
    *,
    device_date: str | None = None,
    focused_app_info: str | None = None,
    screen_off: bool = False,
    no_perception: bool = False,
    language: PromptLanguage,
) -> str:
    """设备状态块（日期 / 前台 app / 息屏 / 无感知通道）；无内容返回空串。

    息屏那句是关键：真机实测 30 分钟息屏超时让一个任务从 30 步涨到 51 步还没跑完，
    模型看不到「屏是黑的」，就在黑屏上反复尝试。
    """
    return _blocks(language).device_state_message(
        device_date=device_date,
        focused_app_info=focused_app_info,
        screen_off=screen_off,
        no_perception=no_perception,
    )


def render_screen_message(elements: list[ScreenElement] | None, *, language: PromptLanguage) -> str:
    """元素表块：表头在模板里，行由共用渲染器生成（索引契约不能拆两份）。"""
    rows = render_screen_rows(elements or [])
    if not rows:
        return ""
    return _blocks(language).screen_elements_message(rows=rows)


def render_dispatch_rejection(reason: str, *, language: PromptLanguage) -> str:
    """动作没被派发时的回灌（不是 ToolMessage：那份没有对应的 tool_call）。"""
    return _blocks(language).dispatch_rejection_message(reason=reason)


def render_vision_failure(reason: str, *, language: PromptLanguage) -> str:
    """视觉定位失败时的回灌（和 dispatch 拒绝同一条出口：不派发，模型重决定）。

    它必须带上**具体原因**（看不见 / 模型不支持图片 / 息屏），否则模型只会把
    同一句话原样再报一遍。
    """
    return _blocks(language).vision_failure_message(reason=reason)


def render_retry_nudge(reason: str | None, *, language: PromptLanguage) -> str:
    """解析失败后重试一次时的提醒，必须带上具体的失败原因。"""
    return _blocks(language).retry_message(reason=reason)


def render_extra_actions_notice(count: int, *, language: PromptLanguage) -> str:
    """一轮只执行第一个动作时，告诉模型其余动作没有被执行。"""
    return _blocks(language).extra_actions_message(count=count)


def render_step_limit_message(limit: int, *, language: PromptLanguage) -> str:
    """步数上限触发时的收尾文案。"""
    return _blocks(language).step_limit_message(limit=limit)


def render_no_goal_message(*, language: PromptLanguage) -> str:
    """没有子目标时的收尾文案。"""
    return _blocks(language).no_goal_message()


def render_done_fallback(*, language: PromptLanguage) -> str:
    """``done`` 但没有 done_reason / agent_thought 时的收尾文案。"""
    return _blocks(language).done_fallback_message()


def render_failed_fallback(*, language: PromptLanguage) -> str:
    """``failed`` 但没有 failed_reason / agent_thought 时的收尾文案。"""
    return _blocks(language).failed_fallback_message()


def render_stall_notice(description: str, *, language: PromptLanguage) -> str:
    """「没有进展」的提醒块。

    它是信息，不是限制：这块只出现在「检测到重复」的那一轮，不截断、不改结果，
    也不替模型做决定（见 ``agents/executor/progress.py``）。它告诉模型两件事：
    别重复刚才那个动作；做不成就用 ``failed`` 说出来。
    """
    return _blocks(language).stall_notice_message(description=description)


def has_element_table(screen_elements: list[ScreenElement] | None) -> bool:
    """本屏有没有统一元素表。这是两种寻址方式的唯一判据。

    必须和工具侧 ``tools/utils.resolve_target`` 的判据一致（那边看的也是
    ``state.latest_screen_elements``）。有表时模型填 ``Target.index``
    （``ocr_index`` 是它的别名），没表时模型只能填 ``Target.ocr_index``。两边
    判据一旦不一致，模型填的字段和工具认的字段就对不上。

    抽成独立函数而不是内联 ``if screen_elements:``，是为了让测试能直接断言判据
    本身，不用去拼整份 prompt 再 grep。
    """
    return bool(screen_elements)


def render_legacy_ocr_block(ocr_results: list[Any] | None, language: PromptLanguage) -> str:
    """没有元素表时的屏幕描述：说明段 + 表头 + 共用的 OCR 行。

    ``render_ocr_rows`` 是共用的，不能改它。它打的 ``[i]`` 和
    ``ocr_view.resolve_ocr_index`` 吃的下标是同一条契约的两端，模板只加说明和
    表头，编号本身一个字节都不动。

    字面量不许翻：``ocr_index`` / ``index`` 是模型要填的字段名，那句报错原文是
    工具真的会给出来的字符串，模型要照着它认错。

    没有 OCR 结果时返回空串（调用方据此不发消息）。
    """
    rows = render_ocr_rows(ocr_results)
    if not rows:
        return ""
    return _blocks(language).ocr_message(rows=rows)


# 工具结果截断
#
# 局部历史是唯一随步数线性增长的一段：每走一步就多一条工具结果，而下一轮它要
# 整段重发。工具结果可能很长（读笔记 = 整条笔记回显、输入文本 = 原文回显、
# 失败详情 = 所有尝试的拼接），一段历史就能把 prompt 撑爆。
#
# 只截成功结果，错误结果逐字保留：报错那部分模型要靠它判断「为什么失败、
# 下一步换哪」，压了就等于让它盲猜。
#
# 只对 executor 局部历史做。Reporter 那一路走的是 ``state.executor_messages``
# （原对象），守「事实不能压」的纪律（``utils/thoughts_compaction.py`` 的模块
# 文档），一律不动。这里的截断只影响发给 Executor LLM 的那份副本。

# 一条工具结果在局部历史里最多留多少 token（80~120 tok 上沿）。
MAX_TOOL_RESULT_TOKENS = 120

# 首句 / 末句之间的省略标记（与 thoughts_compaction 用同一个符号，避免两套记号）
_ELLIPSIS = " … "


def _is_cjk(ch: str) -> bool:
    """这个字符是不是「一个字算一个 token」的那种（CJK / 谚文 / 全角）。"""
    code = ord(ch)
    return (
        0x3000 <= code <= 0x303F  # CJK 标点
        or 0x4E00 <= code <= 0x9FFF  # CJK 统一表意文字
        or 0xAC00 <= code <= 0xD7AF  # 谚文音节
        or 0xFF00 <= code <= 0xFF60  # 全角形式
    )


def estimate_tokens(text: str) -> float:
    """粗略的 token 估计。

    CJK 一字一 token，其余 4 字一 token，够用来定一个「这条结果要不要截」的
    门槛，不追求精确。不引入分词器依赖：一个近似的预算判断不值得给运行时加一个
    tokenizer，而且不同 provider 的分词器本来就不一样。
    """
    cjk = sum(1 for ch in text if _is_cjk(ch))
    return cjk + (len(text) - cjk) / 4


def _cap_by_tokens(text: str, budget: float) -> str:
    """按 token 预算砍掉尾巴（保前缀）。"""
    used = 0.0
    kept: list[str] = []
    for ch in text:
        cost = 1.0 if _is_cjk(ch) else 0.25
        if used + cost > budget:
            break
        used += cost
        kept.append(ch)
    return "".join(kept)


def truncate_tool_result(text: str, *, max_tokens: float = MAX_TOOL_RESULT_TOKENS) -> str:
    """把一条过长的工具结果压成「首句 … 末句」。

    - 不超预算就原样返回，短结果一个字节都不动，这是绝大多数情况。
    - 超预算就留首句加末句：首句是结论（「已成功点中 xxx」），末句是收尾
      （「新文本是 xxx」）。中间是冗长回显，价值最低。
    - 没有句子边界（整段 JSON / 一整屏文本）就只能砍尾巴。
    """
    if estimate_tokens(text) <= max_tokens:
        return text

    sentences = split_sentences(text)
    if len(sentences) < 2:
        # 一整块没有句号的内容，砍尾巴总比整块留着强
        return _cap_by_tokens(text, max_tokens)

    # 省略标记本身也占预算，不扣掉它，结果会刚好超线一点点
    head_budget = max(1.0, max_tokens * 0.6)
    tail_budget = max(1.0, max_tokens - head_budget - estimate_tokens(_ELLIPSIS))
    head = _cap_by_tokens(sentences[0], head_budget)
    tail = _cap_by_tokens(sentences[-1], tail_budget)
    return f"{head}{_ELLIPSIS}{tail}"


# 这些工具的结果是载荷，不是「冗长回显」，截断它们等于丢信息。
#
# - ``read_note`` 返回整条笔记的内容（Executor 可以把整屏文本存成笔记，长度没有
#   上界），截掉的就是模型要读的东西本身。
# - ``list_notes`` 返回全部笔记 key（随笔记数增长），截掉就是「有几个笔记」这个
#   事实被抹了。
# - ``focus_and_input_text`` / ``focus_and_clear_text`` 返回输入框的最终内容，
#   模型要拿它确认「到底打进去没有 / 清干净没有」。
#
# 豁免它们的收益本来就是零：实测真机工具结果最长 21 token，120 的门槛一次都没
# 触发过。所以这条豁免不花任何东西，只是挡住一个「万一某天门槛被调低，或某天
# 真的出现了长结果」就会静默丢信息的坑。
_PAYLOAD_TOOLS = frozenset(
    {
        "read_note",
        "list_notes",
        "focus_and_input_text",
        "focus_and_clear_text",
    }
)


def _tool_name_by_call_id(messages: list[AnyMessage]) -> dict[str, str]:
    """``tool_call_id`` 到工具名。

    局部历史里的 ``ToolMessage`` 大多没有 ``name`` 字段（工具自己建的时候没传），
    所以只能从前一条 ``AIMessage`` 的 ``tool_calls`` 里反查，``tool_call_id`` 就是
    两边配对的依据。
    """
    names: dict[str, str] = {}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            call_id = call.get("id")
            name = call.get("name")
            if call_id and name:
                names[call_id] = name
    return names


def truncate_local_history(messages: list[AnyMessage]) -> list[AnyMessage]:
    """局部历史转成给 LLM 的那份（工具结果按需截断）。

    只改内容，不改结构：tool_call_id / status / 名字全部保留。tool_call_id 是
    OpenAI 兼容端点把 AIMessage 和 ToolMessage 配对的依据，动了会 400。

    载荷类工具豁免（见 ``_PAYLOAD_TOOLS``）：它们的结果不是冗长回显，截断等于
    丢信息。

    多模态内容（带图的消息）不能整体 ``str()``。带图消息的 content 是
    ``[文字块, 图片块]``，以前那句 ``str(message.content or "")`` 会把整个列表
    变成 Python 字面量（几十 KB 的 base64 混在里面），然后截断再把它写回成一个
    字符串，图就没了，模型收到的是一坨 base64 噪声。所以文字部分照常截断，
    图片块原样保留（见 ``utils/message_content.with_text_replaced``）。
    """
    names = _tool_name_by_call_id(messages)
    result: list[AnyMessage] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            result.append(message)
            continue
        raw_content = message.content
        content = content_to_text(raw_content)
        # 报错逐字保留：模型要靠它知道为什么失败。
        if getattr(message, "status", "") == "error":
            result.append(message)
            continue
        # 载荷类工具逐字保留：内容本身就是模型要的东西。
        tool_name = message.name or names.get(message.tool_call_id or "")
        if tool_name in _PAYLOAD_TOOLS:
            result.append(message)
            continue
        truncated = truncate_tool_result(content)
        if truncated == content and not has_image_block(raw_content):
            result.append(message)
            continue
        # model_copy 而不是新建：additional_kwargs / artifact / response_metadata
        # 都原样带过去，不在这里丢信息。
        result.append(
            message.model_copy(update={"content": with_text_replaced(raw_content, truncated)})
        )
    return result


#: 模型唯一能调用的工具，也就是结构化输出 schema 绑上去的那个。
_EXECUTOR_OUTPUT_TOOL = "ExecutorOutput"


def rewrite_history_tool_calls(messages: list[AnyMessage]) -> list[AnyMessage]:
    """把局部历史里 assistant 消息的 tool_calls 改写成「模型当初调的是 ExecutorOutput」。

    历史是模型下一轮模仿的对象。以前历史里的 assistant 消息带着
    ``{"name": "tap"}`` 这样的设备工具名，而模型唯一能调的工具是
    ``ExecutorOutput``，它照着历史学，就学会了喊 ``tap``。

    真机实测（2026-10-22，快手极速版 / Redmi Turbo 4）：executor 66 次调用里
    37 次喊错了工具名（tap / wait_for_delay / swipe_percentages /
    focus_and_input_text 等），动作被解析层吞成空动作，回灌一条无关的拒绝理由，
    连着 8 轮零动作。对照组里历史还没有设备工具名的那几次，一次都没错。

    所以从源头改写历史，比事后纠正便宜得多，事后纠正要多烧一次 LLM 调用。

    改写后模型看到的是它自己当初的形状::

        assistant: ExecutorOutput(actions=[{"action": "tap", ...}], agent_thought=...)
        tool:      "Tap on element ... was successful."

    ``tool_call_id`` 一律不动，它是 OpenAI 兼容端点把 AIMessage 和 ToolMessage
    配对的依据，动了会 400（见 ``truncate_local_history`` 的说明）。ToolMessage
    的 ``name`` 跟着改成同一个名字（配对的是 id，但同名更不容易被端点挑刺）；
    设备工具名本身没丢，它在 assistant 那边的 ``action`` 里，工具结果正文里也
    写着。
    """
    renamed_ids: set[str] = set()
    result: list[AnyMessage] = []
    for message in messages:
        calls = getattr(message, "tool_calls", None) or []
        if not calls:
            result.append(message)
            continue
        new_calls: list[dict[str, Any]] = []
        changed = False
        for call in calls:
            name = call.get("name")
            if not name or name == _EXECUTOR_OUTPUT_TOOL:
                new_calls.append(call)
                continue
            args = dict(call.get("args") or {})
            thought = args.get("agent_thought") or ""
            # ``agent_thought`` 只在顶层留一份。模型自己产出的 ExecutorOutput
            # 就是这个形状（actions 条目里没有 agent_thought），历史必须和它长得
            # 一模一样：历史是给模型模仿的，长得不一样就是在教它另一种格式。
            action = {
                "action": name,
                **{k: v for k, v in args.items() if k != "agent_thought"},
            }
            # 视觉目标：历史里**只留模型自己写的那句话**。`bounds` 是框架定位后
            # 填进去的；留在历史里，模型会读成「我给过坐标、框架用的就是它」，
            # 于是反复怀疑视觉没生效（真机实测 2026-10-28，连着 3 轮这么想）。
            target = action.get("target")
            if isinstance(target, dict) and target.get("vision"):
                action["target"] = {"vision": target["vision"]}
            new_calls.append(
                {
                    **call,
                    "name": _EXECUTOR_OUTPUT_TOOL,
                    "args": {
                        "actions": [action],
                        "agent_thought": thought,
                    },
                }
            )
            if call.get("id"):
                renamed_ids.add(call["id"])
            changed = True
        if not changed:
            result.append(message)
            continue
        # 同时清掉 additional_kwargs 里的原始 tool_calls。
        #
        # provider 解析出来的 AIMessage 会在 ``additional_kwargs["tool_calls"]``
        # 里留一份原始副本。langchain 的转换器优先用 ``message.tool_calls``
        # （见 langchain_openai 的 ``_convert_message_to_dict``），所以留着它
        # 现在不会出错，但那是两份真值来源，改一份忘一份就是真 bug。改写完就把
        # 这份陈旧副本删掉，改写后的 ``tool_calls`` 是唯一真值。
        additional = dict(getattr(message, "additional_kwargs", None) or {})
        additional.pop("tool_calls", None)
        # model_copy 而不是新建：artifact / response_metadata 都原样带过去，
        # 不在这里丢信息。
        result.append(
            message.model_copy(update={"tool_calls": new_calls, "additional_kwargs": additional})
        )

    if not renamed_ids:
        return result
    # ToolMessage 的 name 跟 assistant 那边的工具名对齐。
    return [
        (
            m.model_copy(update={"name": _EXECUTOR_OUTPUT_TOOL})
            if isinstance(m, ToolMessage) and (m.tool_call_id or "") in renamed_ids
            else m
        )
        for m in result
    ]


def build_executor_messages(
    *,
    ctx: MobileUseContext,
    goal: str,
    screen_elements: list[ScreenElement] | None,
    failures: list[dict[str, Any]],
    local_messages: list[AnyMessage],
    device_state_lines: list[str],
    ocr_results: list[Any] | None = None,
    stall_notice: str | None = None,
) -> list[BaseMessage]:
    """组装 Executor 每一轮的 prompt。

    顺序上每轮变的一律靠后：

    1. SystemMessage：``executor.md`` 的渲染结果（静态）。
    2. HumanMessage：目标陈述（子目标内恒定）。
    3. 局部历史：动作 + 工具反馈，只追加。
    4. HumanMessage：失败事实清单（每轮变，会增长）。
    5. HumanMessage：设备状态（每轮变，前台 app 会变）。
    6. HumanMessage：「没有进展」提醒（偶发，只在检测到重复的那一轮）。
    7. HumanMessage：屏幕元素表（每轮变，重编号）。

    顺序直接等于缓存命中率。前缀缓存只覆盖从头开始的连续前缀，任何每轮变的内容
    排在局部历史之前，它之后的整段历史每轮都要满价重算。真机实测（2026-10-21）：
    设备状态和失败清单卡在历史前面时，executor 的缓存命中率只有 15.8%（六个
    agent 里最差，而它调用量最大）。

    屏幕截图这条通道已整条移除（含按需工具 ``take_screenshot``），所以没有第 8
    块，prompt 里不再有任何图，元素表 / OCR 块就是最后一块。截图本身仍用于 OCR
    与坐标校验（框架内部取数），只是不再进模型。

    Args:
        ctx: 运行上下文。
        goal: 本次子目标的目标陈述。
        screen_elements: 当屏的元素表（Contextor 建的，当屏的）。空或 None 表示
            本屏没有元素表（iOS / 云设备路径，或 ``perception.ui_hierarchy``
            关着），那时改用 legacy OCR 块加 ``ocr_index`` 寻址。
        failures: 跨子目标的失败事实账本。
        local_messages: 本子目标内的局部历史。发给 LLM 的那份会按需截断过长的
            成功工具结果（见 :func:`truncate_local_history`），``state`` 里的原
            对象不动，Reporter 读的是原对象。
        device_state_lines: 设备状态行。
        ocr_results: OCR 结果。只在没有元素表时用（iOS / 云设备路径），渲染成
            legacy OCR 块，并带上「用 ocr_index」的说明段。

    Returns:
        发给 LLM 的消息列表。
    """
    messages: list[BaseMessage] = [
        SystemMessage(content=render_executor_system(ctx)),
    ]

    # 任务开始时间：系统提示词之后，单独一条消息。
    #
    # 为什么放这儿而不是塞进系统提示词的内容里：系统提示词必须逐字节稳定，它是
    # 整段请求里最大的一块。塞进去，它就变成「每任务唯一」，同一 persona 的不同
    # 任务之间不再共享那段缓存。单独一条，系统提示词一个字不动，跨任务照样能复用。
    #
    # 为什么这条消息是安全的：任务开始时间整场运行不变，它和它前面的系统提示词
    # 一样是前缀的一部分，永远命中。
    task_started_at = format_task_started_at(ctx.task_started_at)
    if task_started_at:
        messages.append(
            HumanMessage(content=_blocks(ctx.prompt_language).task_started_message(task_started_at))
        )

    messages.append(HumanMessage(content=render_goal_message(goal, language=ctx.prompt_language)))

    # 局部历史排在「每轮变的块」之前，见模块文档的顺序说明。
    #
    # 工具结果在这一步按需截断：局部历史是唯一随步数线性增长的一段，而过长的一条
    # 结果能把这整段历史撑爆。只截成功结果，错误逐字保留。
    #
    # 顺序是先截断、后改写。截断要靠原工具名判断载荷豁免（``_PAYLOAD_TOOLS``），
    # 改写之后名字变成 ``ExecutorOutput``，豁免就失效了，载荷类工具的结果会被
    # 截掉，等于把模型要读的东西本身丢了。
    messages.extend(rewrite_history_tool_calls(truncate_local_history(local_messages)))

    # 下面这三个块都是「每轮变」的，必须排在局部历史之后。
    #
    # 原因：前缀缓存只覆盖从头开始的连续前缀。任何每轮变的内容出现在局部历史
    # 之前，它之后的所有字节（也就是整段局部历史）每轮都要满价重算。
    #
    # 真机实测（2026-10-21）：设备状态里「前台 app」几乎每次导航都变，失败清单
    # 也会随子目标增长。两者卡在历史前面时，executor 的缓存命中率只有 15.8%
    # （六个 agent 里最差，而它调用量最大）。挪到历史之后，历史那段就能被缓存覆盖。
    #
    # 元素表仍然排在最后，它每轮重编号，永不缓存；设备状态和失败清单放在它前面、
    # 局部历史之后。

    failure_line = render_failures_message(failures, language=ctx.prompt_language)
    if failure_line:
        messages.append(HumanMessage(content=failure_line))

    if device_state_lines:
        messages.append(HumanMessage(content="\n".join(device_state_lines)))

    # 「没有进展」的提醒放在这里（历史之后、元素表之前）。它是偶发的，只在检测到
    # 重复的那一轮出现，但出现时每轮都变，所以同样不能排在局部历史前面。
    if stall_notice:
        messages.append(HumanMessage(content=stall_notice))

    # 元素表每轮都变，所以放最后。判据统一走 has_element_table，提示侧和工具侧
    # 必须用同一个判据。
    elements = screen_elements or []
    if has_element_table(elements):
        screen_text = render_screen_message(elements, language=ctx.prompt_language)
        if screen_text:
            messages.append(HumanMessage(content=screen_text))
    else:
        # iOS / 云设备：层级树形状不认识，建不出统一元素表。这时必须用 legacy
        # OCR 块，它的 ``[i]`` 与 ``state.latest_ocr_results`` 的下标严格对应
        # （``utils/ocr_view.py:51-52``），而工具侧的 ``ocr_index`` 正是按那个
        # 下标解析的。
        #
        # 换成别的渲染方式会让模型看到的编号和工具侧的下标对不上，那是索引契约的
        # 破坏，比「少一层信息」严重得多。
        #
        # 说明段（用 ``ocr_index`` 而不是 ``index``）由 render_legacy_ocr_block
        # 负责：静态提示词只教 ``index``，没有它模型每个动作都会被工具拒掉。
        #
        # OCR 通道关着时不给这一块。Manager 把 ``perception.ocr`` 关掉就是要
        # 「少一条通道」，不是「关了但还偷偷给」。这时模型手里什么都没有，
        # Executor 节点会往 device_state_lines 里加一句告知，让它如实说「看不见」，
        # 而不是瞎猜。
        if ctx.perception.ocr:
            ocr_text = render_legacy_ocr_block(ocr_results, ctx.prompt_language)
            if ocr_text:
                messages.append(HumanMessage(content=ocr_text))

    return messages
