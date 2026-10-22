"""Executor 的 prompt 组装。

**消息顺序是这里最重要的一件事**（方案 §4.1.1）：

    每轮变的东西一律放**末尾**。

因为 provider 的缓存是**前缀**匹配 —— 任何每轮变的内容放在中间，
它之后的全部字节都要重算。屏幕元素表每轮都变（每轮重抓屏），
所以它必须在**局部历史之后**：

    错： [静态][目标][失败][设备][**元素表**][历史]   → 命中率递减到 48%
    对： [静态][目标][失败][设备][历史][**元素表**]   → 命中率递增到 93%

实测数字见 ``doc/executor-autonomy-plan.md`` §4.1.1，
复现脚本 ``scripts/audit_executor_cache_layout.py``。

**另一条硬约束：不要 bind_tools 那 16 个设备工具。**
bind_tools 的 JSON schema 是 4,008 token，而且**不参与缓存键**（实测 E5），
所以每轮都按满价计费。Executor 的工具用法靠 ``executor.md`` 里的
**文本工具表**教（365 token，静态，可缓存）。

**屏幕描述有两种形态，寻址字段不一样**
------------------------------------------------
``executor.md`` 是静态的，必须同时讲清两种形态；但「这一屏是哪一种」
只有运行时知道（Android 有统一元素表，iOS / 云设备只有 legacy OCR 块）。

    有元素表 → 模型填 ``Target.index``
    没元素表 → 模型只能填 ``Target.ocr_index``

判据在 ``has_element_table``，它和工具侧 ``tools/utils.resolve_target``
看的是同一个 ``state.latest_screen_elements`` —— 两边判据一旦不一致，
模型填的字段和工具认的字段就对不上，表现是**每个动作都解析失败**。

legacy OCR 分支因此必须**自带**一段「这里用 ``ocr_index``」的说明：
静态提示词只教 ``index`` 时，模型在 iOS / 云设备上每个动作都会被工具
拒掉（``index=N is not a row in the screen element table (no table)``）。

**工具结果截断（L5）**
--------------------
局部历史是 prompt 里唯一随步数线性增长的一段，而工具结果可能很长
（读笔记 = 整条笔记回显、输入文本 = 原文回显）。所以发给 LLM 的那份
局部历史里，**过长的成功结果会被截成「首句 … 末句」**（上限 120 token）。

**错误结果逐字保留** —— 报错那部分模型要靠它判断「为什么失败」，压了就
等于让它盲猜。

⚠️ **只截 Executor 局部历史这一路**。Reporter 走的是 ``state.executor_messages``
（原对象），守「事实不能压」的纪律（``utils/thoughts_compaction.py`` 的模块
文档），一个字都不动。
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from jinja2 import Template
from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.tools.index import format_tools_list
from minitap.mobile_use.utils.failed_targets import render_failure_facts
from minitap.mobile_use.utils.ocr_view import render_ocr_results
from minitap.mobile_use.utils.screen_elements import ScreenElement, render_screen_elements
from minitap.mobile_use.utils.thoughts_compaction import split_sentences

EXECUTOR_PROMPT_PATH = Path(__file__).parent / "executor.md"


@cache
def _load_template() -> Template:
    return Template(EXECUTOR_PROMPT_PATH.read_text(encoding="utf-8"))


def render_executor_system(ctx: MobileUseContext) -> str:
    """渲染 Executor 的 system prompt（静态部分）。

    ``executor.md`` 的模板变量恰好三个：``platform`` / ``executor_tools_list``
    / ``goal``。

    ⚠️ **``goal`` 故意传空串**：目标是**每子目标**变的，放进 system prompt
    会让整个 system 块在换子目标时全部重算。目标走独立的消息块
    （见 `build_executor_messages` 的 [2]），子目标内它是恒定的。
    """
    return _load_template().render(
        platform=ctx.device.mobile_platform.value,
        executor_tools_list=format_tools_list(
            ctx=ctx,
            wrappers=list(_executor_wrappers(ctx)),
        ),
        goal="",
    )


def _executor_wrappers(ctx: MobileUseContext):
    from minitap.mobile_use.tools.index import EXECUTOR_WRAPPERS_TOOLS, VIDEO_RECORDING_WRAPPERS

    wrappers = list(EXECUTOR_WRAPPERS_TOOLS)
    if ctx.video_recording_enabled:
        wrappers.extend(VIDEO_RECORDING_WRAPPERS)
    return wrappers


def render_device_state(state_lines: list[str]) -> str:
    """设备状态块（息屏 / 前台 app / 日期）。

    息屏那行是关键：真机实测 30 分钟息屏超时让一个任务从 30 步涨到 51 步
    还没跑完 —— 模型看不到「屏是黑的」，就在黑屏上反复尝试。
    """
    return "\n".join(state_lines)


# 无元素表时的说明段。放在 OCR 块**前面**：模型是从上往下读的，
# 一旦先看到 `[i]`，它就会照着静态提示词里「index 直接用」那句话去填
# `index` —— 而无元素表时 `index` 必然解析失败，每个动作都被工具拒掉。
_NO_TABLE_NOTE = """\
NO SCREEN ELEMENT TABLE ON THIS SCREEN.
The device could not report an element tree, so the OCR block below is the whole
screen description.

Address those lines with **`ocr_index`** - the leading `[i]` of the OCR block,
1-based, copied verbatim. Do NOT use `index` here: with no table there is no row
for it to point at, so every `index` target fails with:
"index=N is not a row in the screen element table (no table)".

```json
{"action": "tap", "target": {"ocr_index": 3}}
```
"""


def has_element_table(screen_elements: list[ScreenElement] | None) -> bool:
    """本屏有没有统一元素表。**这是两种寻址方式的唯一判据。**

    必须和工具侧 ``tools/utils.resolve_target`` 的判据一致（那边看的也是
    ``state.latest_screen_elements``）：

    * 有表 → 模型填 ``Target.index``（``ocr_index`` 是它的别名）
    * 没表 → 模型只能填 ``Target.ocr_index``

    两边判据一旦不一致，模型填的字段和工具认的字段就对不上。

    抽成独立函数而不是内联 ``if screen_elements:``，是为了让测试能直接
    断言判据本身，不用去拼整份 prompt 再 grep。
    """
    return bool(screen_elements)


def render_legacy_ocr_block(ocr_results: list | None) -> str:
    """没有元素表时的屏幕描述：说明段 + 共用的 OCR 渲染。

    ``render_ocr_results`` 是**共用**的，不能改它 —— 它打的 ``[i]`` 和
    ``ocr_view.resolve_ocr_index`` 吃的下标是同一条契约的两端。这里只在
    前面加说明，编号本身一个字节都不动。

    没有 OCR 结果时返回空串（调用方据此不发消息）。
    """
    body = render_ocr_results(ocr_results)
    if not body:
        return ""
    return f"{_NO_TABLE_NOTE}\n{body}"


# ---------------------------------------------------------------------------
# 工具结果截断（L5）
# ---------------------------------------------------------------------------
#
# 局部历史是**唯一随步数线性增长**的一段：每走一步就多一条工具结果，
# 而下一轮它要**整段重发**。工具结果可能很长（读笔记 = 整条笔记回显、
# 输入文本 = 原文回显、失败详情 = 所有尝试的拼接），一段历史就能把 prompt 撑爆。
#
# 只截**成功**结果，**错误结果逐字保留** ——
# 报错那部分模型要靠它判断「为什么失败、下一步换哪」，压了就等于让它盲猜。
#
# ⚠️ **只对 executor 局部历史做**。Reporter 那一路走的是 ``state.executor_messages``
# （原对象），守「事实不能压」的纪律（``utils/thoughts_compaction.py`` 的模块文档），
# 一律不动。这里的截断只影响**发给 Executor LLM 的那份**副本。

# 一条工具结果在局部历史里最多留多少 token（方案 L5：80~120 tok 的上沿）。
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

    CJK 一字一 token，其余 4 字一 token —— 够用来定一个「这条结果要不要截」
    的门槛，不追求精确。**不引入分词器依赖**：一个近似的预算判断不值得
    给运行时加一个 tokenizer（而且不同 provider 的分词器本来就不一样）。
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
    """把一条**过长**的工具结果压成「首句 … 末句」。

    * 不超预算 → **原样返回**（短结果一个字节都不动，这是绝大多数情况）。
    * 超预算 → 留首句 + 末句：首句是结论（「已成功点中 xxx」），
      末句是收尾（「新文本是 xxx」）。中间是冗长回显，价值最低。
    * 没有句子边界（整段 JSON / 一整屏文本）→ 只能砍尾巴。
    """
    if estimate_tokens(text) <= max_tokens:
        return text

    sentences = split_sentences(text)
    if len(sentences) < 2:
        # 一整块没有句号的内容 —— 砍尾巴总比整块留着强
        return _cap_by_tokens(text, max_tokens)

    # 省略标记本身也占预算 —— 不扣掉它，结果会刚好超线一点点
    head_budget = max(1.0, max_tokens * 0.6)
    tail_budget = max(1.0, max_tokens - head_budget - estimate_tokens(_ELLIPSIS))
    head = _cap_by_tokens(sentences[0], head_budget)
    tail = _cap_by_tokens(sentences[-1], tail_budget)
    return f"{head}{_ELLIPSIS}{tail}"


# 这些工具的结果是**载荷**，不是「冗长回显」—— 截断它们等于丢信息。
#
# * `read_note` 返回整条笔记的内容（Executor 可以把整屏文本存成笔记，
#   长度没有上界）—— 截掉的就是模型要读的东西本身。
# * `list_notes` 返回全部笔记 key（随笔记数增长）—— 截掉就是「有几个
#   笔记」这个事实被抹了。
# * `focus_and_input_text` / `focus_and_clear_text` 返回输入框的**最终内容**
#   —— 模型要拿它确认「到底打进去没有 / 清干净没有」。
#
# ⚠️ 豁免它们的**收益本来就是零**：实测真机工具结果最长 21 token
# （`scripts/probe_tool_result_lengths.py`），120 的门槛一次都没触发过。
# 所以这条豁免不花任何东西，只是挡住一个「万一某天门槛被调低 /
# 某天真的出现了长结果」就会静默丢信息的坑。
_PAYLOAD_TOOLS = frozenset(
    {
        "read_note",
        "list_notes",
        "focus_and_input_text",
        "focus_and_clear_text",
    }
)


def _tool_name_by_call_id(messages: list[AnyMessage]) -> dict[str, str]:
    """`tool_call_id` → 工具名。

    局部历史里的 `ToolMessage` **大多没有 `name` 字段**（工具自己建的时候
    没传），所以只能从**前一条 `AIMessage` 的 `tool_calls`** 里反查 ——
    `tool_call_id` 就是两边配对的依据。
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
    """局部历史 → 给 LLM 的那份（工具结果按需截断）。

    **只改内容，不改结构**：tool_call_id / status / 名字全部保留 ——
    tool_call_id 是 OpenAI 兼容端点把 AIMessage 和 ToolMessage 配对的依据，
    动了会 400。

    **载荷类工具豁免**（见 `_PAYLOAD_TOOLS`）：它们的结果不是冗长回显，
    截断等于丢信息。
    """
    names = _tool_name_by_call_id(messages)
    result: list[AnyMessage] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            result.append(message)
            continue
        content = str(message.content or "")
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
        if truncated == content:
            result.append(message)
            continue
        # model_copy 而不是新建：additional_kwargs / artifact / response_metadata
        # 都原样带过去，不在这里丢信息。
        result.append(message.model_copy(update={"content": truncated}))
    return result


def build_executor_messages(
    *,
    ctx: MobileUseContext,
    goal: str,
    screen_elements: list[ScreenElement] | None,
    failures: list[dict],
    local_messages: list[AnyMessage],
    device_state_lines: list[str],
    ocr_results: list | None = None,
) -> list[BaseMessage]:
    """组装 Executor 每一轮的 prompt。

    顺序（**每轮变的一律靠后**）::

        [1] SystemMessage   executor.md 渲染结果（静态）
        [2] HumanMessage    目标陈述（子目标内恒定）
        [3] ...             局部历史（动作 + 工具反馈）—— 只追加
        [4] HumanMessage    失败事实清单（每轮变：会增长）
        [5] HumanMessage    设备状态（每轮变：前台 app 会变）
        [6] HumanMessage    屏幕元素表（每轮变：重编号）—— **必须最后**

    ⚠️ **顺序 = 缓存命中率**。前缀缓存只覆盖从头开始的连续前缀：
    任何每轮变的内容排在局部历史**之前**，它之后的整段历史每轮都要满价重算。
    真机实测（2026-10-21）：设备状态和失败清单卡在历史前面时，
    executor 的缓存命中率只有 15.8%（六个 agent 里最差，而它调用量最大）。

    🔴 **2026-10-22 删掉了原来的 [7] 截图块**。Executor 的截图通道整条关掉了
    （见 `agents/executor/types.py`），prompt 里不再有图。
    这同时少了一个**永远不可缓存**的块 —— 截图的 base64 长度随画面变，
    前缀缓存直接失效。

    Args:
        ctx: 运行上下文。
        goal: 本次子目标的目标陈述。
        screen_elements: 当屏的元素表（Contextor 建的，**当屏的**）。
            空 / None 表示本屏**没有**元素表（iOS / 云设备路径），
            那时改用 legacy OCR 块 + ``ocr_index`` 寻址。
        failures: 跨子目标的失败事实账本。
        local_messages: 本子目标内的局部历史。**发给 LLM 的那份会按需截断
            过长的成功工具结果**（见 :func:`truncate_local_history`），
            ``state`` 里的原对象不动 —— Reporter 读的是原对象。
        device_state_lines: 设备状态行。
        ocr_results: OCR 结果。**只在没有元素表时用**（iOS / 云设备路径），
            渲染成 legacy OCR 块，并**带上「用 ocr_index」的说明段**。

    Returns:
        发给 LLM 的消息列表。
    """
    messages: list[BaseMessage] = [
        SystemMessage(content=render_executor_system(ctx)),
        HumanMessage(content=f"Your goal:\n{goal}"),
    ]

    # 局部历史在「每轮变的块」**之前** —— 见模块文档的顺序说明。
    #
    # 工具结果在这一步按需截断（L5）：局部历史是唯一随步数线性增长的一段，
    # 而过长的一条结果能把这整段历史撑爆。**只截成功结果，错误逐字保留。**
    messages.extend(truncate_local_history(local_messages))

    # ↓↓↓ 这三个块都是「每轮变」的，必须排在局部历史之后 ↓↓↓
    #
    # 为什么：前缀缓存只覆盖**从头开始的连续前缀**。任何每轮变的内容
    # 出现在局部历史**之前**，它之后的所有字节（也就是整段局部历史）
    # 每轮都要满价重算。
    #
    # 真机实测（2026-10-21）：设备状态里「前台 app」几乎每次导航都变，
    # 失败清单也会随子目标增长 —— 两者卡在历史前面时，
    # executor 的缓存命中率只有 **15.8%**（六个 agent 里最差，而它调用量最大）。
    # 挪到历史之后，历史那段就能被缓存覆盖。
    #
    # ⚠️ 顺序不是随便定的：**元素表必须仍在最后**（它每轮重编号，永不缓存），
    # 设备状态和失败清单放在它前面、局部历史之后。

    failure_line = render_failure_facts(failures)
    if failure_line:
        messages.append(HumanMessage(content=failure_line))

    if device_state_lines:
        messages.append(HumanMessage(content=render_device_state(device_state_lines)))

    # 元素表：每轮都变，所以放最后。
    # 判据统一走 has_element_table —— 提示侧和工具侧必须用同一个判据。
    elements = screen_elements or []
    if has_element_table(elements):
        screen_text = render_screen_elements(elements)
        if screen_text:
            messages.append(HumanMessage(content=screen_text))
    else:
        # iOS / 云设备：层级树形状不认识，建不出统一元素表。
        # 这时**必须**用 legacy OCR 块 —— 它的 `[i]` 与
        # `state.latest_ocr_results` 的下标严格对应（`utils/ocr_view.py:51-52`），
        # 而工具侧的 `ocr_index` 正是按那个下标解析的。
        #
        # 换成别的渲染方式会让模型看到的编号和工具侧的下标对不上 ——
        # 那是索引契约的破坏，比「少一层信息」严重得多。
        #
        # 说明段（用 `ocr_index` 而不是 `index`）由 render_legacy_ocr_block
        # 负责：静态提示词只教 `index`，没有它模型每个动作都会被工具拒掉。
        ocr_text = render_legacy_ocr_block(ocr_results)
        if ocr_text:
            messages.append(HumanMessage(content=ocr_text))

    return messages