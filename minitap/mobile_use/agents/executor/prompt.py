"""Executor 的 prompt 组装。

消息顺序是这里最重要的一件事：每轮变的东西一律放末尾。

provider 的缓存是前缀匹配。任何每轮变的内容放在中间，它之后的全部字节都要重算。
屏幕元素表每轮都变（每轮重抓屏），所以它必须排在局部历史之后。实测的差别很大：
元素表排在历史前面，缓存命中率递减到 48%；排在历史后面，命中率递增到 93%。

文案全部在 ``prompt_manifest.toml`` 列出的文件里：``prompts/system/`` 是稳定指令
（SystemMessage），``prompts/messages/`` 是运行时块（HumanMessage）。本模块只负责
挑数据、按 manifest 的 order 拼 system、按显式顺序拼消息列表，不再自带任何提示词
散文。段边界、顺序、每段多久变一次都写在清单里 —— 这是「为什么它在这」的说明书。

两处例外：元素行 / OCR 行的编号契约要和工具侧的反解共用，不能拆两份；工具表的
「说明」列也不在模板里 —— 工具是否出现由 ``executor_wrappers(ctx)`` 决定，说明文字
必须和它一起生成，否则开关一变，表里就会留下调不到的工具。

不要 bind_tools 那些设备工具。它们的 JSON schema 有 4,000+ token，而且不参与
缓存键（实测），所以每轮都按满价计费。Executor 的工具用法靠
``prompts/system/40-tools.md`` 里的文本工具表教，静态，可缓存。

屏幕描述只有一种形态：统一元素表。Contextor 把所有平台的数据都建成
``state.latest_screen_elements`` —— 层级树形状认识时是 UI 行 + OCR 行，
iOS / 云设备这类不认识的形状只有 OCR 行。模型一律填 ``Target.index``，
工具侧 ``tools/utils.resolve_target`` 按同一张表还原；两边判据都是
``has_element_table``。

一条感知是元素表（UI 行 + OCR 行）+ **按需视觉主体列表**。
视觉主体由 Executor 自己调用 ``scan_visual_subjects`` 触发（见
``prompts/system/30-vision.md`` 的 Visual perception 段）；框架只负责把最近一次
扫描的缓存渲染成一块表，并在屏幕变化后清掉它。

屏幕截图不进模型。prompt 里没有任何图，模型只有两条文字通道：元素表、
视觉主体列表。

工具结果截断。局部历史是 prompt 里唯一随步数线性增长的一段，而工具结果可能
很长（读笔记 = 整条笔记回显、输入文本 = 原文回显）。所以发给 LLM 的那份局部历史
里，过长的成功结果会被截成「首句 … 末句」，上限 120 token。错误结果逐字保留：
报错那部分模型要靠它判断「为什么失败」，压了就等于让它盲猜。

只截 Executor 局部历史这一路。Reporter 走的是 ``state.executor_messages``
（原对象），守「事实不能压」的纪律（``utils/thoughts_compaction.py`` 的模块
文档），一个字都不动。
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from minitap.mobile_use.agents.executor.types import ExecutorReport
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.prompt_language import PromptLanguage, load_prompt_template, pick
from minitap.mobile_use.services.vision import VisionElement, render_vision_rows
from minitap.mobile_use.tools.index import executor_wrappers, get_tools_from_wrappers
from minitap.mobile_use.utils.failed_targets import describe_target
from minitap.mobile_use.utils.message_content import (
    content_to_text,
    has_image_block,
    with_text_replaced,
)
from minitap.mobile_use.utils.screen_elements import ScreenElement, render_screen_rows
from minitap.mobile_use.utils.thoughts_compaction import split_sentences
from minitap.mobile_use.utils.time import format_task_started_at

EXECUTOR_PROMPTS_DIR = Path(__file__).parent / "prompts"
EXECUTOR_PROMPT_MANIFEST = Path(__file__).parent / "prompt_manifest.toml"


@dataclass(frozen=True)
class _PromptSection:
    """``[[system]]`` 的一行：稳定指令的一段。"""

    name: str
    order: int
    file: str
    when: str | None = None


@dataclass(frozen=True)
class _PromptMessage:
    """``[[message]]`` 的一行：运行时块。

    顺序不在这里 —— 它由 :func:`build_executor_messages` 的显式序列决定。
    这里只声明它的文件与生命周期（``volatility``）。
    """

    name: str
    file: str
    volatility: str


#: 允许出现在 ``[[system]].when`` 里的开关名。写错就是「这段永远不出现」，必须挡住。
_WHEN_KEYS = frozenset({"vision_enabled"})

#: 消息块的生命周期标签。清单里只做声明与校验，不参与排序。
_VOLATILITIES = frozenset({"per-task", "per-goal", "per-turn", "occasional", "terminal"})


def _load_manifest() -> tuple[tuple[_PromptSection, ...], dict[str, _PromptMessage]]:
    """读取并校验 ``prompt_manifest.toml``。

    坏清单在 import 时就抛：重名、重复 order、文件缺失、未知的 ``when`` /
    ``volatility`` 都不该等到第一次真机任务才发现。
    """
    data = tomllib.loads(EXECUTOR_PROMPT_MANIFEST.read_text(encoding="utf-8"))

    sections: list[_PromptSection] = []
    seen_names: set[str] = set()
    seen_orders: set[int] = set()
    for row in data.get("system", []):
        name = str(row["name"])
        order = int(row["order"])
        rel = str(row["file"])
        if name in seen_names:
            raise ValueError(f"prompt_manifest.toml: system 段 {name!r} 重名")
        if order in seen_orders:
            raise ValueError(f"prompt_manifest.toml: system 段 order={order} 重复")
        if not (EXECUTOR_PROMPTS_DIR / rel).is_file():
            raise ValueError(f"prompt_manifest.toml: system 段 {name!r} 的文件不存在：{rel}")
        when = row.get("when")
        if when is not None and when not in _WHEN_KEYS:
            raise ValueError(f"prompt_manifest.toml: system 段 {name!r} 的 when={when!r} 未知")
        seen_names.add(name)
        seen_orders.add(order)
        sections.append(_PromptSection(name=name, order=order, file=rel, when=when))

    messages: dict[str, _PromptMessage] = {}
    for row in data.get("message", []):
        name = str(row["name"])
        rel = str(row["file"])
        volatility = str(row["volatility"])
        if name in messages:
            raise ValueError(f"prompt_manifest.toml: message 块 {name!r} 重名")
        if volatility not in _VOLATILITIES:
            raise ValueError(
                f"prompt_manifest.toml: message 块 {name!r} 的 volatility={volatility!r} 未知"
            )
        if not (EXECUTOR_PROMPTS_DIR / rel).is_file():
            raise ValueError(f"prompt_manifest.toml: message 块 {name!r} 的文件不存在：{rel}")
        messages[name] = _PromptMessage(name=name, file=rel, volatility=volatility)

    sections.sort(key=lambda section: section.order)
    return tuple(sections), messages


# 清单是随包发布的静态资源：import 时就校验一次，坏清单不会等到第一次真机任务才炸。
_SECTIONS, _MESSAGES = _load_manifest()


def _render(path: Path, values: Mapping[str, object], *, language: PromptLanguage) -> str:
    """渲染一个模板文件；语言按文件逐段回落（见 ``load_prompt_template``）。"""
    return load_prompt_template(path, language).render(**values).strip()


def render_system(
    values: Mapping[str, object],
    *,
    language: PromptLanguage,
) -> str:
    """按 manifest 的 order 拼接稳定指令。

    ``when`` 为真的段才参与；渲染为空的段自然消失（与 DSH 的 empty section
    语义一致）。段与段之间空一行。
    """
    chunks: list[str] = []
    for section in _SECTIONS:
        if section.when is not None and not values.get(section.when):
            continue
        text = _render(EXECUTOR_PROMPTS_DIR / section.file, values, language=language)
        if text:
            chunks.append(text)
    return "\n\n".join(chunks)


def _render_message(
    name: str,
    *,
    language: PromptLanguage,
    **values: object,
) -> str:
    """渲染一个运行时块；名字必须是 manifest 里声明过的。"""
    message = _MESSAGES[name]
    return _render(EXECUTOR_PROMPTS_DIR / message.file, values, language=language)


#: 工具表的「说明」列（英文在前、中文在后）。
#:
#: 工具**是否出现**由 ``executor_wrappers(ctx)`` 决定 —— 录屏工具跟
#: ``video_recording_enabled`` 走，视觉扫描跟 ``perception.vision`` 走。所以整张表
#: 必须在 Python 里和注册集合一起生成；手写在模板里，开关一变就会漂移，模型会照着
#: 表调一个不存在的工具。
_TOOL_NOTES: dict[str, tuple[str, str]] = {
    "launch_app": ("Open an app", "打开应用"),
    "open_link": ("Handles deep links correctly", "能正确处理深链"),
    "tap": ("Tap", "点击"),
    "long_press_on": ("Long press", "长按"),
    "focus_and_input_text": (
        "Focus then type; verify if feedback is empty",
        "先聚焦再输入，如果反馈为空要核实",
    ),
    "focus_and_clear_text": (
        "If it fails: long press -> select all -> `erase_one_char`",
        "如果失败：长按 -> 全选 -> `erase_one_char`",
    ),
    "swipe_percentages": (
        "**Prefer this**; correct across screen sizes",
        "**优先用它**；跨屏幕尺寸都正确",
    ),
    "swipe_coordinates": (
        "Only for controls needing precision, e.g. sliders",
        "只用于需要精确控制的位置，比如滑块",
    ),
    "stop_app": ("Close the app", "关闭应用"),
    "press_key": (
        "`Enter` confirm/search, `Home` go home, `Back` go back",
        "`Enter` 确认/搜索、`Home` 回到桌面、`Back` 返回",
    ),
    "wait_for_delay": ("Wait that many milliseconds", "等待整数毫秒"),
    "back": ("—", "—"),
    "erase_one_char": ("—", "—"),
    "wake_screen": ("Turn the screen on", "点亮屏幕"),
    "start_video_recording": (
        "Starts a background screen recording: max 3 min (Android) / 15 min (iOS), "
        "no audio; runs until `stop_video_recording`",
        "开始后台录屏：最长 3 分钟（Android）/ 15 分钟（iOS），不录音频；"
        "一直录到调用 `stop_video_recording`",
    ),
    "stop_video_recording": (
        'Stops the recording, then analyzes the video; `prompt` says what to extract '
        '(e.g. "what happens after each 10s")',
        "停止录制，然后分析视频；`prompt` 说明要提取什么（例如「每 10 秒发生了什么」）",
    ),
    "scan_visual_subjects": (
        "Looks at the current screen and lists **every** visual subject it can see, "
        "numbered; then pick one with `vision_index`",
        "看当前屏幕，把能看到的**每一个**视觉主体编号列出；下一轮用 `vision_index` 选一个",
    ),
}

#: 不进「参数」列的参数：``agent_thought`` 是 ``ExecutorOutput`` 的顶层字段
#: （dispatcher 统一注入），其余是 LangChain 的注入参数。
_HIDDEN_TOOL_ARGS = frozenset(
    {"agent_thought", "tool_call_id", "state", "store", "runtime", "config"}
)


def _required_tool_args(tool: BaseTool) -> set[str]:
    """模型可见的必填参数（判据和 dispatcher 的 ``_required_args`` 同源）。"""
    schema = tool.args_schema
    fields = getattr(schema, "model_fields", None)
    if not fields:
        return set()
    return {
        name
        for name, field in fields.items()
        if field.is_required() and name not in _HIDDEN_TOOL_ARGS
    }


def _tool_note(tool: BaseTool, *, language: PromptLanguage) -> str:
    """一个工具的说明；没登记说明时退回工具自己的第一行描述。"""
    notes = _TOOL_NOTES.get(tool.name)
    if notes is not None:
        return pick(*notes, language=language)
    lines = (tool.description or "").strip().splitlines()
    return lines[0] if lines else ""


def format_executor_tools_table(ctx: MobileUseContext, *, language: PromptLanguage) -> str:
    """按**实际注册**的工具生成文本工具表（工具 / 参数 / 说明）。

    和 ``executor_wrappers(ctx)`` 同一份集合：这次能调到的工具，一行不多、一行不少。
    参数名从 ``BaseTool.args`` 取，必填/可选从 ``args_schema`` 的字段定义取，
    所以工具签名改了，表会跟着改。
    """
    tools = get_tools_from_wrappers(ctx, executor_wrappers(ctx))
    optional = pick("optional", "可选", language=language)
    none = pick("*(none)*", "*（无）*", language=language)

    rows: list[str] = []
    for tool in sorted(tools, key=lambda item: item.name):
        required = _required_tool_args(tool)
        args = [
            f"`{name}`" if name in required else f"{optional} `{name}`"
            for name in tool.args
            if name not in _HIDDEN_TOOL_ARGS
        ]
        rows.append(
            f"| `{tool.name}` | {', '.join(args) or none} | {_tool_note(tool, language=language)} |"
        )
    return "\n".join(rows)


def render_executor_system(ctx: MobileUseContext) -> str:
    """渲染 Executor 的 system prompt（稳定指令，整场运行逐字节不变）。

    目标不在 system 里。目标是每子目标变的，放进 system 会让整个 system 块
    在换子目标时全部重算。目标走独立的消息块（``goal``，见
    ``build_executor_messages`` 的第 2 项），子目标内它是恒定的。

    ``vision_enabled`` 是感知开关的状态：一次运行里恒定，所以它只决定
    ``30-vision.md`` 装不装，不伤缓存。条件判断留在 Python，模板里没有
    ``{% if %}``。

    语言跟着 ``ctx.prompt_language`` 走；每个段独立回落 —— 缺哪段的中文版，
    只有那一段变英文（见 ``load_prompt_template``）。
    """
    return render_system(
        {
            "platform": ctx.device.mobile_platform.value,
            "executor_tools_table": format_executor_tools_table(
                ctx, language=ctx.prompt_language
            ),
            "vision_enabled": ctx.perception.vision,
        },
        language=ctx.prompt_language,
    )


def render_task_started(task_started_at: str, *, language: PromptLanguage) -> str:
    """任务开始时间（整场运行不变，和 system 一起构成可缓存前缀）。"""
    return _render_message("task_started", language=language, task_started_at=task_started_at)


def render_goal_message(goal: str, *, language: PromptLanguage) -> str:
    """目标陈述块（子目标内恒定，排在局部历史之前才能被缓存覆盖）。"""
    return _render_message("goal", language=language, goal=goal)


def render_previous_attempt(report: ExecutorReport, *, language: PromptLanguage) -> str:
    """上一轮尝试的事实摘要（Cortex 在轮起点注入，是新一轮局部历史的第一条）。

    为什么需要它：局部历史按「轮」清空（一轮 = 一次执行尝试）。清空挡住了跨轮
    的线性增长，但模型也会忘掉「哪条路已经走不通」—— 上一轮汇报里的 summary、
    结束原因和失败目标正好是这个信息。压成一条消息带过来，它的长度只跟失败条数
    有关，不随步数增长。

    失败目标用 ``describe_target``（**不带 index**）：索引每轮重编号，带进新一轮
    会让模型去点一个完全无关的元素（见 ``utils/failed_targets.py`` 的说明）。
    """
    bits: list[str] = []
    for target in report.failed_targets or []:
        identity = describe_target(label=target.label or None, resource_id=target.resource_id)
        bits.append(f"{identity} ({target.reason[:80]})" if target.reason else identity)
    return _render_message(
        "previous_attempt",
        language=language,
        outcome=report.outcome,
        summary=report.summary,
        reason=report.failure_reason or report.blocked_reason or "",
        failed_targets="; ".join(bits),
    )


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
    return _render_message(
        "device_state",
        language=language,
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
    return _render_message("screen_elements", language=language, rows=rows)


def render_dispatch_rejection(reason: str, *, language: PromptLanguage) -> str:
    """动作没被派发时的回灌（不是 ToolMessage：那份没有对应的 tool_call）。"""
    return _render_message("dispatch_rejection", language=language, reason=reason)


def render_vision_subjects_message(
    elements: list[VisionElement] | None, *, language: PromptLanguage
) -> str:
    """最近一次按需扫描的视觉主体表；没有缓存时返回空串。

    编号由 ``render_vision_rows`` 生成 —— 它和 ``resolve_vision_index`` 是
    同一条编号契约的两端，模板只加表头，编号一个字节都不动。
    """
    rows = render_vision_rows(elements or [])
    if not rows:
        return ""
    return _render_message("vision_subjects", language=language, rows=rows)


def render_retry_nudge(reason: str | None, *, language: PromptLanguage) -> str:
    """解析失败后重试一次时的提醒，必须带上具体的失败原因。"""
    return _render_message("retry", language=language, reason=reason)


def render_extra_actions_notice(count: int, *, language: PromptLanguage) -> str:
    """一轮只执行第一个动作时，告诉模型其余动作没有被执行。"""
    return _render_message("extra_actions", language=language, count=count)


def render_step_limit_message(limit: int, *, language: PromptLanguage) -> str:
    """步数上限触发时的收尾文案。"""
    return _render_message("step_limit", language=language, limit=limit)


def render_no_goal_message(*, language: PromptLanguage) -> str:
    """没有子目标时的收尾文案。"""
    return _render_message("no_goal", language=language)


def render_done_fallback(*, language: PromptLanguage) -> str:
    """``done`` 但没有 done_reason / agent_thought 时的收尾文案。"""
    return _render_message("done_fallback", language=language)


def render_failed_fallback(*, language: PromptLanguage) -> str:
    """``failed`` 但没有 failed_reason / agent_thought 时的收尾文案。"""
    return _render_message("failed_fallback", language=language)


def render_stall_notice(
    *,
    kind: str,
    repeats: int,
    window: int,
    fingerprint: str,
    vision_enabled: bool,
    language: PromptLanguage,
) -> str:
    """「没有进展」的提醒块。

    它是信息，不是限制：这块只出现在「检测到重复」的那一轮，不截断、不改结果，
    也不替模型做决定（见 ``agents/executor/progress.py``）。它告诉模型两件事：
    别重复刚才那个动作；做不成就用 ``failed`` 说出来。

    文案（含中英两种句式）住在模板里，``StallSignal`` 只提供事实字段 —— 以前
    描述句硬编码在 Python 里，英文 prompt 会混进一句中文。``vision_enabled``
    控制最后一段：视觉关掉时 ``scan_visual_subjects`` 没有注册，不能教模型调它。
    """
    return _render_message(
        "stall_notice",
        language=language,
        kind=kind,
        repeats=repeats,
        window=window,
        fingerprint=fingerprint,
        vision_enabled=vision_enabled,
    )


def has_element_table(screen_elements: list[ScreenElement] | None) -> bool:
    """本屏有没有统一元素表。

    所有平台都由 Contextor 建同一张表（层级树不认识的平台只有 OCR 行），
    所以模型一律填 ``Target.index``；工具侧 ``tools/utils.resolve_target``
    看的是同一份 ``state.latest_screen_elements``。两边判据必须一致。

    抽成独立函数而不是内联 ``if screen_elements:``，是为了让测试能直接断言判据
    本身，不用去拼整份 prompt 再 grep。
    """
    return bool(screen_elements)


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
        # 视觉主体表就是模型要读的东西本身：它不是冗长回显，截成「首句 … 末句」
        # 会把 20 行编号表砍得只剩两行，模型就没法按编号选了。
        "scan_visual_subjects",
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
    local_messages: list[AnyMessage],
    device_state_lines: list[str],
    vision_elements: list[VisionElement] | None = None,
    stall_notice: str | None = None,
) -> list[BaseMessage]:
    """组装 Executor 每一轮的 prompt。

    顺序上每轮变的一律靠后：

    1. SystemMessage：``render_system`` 按 manifest 拼出的稳定指令。
    2. HumanMessage：目标陈述（子目标内恒定）。
    3. 局部历史：动作 + 工具反馈，只追加。
    4. HumanMessage：设备状态（每轮变，前台 app 会变）。
    5. HumanMessage：「没有进展」提醒（偶发，只在检测到重复的那一轮）。
    6. HumanMessage：本屏视觉主体表（只在最近一次按需扫描的缓存还有效时出现）。
    7. HumanMessage：屏幕元素表（每轮变，重编号；**必须最后一块**）。

    顺序直接等于缓存命中率。前缀缓存只覆盖从头开始的连续前缀，任何每轮变的内容
    排在局部历史之前，它之后的整段历史每轮都要满价重算。真机实测（2026-10-21）：
    设备状态卡在历史前面时，executor 的缓存命中率只有 15.8%（六个 agent 里最差，
    而它调用量最大）。

    屏幕元素表必须是最后一块：``test_t18b_the_last_message_is_the_element_table``
    守着这条。视觉主体表同样每轮可能变，但它是扫描工具的回执配套，放在元素表
    之前即可。

    屏幕截图这条通道已整条移除（含按需工具 ``take_screenshot``），prompt 里不再
    有任何图。视觉主体表是**文字**：编号 + 描述 + 框，图只进一次独立的小调用。

    Args:
        ctx: 运行上下文。
        goal: 本次子目标的目标陈述。
        screen_elements: 当屏的统一元素表（Contextor 建的，当屏的；UI 行和 OCR
            行共用一套编号）。空或 None 表示这一屏没有任何可寻址的行。
        local_messages: 本轮内的局部历史。发给 LLM 的那份会按需截断过长的
            成功工具结果（见 :func:`truncate_local_history`），``state`` 里的原
            对象不动，Reporter 读的是原对象。
        device_state_lines: 设备状态行。
        vision_elements: 最近一次 ``scan_visual_subjects`` 的缓存。只在非空且
            屏幕没变时出现，教模型用 ``vision_index`` 选主体。

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
            HumanMessage(content=render_task_started(task_started_at, language=ctx.prompt_language))
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
    # 真机实测（2026-10-21）：设备状态里「前台 app」几乎每次导航都变，卡在历史
    # 前面时，executor 的缓存命中率只有 15.8%（六个 agent 里最差，而它调用量
    # 最大）。挪到历史之后，历史那段就能被缓存覆盖。
    #
    # 元素表仍然排在最后，它每轮重编号，永不缓存；设备状态放在它前面、
    # 局部历史之后。

    if device_state_lines:
        messages.append(HumanMessage(content="\n".join(device_state_lines)))

    # 「没有进展」的提醒放在这里（历史之后、元素表之前）。它是偶发的，只在检测到
    # 重复的那一轮出现，但出现时每轮都变，所以同样不能排在局部历史前面。
    if stall_notice:
        messages.append(HumanMessage(content=stall_notice))

    # 视觉主体表：最近一次按需扫描的缓存。它和元素表一样每轮可能变，
    # 但元素表必须保持最后一块（索引契约 + 缓存顺序都有测试守着），所以它放前面。
    vision_text = render_vision_subjects_message(vision_elements, language=ctx.prompt_language)
    if vision_text:
        messages.append(HumanMessage(content=vision_text))

    # 元素表每轮都变，所以放最后。判据统一走 has_element_table，提示侧和工具侧
    # 必须用同一个判据。
    elements = screen_elements or []
    if has_element_table(elements):
        screen_text = render_screen_message(elements, language=ctx.prompt_language)
        if screen_text:
            messages.append(HumanMessage(content=screen_text))

    return messages
