"""结构化输出：**必须核对 tool_call 的工具名**。

这个文件锁的是一个真机 bug（2026-10-22，执行 812 / 快手极速版 / Redmi Turbo 4）。

发生了什么
----------
Executor 每轮只绑一个工具（``ExecutorOutput``），而且 ``tool_choice`` 是**强制**的。
但 MiniMax-M3 经常无视 ``tool_choice``，直接喊 prompt「工具表」里的**真实工具名**
（``focus_and_input_text`` / ``tap`` / ``wait_for_delay`` …）—— 66 次调用里有 37 次。

而 ``_ToolCallStructuredRunnable._parse`` 以前**不查名字**，拿第一个 tool_call 的
args 直接 ``model_validate``。``ExecutorOutput`` 的字段全都有默认值
（``actions`` 默认 ``[]``）、pydantic 默认 ``extra='ignore'`` ⇒ **校验「通过」** ⇒
产出一个 ``actions=[]`` 的假结果。

后果链：

    模型喊错工具名 → 被吞成空动作 → dispatcher 判「空动作非法」→ 回灌
    「you returned an empty action list」→ 模型不知道错在哪 → 换个姿势再喊一次
    → **连着 8 轮零动作**

实测代价：42/66 次调用是空转（243 秒，占 executor LLM 时长 41%）。

修法
----
``_parse`` 只接受**名字对得上**的 tool_call；名字不对就记下来，最后抛
``StructuredOutputError``，理由里**点名**喊错的工具名 —— 上层把它讲回模型，
模型一轮就能改对。

这里用的 ``REAL_*`` 样本是**执行 812 的原始响应**（从 manager-server 的
``llm_prompts`` 表里取的），不是编的。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minitap.mobile_use.agents.executor.types import ExecutorOutput  # noqa: E402
from minitap.mobile_use.services.llm import (  # noqa: E402
    StructuredOutputError,
    _ToolCallStructuredRunnable,
)


# ---------------------------------------------------------------------------
# 执行 812 的真实响应（逐字来自 manager-server `llm_prompts` 表）
# ---------------------------------------------------------------------------
# 每一条都是「模型喊了真实工具名而不是 ExecutorOutput」的现场。
# 覆盖 6 个不同的错名：tap / wait_for_delay / swipe_percentages /
# swipe_coordinates / focus_and_input_text / focus_and_clear_text。
REAL_WRONG_TOOL_CALLS: list[tuple[str, dict]] = [
    (
        "focus_and_clear_text",
        {
            "agent_thought": (
                "I'm on the search page; the search field already shows "
                '"传奇手游合集" — I\'ll clear it and type the new term "传奇游戏".'
            ),
            "target": {"index": "18"},
        },
    ),
    (
        "focus_and_input_text",
        {
            "agent_thought": (
                "The keyboard is up and the field is focused; typing the search "
                'term "传奇游戏" now.'
            ),
            "target": {"index": 18},
            "text": "传奇游戏",
        },
    ),
    (
        "focus_and_input_text",
        {
            "agent_thought": (
                'The search field still shows "传奇手游合集"; I\'ll replace it by '
                'typing the required keyword "传奇游戏".'
            ),
            "target": {"index": 18, "text": "传奇游戏"},
        },
    ),
    (
        "swipe_percentages",
        {
            "agent_thought": (
                "The tab row shows 综合/视频(?)/商品/智創 but no 广告 tab visible. "
                'I\'ll scroll the tab strip to look for a "广告" tab.'
            ),
            "end_x_percent": 80,
            "end_y_percent": 24,
            "start_x_percent": 90,
            "start_y_percent": 24,
        },
    ),
    (
        "swipe_coordinates",
        {
            "agent_thought": (
                'I need to find the "广告" tab; the tab strip is a '
                "HorizontalScrollView (row 23), so I'll swipe it left to reveal "
                "more tabs."
            ),
            "end_x": 200,
            "end_y": 344,
            "start_x": 1000,
            "start_y": 344,
        },
    ),
    (
        "wait_for_delay",
        {
            "agent_thought": "The results may still be loading; wait a moment.",
            "time_in_ms": 2000,
        },
    ),
    (
        "tap",
        {
            "agent_thought": "Tapping the 广告 filter tab to isolate ad videos.",
            "target": {"index": 39},
        },
    ),
]


def _msg_with_tool_call(name: str, args: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": "call_1", "type": "tool_call"}],
    )


class _FakeLLM:
    """只会回放一条固定响应的假客户端。"""

    structured_output_strategy = "tool"

    def __init__(self, msg: AIMessage):
        self._msg = msg

    def bind_tools(self, *args, **kwargs):
        return self

    def invoke(self, *args, **kwargs):
        return self._msg

    async def ainvoke(self, *args, **kwargs):
        return self._msg


def _runner(msg: AIMessage) -> _ToolCallStructuredRunnable:
    # cast：假客户端只实现了 `_ToolCallStructuredRunnable` 真正用到的那几个
    # 方法（bind_tools / invoke），不是 ChatOpenAI。运行时没问题，类型上
    # 只能用 cast 过 —— 这里不引入真的 ChatOpenAI（要 key、要网络）。
    return _ToolCallStructuredRunnable(cast(ChatOpenAI, _FakeLLM(msg)), ExecutorOutput)


def _parse(msg: AIMessage) -> ExecutorOutput:
    """跑一次解析并把结果窄化成 ``ExecutorOutput``。"""
    out = _runner(msg).invoke([])
    assert isinstance(out, ExecutorOutput), f"解析结果类型不对：{type(out).__name__}"
    return out


def _as_ai(message: AnyMessage) -> AIMessage:
    assert isinstance(message, AIMessage), f"期望 AIMessage，实际 {type(message).__name__}"
    return message


def _as_tool(message: AnyMessage) -> ToolMessage:
    assert isinstance(message, ToolMessage), f"期望 ToolMessage，实际 {type(message).__name__}"
    return message


# ---------------------------------------------------------------------------
# 1) 核心回归：喊错工具名**不许**被静默吞成空动作
# ---------------------------------------------------------------------------


def test_real_wrong_tool_call_is_not_swallowed_into_empty_actions():
    """真机现场：喊 `focus_and_input_text` ⇒ 以前得到 ``actions=[]``（动作被吞）。

    现在必须**报错**，而且报错里要**点名**喊错的工具名。
    """
    msg = _msg_with_tool_call(*REAL_WRONG_TOOL_CALLS[1])
    with pytest.raises(StructuredOutputError) as excinfo:
        _runner(msg).invoke([])

    reason = excinfo.value.reason
    assert "focus_and_input_text" in reason, f"理由必须点名喊错的工具名：{reason}"
    assert "ExecutorOutput" in reason, f"理由必须说清该调什么：{reason}"
    # 旧的病：被吞成空动作，外面只看见「你返回了空动作列表」
    assert "empty action list" not in reason


@pytest.mark.parametrize(("wrong_name", "args"), REAL_WRONG_TOOL_CALLS)
def test_every_real_wrong_tool_call_now_raises(wrong_name, args):
    """37 条真机错名（这里取 7 个代表）必须全部**响亮地失败**，不能有一条被吞。"""
    with pytest.raises(StructuredOutputError) as excinfo:
        _runner(_msg_with_tool_call(wrong_name, args)).invoke([])
    assert wrong_name in excinfo.value.reason


def test_wrong_tool_call_no_longer_yields_an_empty_action_list():
    """旧行为的精确复现：以前这条会**返回**一个 actions=[] 的假结果。

    这条断言是给「以后谁把 name 校验删了」留的哨兵。
    """
    msg = _msg_with_tool_call("tap", {"target": {"index": 39}})
    runner = _runner(msg)
    try:
        result = runner.invoke([])
    except StructuredOutputError:
        return  # 期望路径
    # 没抛 = 病又回来了
    assert isinstance(result, ExecutorOutput)
    assert result.actions != [], "错名调用又被吞成了空动作"
    pytest.fail("错名调用没有抛异常，动作又被静默吞掉了")


# ---------------------------------------------------------------------------
# 2) 正常路径不许被误伤
# ---------------------------------------------------------------------------


def test_correct_tool_name_still_parses():
    msg = _msg_with_tool_call(
        "ExecutorOutput",
        {
            "actions": [{"action": "tap", "target": {"index": 39}}],
            "agent_thought": "tapping the ad tab",
            "done": False,
        },
    )
    out = _parse(msg)
    assert out.actions == [{"action": "tap", "target": {"index": 39}}]
    assert out.agent_thought == "tapping the ad tab"


def test_mixed_tool_calls_prefer_the_schema_tool():
    """模型同时喊了错的和对的 ⇒ 用**对的**那个，不要因为有一个错名就整体失败。"""
    msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "tap", "args": {"target": {"index": 1}}, "id": "c1", "type": "tool_call"},
            {
                "name": "ExecutorOutput",
                "args": {"actions": [{"action": "tap", "target": {"index": 39}}]},
                "id": "c2",
                "type": "tool_call",
            },
        ],
    )
    out = _parse(msg)
    assert out.actions == [{"action": "tap", "target": {"index": 39}}]


def test_markdown_json_answer_still_works():
    """content 路径（模型回 markdown JSON 而不是调工具）不许被这次改动打死。"""
    msg = AIMessage(
        content='```json\n{"actions": [{"action": "tap", "target": {"index": 7}}]}\n```'
    )
    out = _parse(msg)
    assert out.actions == [{"action": "tap", "target": {"index": 7}}]


# ---------------------------------------------------------------------------
# 3) 理由分层：三种失败要说三句不同的话
# ---------------------------------------------------------------------------


def test_prose_answer_reason_mentions_prose():
    msg = AIMessage(content='The field now shows "传奇游戏"; tapping the 搜索 button.')
    with pytest.raises(StructuredOutputError) as excinfo:
        _runner(msg).invoke([])
    reason = excinfo.value.reason
    assert "prose" in reason.lower()
    assert "ExecutorOutput" in reason


def test_empty_answer_reason_is_about_being_empty():
    msg = AIMessage(content="")
    with pytest.raises(StructuredOutputError) as excinfo:
        _runner(msg).invoke([])
    assert "empty" in excinfo.value.reason.lower()


def test_detail_keeps_the_operator_facing_facts():
    """``reason`` 给模型、``detail`` 给人 —— 两者分开，detail 里要有原文。"""
    msg = _msg_with_tool_call("tap", {"target": {"index": 39}})
    with pytest.raises(StructuredOutputError) as excinfo:
        _runner(msg).invoke([])
    err = excinfo.value
    assert "wrong_tools" in err.detail
    assert "finish_reason" in err.detail
    # reason 是短的，不能把模型的原文塞进去
    assert len(err.reason) < 300


# ---------------------------------------------------------------------------
# 4) executor 层：重试的 nudge 必须带上真实原因
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_executor_retry_nudge_names_the_wrong_tool(monkeypatch):
    """模型喊错工具名 ⇒ 重试的 nudge 里必须点名，否则它会连着重犯（真机 8 轮）。"""
    import minitap.mobile_use.agents.executor.executor as ex_mod

    calls: list = []

    class _FlakyRunner:
        """第一次喊错工具名，第二次改对了（模拟模型照着 nudge 改了）。"""

        def __init__(self):
            self._n = 0

        async def ainvoke(self, messages):
            calls.append(messages)
            self._n += 1
            if self._n == 1:
                raise StructuredOutputError(
                    reason=(
                        "you called the tool 'focus_and_input_text' but the only "
                        "tool you may call is 'ExecutorOutput' — call "
                        "'ExecutorOutput' instead"
                    ),
                    detail="tool_calls=1, wrong_tools=['focus_and_input_text']",
                )
            return ExecutorOutput(
                actions=[{"action": "tap", "target": {"index": 39}}],
                agent_thought="tapping the ad tab",
            )

    monkeypatch.setattr(ex_mod, "get_structured_llm", lambda **kwargs: _FlakyRunner())

    from tests.mobile_use.test_executor_autonomy import _ctx  # noqa: PLC0415

    state = _executor_state()
    node = ex_mod.ExecutorNode(_ctx())
    update = await node._decide(state=state, goal="find the ad tab")

    assert len(calls) == 2, "第一次失败后必须重试一次"
    nudge = calls[1][-1]
    assert isinstance(nudge, HumanMessage)
    assert "focus_and_input_text" in nudge.content, "nudge 必须点名喊错的工具名"
    assert "ExecutorOutput" in nudge.content, "nudge 必须说清该调什么"
    # 重试之后动作要真的出来了
    assert update.actions == [{"action": "tap", "target": {"index": 39}}]


def _executor_state():
    """造一个够 `_decide` 用的最小 state（照抄 `sdk/agent.py` 的初始 state）。"""
    from minitap.mobile_use.graph.state import State
    from tests.mobile_use.test_executor_autonomy import _initial_state

    return State(
        **{
            **_initial_state(),
            "executor_goal": "find the ad tab",
            "executor_steps": 0,
            "total_steps": 0,
        }
    )


# ---------------------------------------------------------------------------
# 6) invalid_tool_calls：模型**确实调了工具**，但 arguments 不是合法 JSON
# ---------------------------------------------------------------------------
#
# langchain 的 ``parse_tool_call`` 解析不了 ``arguments`` 时，会把整个调用丢进
# ``msg.invalid_tool_calls``（⚠️ AIMessage 的**顶层属性**，不在 additional_kwargs 里）。
# 框架以前只读 ``msg.tool_calls`` ⇒ 这类响应被当成「模型什么都没说」
# ⇒ 报 "your answer was empty" ⇒ **一次本来能救的调用凭空消失**。
#
# 实测来源（2026-10-24，DeepSeek deepseek-flash，真实 API）：模型偶尔把
# arguments 包成 markdown 围栏或前后加说明文字，就落进这一路。


def _msg_with_invalid_tool_call(name: str, raw_args: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[],
        invalid_tool_calls=[
            {"name": name, "args": raw_args, "id": "call_1", "error": "invalid json"}
        ],
    )


def test_invalid_tool_call_with_fenced_arguments_is_recovered():
    """arguments 被包在 ```json 围栏里 ⇒ 救回来，不许当「什么都没说」。"""
    msg = _msg_with_invalid_tool_call(
        "ExecutorOutput",
        '```json\n{"actions": [{"action": "tap", "target": {"index": 39}}]}\n```',
    )
    out = _parse(msg)
    assert out.actions == [{"action": "tap", "target": {"index": 39}}]


def test_invalid_tool_call_with_leading_prose_is_recovered():
    """arguments 前后有说明文字 ⇒ 取第一个配平的 {..} 块救回来。"""
    msg = _msg_with_invalid_tool_call(
        "ExecutorOutput",
        '好的，我来调用。{"actions": [{"action": "tap", "target": {"index": 7}}]} 以上。',
    )
    out = _parse(msg)
    assert out.actions == [{"action": "tap", "target": {"index": 7}}]


def test_invalid_tool_call_is_never_reported_as_an_empty_answer():
    """🔴 核心回归：这类响应以前报 "your answer was empty" ⇒ 调用凭空消失。

    现在要么救回结果，要么报「arguments 不是合法 JSON」，**不能**说成「没说话」。
    """
    msg = _msg_with_invalid_tool_call("ExecutorOutput", "{not json at all")
    with pytest.raises(StructuredOutputError) as excinfo:
        _runner(msg).invoke([])
    reason = excinfo.value.reason
    assert "empty" not in reason.lower(), f"不能把「工具调用被丢弃」说成「没说话」：{reason}"
    assert "JSON" in reason, f"要说清是哪一环坏了：{reason}"


def test_invalid_tool_call_with_the_wrong_name_is_reported_by_name():
    """invalid_tool_calls 里的错名也要**点名** —— 模型才知道自己喊错了。"""
    msg = _msg_with_invalid_tool_call("tap", '{"target": {"index": 1}}')
    with pytest.raises(StructuredOutputError) as excinfo:
        _runner(msg).invoke([])
    assert "tap" in excinfo.value.reason


def test_invalid_tool_call_count_is_in_the_operator_facing_detail():
    """``detail`` 是给人看的 ⇒ 必须能看出「有几次调用落在 invalid 里」。"""
    msg = _msg_with_invalid_tool_call("ExecutorOutput", "{not json at all")
    with pytest.raises(StructuredOutputError) as excinfo:
        _runner(msg).invoke([])
    assert "invalid_tool_calls=1" in excinfo.value.detail


# ---------------------------------------------------------------------------
# 8) 守卫：结构化输出用的 schema，工具名不许带下划线前缀
# ---------------------------------------------------------------------------
#
# 工具名 = pydantic 的**类名**（``convert_to_openai_tool`` 用 ``cls.__name__``），
# 所以给 schema 起个 ``_FooOutput`` 这样的名字，模型看到的就是 ``_FooOutput``。
#
# 实测（2026-10-24，DeepSeek deepseek-flash，三组各 20 轮）：

#   工具名                        描述    结果       失败时喊的名字
#   ``_PlannerOutput``（下划线）   空      18/20     ``false`` / ``invoke``
#   ``PlannerOutput``            空      **20/20**  —
#   ``PlannerOutputDoc``         有      **20/20**  —
#
# ⇒ 元凶是**名字**（不是描述 —— 加不加 docstring 测不出差别）。
#
# 生产的 5 个 schema 现在都不带下划线 ⇒ 这条是**哨兵**：以后谁加一个
# ``class _FooOutput``，这里立刻红。
# （reporter 的 ``ReporterSummary`` 已随节点移除：汇报是纯代码，不再有 LLM 输出。）


def _production_structured_output_schemas() -> list[tuple[str, type]]:
    from minitap.mobile_use.agents.contextor.types import ContextorOutput
    from minitap.mobile_use.agents.cortex.types import CortexOutput
    from minitap.mobile_use.agents.executor.types import ExecutorOutput
    from minitap.mobile_use.agents.hopper.hopper import HopperOutput
    from minitap.mobile_use.agents.planner.types import PlannerOutput

    return [
        ("planner", PlannerOutput),
        ("cortex", CortexOutput),
        ("executor", ExecutorOutput),
        ("contextor", ContextorOutput),
        ("hopper", HopperOutput),
    ]


def test_no_production_schema_produces_an_underscore_prefixed_tool_name():
    """🔴 工具名带下划线 ⇒ 模型把名字记错 ⇒ 整次调用报废（实测 2/20）。"""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    for label, schema in _production_structured_output_schemas():
        name = convert_to_openai_tool(schema)["function"]["name"]
        assert not name.startswith("_"), (
            f"{label} 的工具名是 {name!r}（带下划线前缀）—— 实测 DeepSeek 会把这种"
            f"名字记成编出来的名字（false / invoke），整次调用报废。改类名，别改前缀。"
        )


def test_the_tool_name_is_the_class_name():
    """工具名 = 类名 —— 这是「改类名就能改工具名」的前提，别哪天悄悄变了。

    变了的话，``_parse`` 里对账用的 ``self._tool_name`` 和模型看到的名字就可能
    对不上（模型喊的是我们以为的那个，我们等的却是另一个）。
    """
    from langchain_core.utils.function_calling import convert_to_openai_tool

    for label, schema in _production_structured_output_schemas():
        name = convert_to_openai_tool(schema)["function"]["name"]
        assert name == schema.__name__, f"{label}: 工具名 {name!r} != 类名 {schema.__name__!r}"


# ---------------------------------------------------------------------------
# 9) 根因：历史在教模型喊设备工具名 —— 必须改写成 ExecutorOutput 的形状
# ---------------------------------------------------------------------------
#
# 真机实测（执行 812）：executor 66 次调用里 37 次喊错工具名。
# 对照组显示：**历史里还没有设备工具名的那几次，一次都没错** ——
# 因为历史是模型下一轮模仿的对象，它照着历史里的 `{"name": "tap"}` 学，
# 于是喊了 `tap`。


def _history() -> list:
    """一段典型的局部历史：assistant 喊了设备工具，工具回了结果。"""
    from langchain_core.messages import AIMessage, ToolMessage

    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "tap",
                    "args": {
                        "target": {"index": 15},
                        "agent_thought": "tapping the search bar",
                    },
                    "id": "call_abc",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=(
                "Tap on element with resource_id='com.kuaishou.nebula:id/"
                "search_bar_container' -> coordinates (649, 205) was successful."
            ),
            tool_call_id="call_abc",
            name="tap",
        ),
    ]


def test_history_tool_call_is_rewritten_to_executor_output():
    """历史里的 assistant tool_call 必须变成 ``ExecutorOutput``。"""
    from minitap.mobile_use.agents.executor.prompt import rewrite_history_tool_calls

    out = rewrite_history_tool_calls(_history())
    call = _as_ai(out[0]).tool_calls[0]
    assert call["name"] == "ExecutorOutput", f"历史还在教设备工具名：{call['name']}"
    # 历史必须和**模型自己产出的形状**一模一样：actions 条目里没有
    # agent_thought（它在顶层）。长得不一样就是在教模型另一种格式。
    assert call["args"]["actions"] == [{"action": "tap", "target": {"index": 15}}]
    assert call["args"]["agent_thought"] == "tapping the search bar"


def test_rewrite_keeps_tool_call_id():
    """tool_call_id 是端点配对 AIMessage / ToolMessage 的依据，动了会 400。"""
    from minitap.mobile_use.agents.executor.prompt import rewrite_history_tool_calls

    out = rewrite_history_tool_calls(_history())
    assert _as_ai(out[0]).tool_calls[0]["id"] == "call_abc"
    assert _as_tool(out[1]).tool_call_id == "call_abc"


def test_rewrite_aligns_the_tool_message_name():
    """ToolMessage 的 name 跟着对齐（配对靠 id，同名更不容易被端点挑刺）。"""
    from minitap.mobile_use.agents.executor.prompt import rewrite_history_tool_calls

    out = rewrite_history_tool_calls(_history())
    assert _as_tool(out[1]).name == "ExecutorOutput"


def test_rewrite_does_not_mutate_the_state_objects():
    """``state`` 里的原对象不动 —— Reporter 读的是原对象。"""
    from minitap.mobile_use.agents.executor.prompt import rewrite_history_tool_calls

    history = _history()
    rewrite_history_tool_calls(history)
    assert _as_ai(history[0]).tool_calls[0]["name"] == "tap"


def test_rewrite_drops_the_stale_raw_tool_calls_copy():
    """``additional_kwargs`` 里的原始 tool_calls 副本必须一起清掉。

    provider 解析出来的 AIMessage 会在那儿留一份**旧**的副本。langchain 的
    转换器优先用 ``message.tool_calls``，所以留着暂时不出错 —— 但那是两份
    真值来源，改一份忘一份就是真 bug。
    """
    from minitap.mobile_use.agents.executor.prompt import rewrite_history_tool_calls

    history = _history()
    history[0].additional_kwargs["tool_calls"] = [
        {"id": "call_abc", "type": "function", "function": {"name": "tap", "arguments": "{}"}}
    ]
    out = rewrite_history_tool_calls(history)
    assert "tool_calls" not in _as_ai(out[0]).additional_kwargs
    assert _as_ai(out[0]).tool_calls[0]["name"] == "ExecutorOutput"


def test_rewrite_leaves_an_executor_output_call_alone():
    from minitap.mobile_use.agents.executor.prompt import rewrite_history_tool_calls
    from langchain_core.messages import AIMessage

    msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "ExecutorOutput",
                "args": {"actions": [{"action": "tap"}]},
                "id": "c1",
                "type": "tool_call",
            }
        ],
    )
    out = rewrite_history_tool_calls([msg])
    assert _as_ai(out[0]).tool_calls[0]["name"] == "ExecutorOutput"
    assert _as_ai(out[0]).tool_calls[0]["args"] == {"actions": [{"action": "tap"}]}


def test_payload_tool_results_are_still_exempt_from_truncation():
    """**先截断、后改写** —— 载荷豁免靠原工具名判断，顺序反了就会丢信息。"""
    from minitap.mobile_use.agents.executor.prompt import (
        rewrite_history_tool_calls,
        truncate_local_history,
    )
    from langchain_core.messages import AIMessage, ToolMessage

    long_payload = "x" * 5000
    history = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_note",
                    "args": {"key": "k", "agent_thought": "reading"},
                    "id": "c1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content=long_payload, tool_call_id="c1", name="read_note"),
    ]
    truncated = truncate_local_history(history)
    assert truncated[1].content == long_payload, "载荷类工具的结果必须逐字保留"
    rewritten = rewrite_history_tool_calls(truncated)
    assert rewritten[1].content == long_payload, "改写之后载荷也不能被截"


# ---------------------------------------------------------------------------
# 7) 行为等价性：除了「喊错工具名」，新旧解析必须一模一样
# ---------------------------------------------------------------------------
#
# 这一条是给「活体 API 抖动」和「真回归」分家的：
# 改动的**唯一**行为差异是「喊错工具名不再被吞」。
# 别的形态（没有 tool_calls、args 不合 schema、写散文）必须**照样失败**——
# 否则就是把「静默失败」换成了「静默成功」，那是更糟的回归。


def _old_parse(self, msg: AIMessage) -> BaseModel:
    """HEAD 版 `_parse`（原样贴回，只为对照）。"""
    for tc in msg.tool_calls or []:
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
        if args is None:
            continue
        try:
            if isinstance(args, str):
                return self.schema.model_validate_json(args)
            return self.schema.model_validate(args)
        except ValidationError:
            continue

    content = msg.content if isinstance(msg.content, str) else ""
    if not content.strip():
        raise ValueError("old parse: no tool_calls and no content")

    stripped = self._strip_markdown_json(content)
    for candidate in (stripped, self._extract_json_object(stripped)):
        if not candidate:
            continue
        try:
            return self.schema.model_validate_json(candidate)
        except ValidationError:
            continue
    raise ValueError("old parse: unparseable")


def _run_old(msg: AIMessage) -> BaseModel:
    """用 HEAD 版 `_parse` 跑同一条消息。"""
    with patch.object(_ToolCallStructuredRunnable, "_parse", _old_parse):
        return _runner(msg).invoke([])


@pytest.mark.parametrize(
    ("label", "msg"),
    [
        (
            "没有任何 tool_calls 且内容为空",
            AIMessage(content=""),
        ),
        (
            "工具名对但 args 不合 schema",
            _msg_with_tool_call(
                "ExecutorOutput", {"actions": "这不是一个列表", "agent_thought": "x"}
            ),
        ),
        (
            "只有散文且不是 JSON",
            AIMessage(content="The WLAN page is now open... done: true"),
        ),
    ],
)
def test_new_parse_fails_exactly_where_the_old_one_failed(label, msg):
    """这些形态旧实现失败、新实现也必须失败 —— 只有「喊错工具名」是新行为。"""
    with pytest.raises(ValueError):  # 旧实现：ValueError
        _run_old(msg)
    with pytest.raises(StructuredOutputError):  # 新实现：带理由的 ValueError 子类
        _runner(msg).invoke([])


def test_new_parse_succeeds_exactly_where_the_old_one_succeeded():
    """合法调用：旧实现解析出来的结果必须和新实现**逐字相同**。"""
    msg = _msg_with_tool_call(
        "ExecutorOutput",
        {
            "actions": [{"action": "tap", "target": {"index": 39}}],
            "agent_thought": "tapping the ad tab",
            "done": False,
        },
    )
    old = _run_old(msg)
    new = _runner(msg).invoke([])
    assert old == new


# ---------------------------------------------------------------------------
# 6) 提示词不许再教一个运行时不做承诺的行为
# ---------------------------------------------------------------------------


def test_executor_prompt_does_not_promise_empty_actions_is_valid():
    """Executor 的提示词以前写着「空的 actions 是合法答案」，运行时却拒它。

    提示词说合法、运行时判非法 ⇒ 模型在「我卡住了」和「我得给个动作」之间
    横跳。现在提示词必须和运行时一致。

    ⚠️ 2026-10-24：措辞被压缩过（Output 表格里现在是「required unless `done`,
    empty when done」）。断言的是**契约**（空 actions 只在 done 时合法），
    不是原来那一句。

    ⚠️ 2026-10-28：新增 `failed` 出口后，合法收尾变成 `done` 或 `failed`，
    措辞同步成「required unless `done`/`failed`, empty when done or failed」。
    """
    from minitap.mobile_use.agents.executor.prompt import EXECUTOR_PROMPTS_DIR

    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((EXECUTOR_PROMPTS_DIR / "en").rglob("*.md"))
    )
    assert "An empty `actions` is a valid answer" not in text
    assert "required unless `done`/`failed`, empty when done or failed" in text


def test_executor_prompt_says_executor_output_is_the_only_tool():
    """触发根因的那张「工具表」里全是真实工具名 —— 必须明说它们不能直接调。

    ⚠️ 2026-10-24：措辞被压缩过（现在是「Answer only by calling `ExecutorOutput`」）。
    """
    from minitap.mobile_use.agents.executor.prompt import EXECUTOR_PROMPTS_DIR

    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((EXECUTOR_PROMPTS_DIR / "en").rglob("*.md"))
    )
    assert "only by calling `ExecutorOutput`" in text
