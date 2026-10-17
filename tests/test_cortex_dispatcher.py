"""`agents/cortex/dispatcher.py`：Cortex decisions → tool_calls 的确定性直连。

这个模块的价值全在**一条铁律**上：

    任何一步不确定，就返回 None，让调用方退回 Executor 的 LLM。

所以测试分成两组：
- 组一「该直连的能直连」——证明快路径真的覆盖了常见形状，否则等于没优化。
- 组二「不确定的必须退回」——证明坏数据不会把一整轮搞挂。

下面还有几条**行为等价性**断言：直连发出去的 tool_calls 必须和 Executor LLM
发出去的在语义上等价（同名工具、同样的参数、同样的顺序）。
"""

from __future__ import annotations

from typing import Annotated

import pytest
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState

from minitap.mobile_use.agents.cortex.dispatcher import (
    dispatch_decisions,
    parse_actions,
)


@tool
async def tap(
    agent_thought: str,
    target: dict,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
):
    """Taps on a UI element."""


@tool
async def swipe(
    agent_thought: str,
    swipe_request: dict,
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Swipes."""


@tool
async def back(
    agent_thought: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Goes back."""


@tool
async def launch_app(
    app_name: str,
    agent_thought: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Launches an app."""


@tool
async def wait_for_delay(
    agent_thought: str,
    time_in_ms: int,
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """Waits."""


TOOLS = [tap, swipe, back, launch_app, wait_for_delay]
THOUGHT = "Tapping the send button because the message is already typed."


def _decisions(*actions: dict) -> str:
    import json

    return json.dumps(list(actions), ensure_ascii=False)


class TestShouldDispatch:
    def test_single_tap(self):
        calls = dispatch_decisions(
            _decisions({"action": "tap", "target": {"bounds": {"x": 1, "y": 2, "width": 3, "height": 4}}}),
            THOUGHT,
            TOOLS,
        )
        assert calls is not None
        assert len(calls) == 1
        assert calls[0]["name"] == "tap"
        assert calls[0]["args"]["target"] == {"bounds": {"x": 1, "y": 2, "width": 3, "height": 4}}
        # agent_thought 是所有工具的必填参数，必须被补上
        assert calls[0]["args"]["agent_thought"] == THOUGHT

    def test_action_name_removed_from_args(self):
        """`action` 只是工具名的载体，不能当成参数发过去。"""
        calls = dispatch_decisions(
            _decisions({"action": "back"}), THOUGHT, TOOLS
        )
        assert calls is not None
        assert "action" not in calls[0]["args"]

    def test_multiple_actions_keep_order(self):
        """顺序语义不能丢 —— ToolNode 是按返回顺序串行执行的。"""
        calls = dispatch_decisions(
            _decisions(
                {"action": "tap", "target": {"text": "Search"}},
                {"action": "wait_for_delay", "time_in_ms": 500},
                {"action": "back"},
            ),
            THOUGHT,
            TOOLS,
        )
        assert calls is not None
        assert [c["name"] for c in calls] == ["tap", "wait_for_delay", "back"]

    def test_tool_name_aliases_accepted(self):
        for key in ("action", "tool", "tool_name", "name"):
            calls = dispatch_decisions(_decisions({key: "back"}), THOUGHT, TOOLS)
            assert calls is not None, f"key {key!r} 应该被认成工具名"
            assert calls[0]["name"] == "back"

    def test_per_action_thought_overrides_default(self):
        calls = dispatch_decisions(
            _decisions({"action": "back", "agent_thought": "Going back to the list"}),
            THOUGHT,
            TOOLS,
        )
        assert calls is not None
        assert calls[0]["args"]["agent_thought"] == "Going back to the list"

    def test_calls_get_unique_ids(self):
        calls = dispatch_decisions(
            _decisions({"action": "back"}, {"action": "back"}), THOUGHT, TOOLS
        )
        assert calls is not None
        assert calls[0]["id"] != calls[1]["id"]

    def test_markdown_fenced_json_is_accepted(self):
        raw = '```json\n[{"action": "back"}]\n```'
        assert dispatch_decisions(raw, THOUGHT, TOOLS) is not None

    def test_prose_around_json_is_tolerated(self):
        raw = '好的，我来执行。[{"action": "back"}] 就这样。'
        calls = dispatch_decisions(raw, THOUGHT, TOOLS)
        assert calls is not None
        assert calls[0]["name"] == "back"


class TestShouldFallBack:
    """这些情况必须返回 None，把活儿交回 Executor 的 LLM。"""

    def test_none_and_empty(self):
        assert dispatch_decisions(None, THOUGHT, TOOLS) is None
        assert dispatch_decisions("", THOUGHT, TOOLS) is None
        assert dispatch_decisions("[]", THOUGHT, TOOLS) is None

    def test_malformed_json(self):
        assert dispatch_decisions("{not json", THOUGHT, TOOLS) is None
        assert dispatch_decisions("[{'action': 'back'}]", THOUGHT, TOOLS) is None

    def test_json_object_instead_of_array(self):
        assert dispatch_decisions('{"action": "back"}', THOUGHT, TOOLS) is None

    def test_unknown_tool_name(self):
        assert dispatch_decisions(_decisions({"action": "teleport"}), THOUGHT, TOOLS) is None

    def test_missing_required_arg(self):
        # tap 需要 target，只有 action 不够
        assert dispatch_decisions(_decisions({"action": "tap"}), THOUGHT, TOOLS) is None

    def test_unknown_arg_name(self):
        """Cortex 幻觉出来的参数名必须挡住，否则会当成正常参数发下去。"""
        calls = dispatch_decisions(
            _decisions({"action": "back", "coord_x": 3}), THOUGHT, TOOLS
        )
        assert calls is None

    def test_object_param_given_a_string(self):
        """target 声明是 object，传字符串说明形状不对。"""
        assert dispatch_decisions(
            _decisions({"action": "tap", "target": "the send button"}), THOUGHT, TOOLS
        ) is None

    def test_no_agent_thought_available(self):
        """没有 decisions_reason 又没写 agent_thought 时发不出去。"""
        assert dispatch_decisions(_decisions({"action": "back"}), None, TOOLS) is None
        assert dispatch_decisions(_decisions({"action": "back"}), "   ", TOOLS) is None

    def test_one_bad_action_poisons_the_whole_batch(self):
        """半个批次直发出去比全退回更危险：ToolNode 会执行到一半才炸。"""
        calls = dispatch_decisions(
            _decisions({"action": "back"}, {"action": "teleport"}), THOUGHT, TOOLS
        )
        assert calls is None


class TestParseActions:
    def test_returns_list_of_dicts(self):
        assert parse_actions('[{"action": "back"}]') == [{"action": "back"}]

    def test_rejects_non_dict_items(self):
        assert parse_actions('["back", "tap"]') is None

    def test_bracket_balancing_survives_nested_and_escapes(self):
        """贪婪正则会在嵌套数组/转义引号上截断，配平扫描不会。"""
        raw = '[{"action": "tap", "target": {"text": "he said \\"hi\\"", "tags": ["a","b"]}}]'
        assert parse_actions(raw) == [
            {"action": "tap", "target": {"text": 'he said "hi"', "tags": ["a", "b"]}}
        ]

    @pytest.mark.parametrize("raw", ["", "   ", None, "no json here", "{]"])
    def test_garbage_returns_none(self, raw):
        assert parse_actions(raw) is None


class TestBehaviouralEquivalence:
    def test_injected_params_never_leak_into_args(self):
        """tool_call_id / state 由 LangChain 注入，模型不该也不能提供。"""
        calls = dispatch_decisions(
            _decisions(
                {
                    "action": "tap",
                    "target": {"text": "x"},
                    "tool_call_id": "forged",
                    "state": {"evil": 1},
                }
            ),
            THOUGHT,
            TOOLS,
        )
        # 这两个键不在 tool.args 里，会被判成未知参数 -> 整体退回
        assert calls is None


class TestRequiredArgsDetection:
    def test_scalars_and_objects_both_detected(self):
        """必填参数表是从 args_schema 推的，不能只认对象类型。"""
        ok = dispatch_decisions(
            _decisions({"action": "launch_app", "app_name": "WhatsApp"}), THOUGHT, TOOLS
        )
        assert ok is not None
        assert ok[0]["args"]["app_name"] == "WhatsApp"

        bad = dispatch_decisions(_decisions({"action": "launch_app"}), THOUGHT, TOOLS)
        assert bad is None
