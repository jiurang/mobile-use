"""End-to-end tests for ExecutorToolNode inside a compiled LangGraph graph.

ExecutorToolNode relies on private langgraph-prebuilt ToolNode methods, whose signatures
change across minor releases (#213: `_extract_state` gained a required `config` arg in
langgraph-prebuilt 1.1.0, which broke every tool call). These tests run the node through a
real graph so that any such drift fails here when langgraph is bumped.
"""

import asyncio
import operator
from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from minitap.mobile_use.agents.executor.tool_node import ExecutorToolNode

KEY = "executor_messages"


class _State(TypedDict):
    executor_messages: Annotated[list, add_messages]
    agents_thoughts: Annotated[list[str], operator.add]
    marker: str
    # NotRequired：工具注入的是整份 state，缺这个键时不能算 schema 错误
    last_action_at: NotRequired[float]


def _command(tool_call_id: str, ok: bool, thought: str) -> Command:
    return Command(
        update={
            "agents_thoughts": [thought],
            KEY: [
                ToolMessage(
                    tool_call_id=tool_call_id,
                    content="ok" if ok else "boom",
                    status="success" if ok else "error",
                )
            ],
        }
    )


@tool
async def good(
    agent_thought: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[_State, InjectedState],
) -> Command:
    """Succeeds and echoes the injected state's marker."""
    return _command(tool_call_id, True, f"{agent_thought}|{state['marker']}")


@tool
async def bad(
    agent_thought: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[_State, InjectedState],
) -> Command:
    """Always fails."""
    return _command(tool_call_id, False, agent_thought)


@tool
async def multi(
    agent_thought: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[_State, InjectedState],
) -> list[Command]:
    """Returns a state-only Command plus the call's ToolMessage; fails on "fail"."""
    ok = agent_thought != "fail"
    return [
        Command(update={"agents_thoughts": ["first"]}),
        _command(tool_call_id, ok, agent_thought),
    ]


def _call(name: str, i: int) -> dict:
    return {"name": name, "args": {"agent_thought": f"t{i}"}, "id": f"c{i}", "type": "tool_call"}


def _run(tool_calls: list[dict]) -> dict:
    graph = StateGraph(_State)
    graph.add_node("executor_tools", ExecutorToolNode(tools=[good, bad, multi], messages_key=KEY))
    graph.add_edge(START, "executor_tools")
    graph.add_edge("executor_tools", END)
    app = graph.compile()
    return asyncio.run(
        app.ainvoke(
            {
                KEY: [AIMessage(content="", tool_calls=tool_calls)],
                "agents_thoughts": [],
                "marker": "STATE",
            }
        )
    )


def _tool_messages(out: dict) -> list[tuple]:
    return [(m.tool_call_id, m.status, m.content) for m in out[KEY] if isinstance(m, ToolMessage)]


def test_runs_tool_calls_in_order_with_injected_state():
    out = _run([_call("good", 1), _call("good", 2)])

    assert _tool_messages(out) == [("c1", "success", "ok"), ("c2", "success", "ok")]
    assert out["agents_thoughts"] == ["t1|STATE", "t2|STATE"]


def test_aborts_remaining_tool_calls_after_a_failure():
    out = _run([_call("good", 1), _call("bad", 2), _call("good", 3)])

    assert _tool_messages(out) == [
        ("c1", "success", "ok"),
        ("c2", "error", "boom"),
        ("c3", "error", "Aborted: a previous tool call failed!"),
    ]
    assert out["agents_thoughts"] == ["t1|STATE", "t2"]


def test_tool_returning_a_list_of_commands_succeeds():
    out = _run([_call("multi", 1), _call("good", 2)])

    assert _tool_messages(out) == [("c1", "success", "ok"), ("c2", "success", "ok")]
    assert out["agents_thoughts"] == ["first", "t1", "t2|STATE"]


def test_tool_returning_a_list_with_a_failure_aborts_the_rest():
    fail = {"name": "multi", "args": {"agent_thought": "fail"}, "id": "c1", "type": "tool_call"}
    out = _run([fail, _call("good", 2)])

    assert _tool_messages(out) == [
        ("c1", "error", "boom"),
        ("c2", "error", "Aborted: a previous tool call failed!"),
    ]


# ---------------------------------------------------------------------------
# 动过屏幕的动作要记下时刻（感知前的沉降等待靠它）
# ---------------------------------------------------------------------------


@tool
async def tap(
    agent_thought: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[_State, InjectedState],
) -> Command:
    """A screen-changing tool."""
    return _command(tool_call_id, True, agent_thought)


@tool
async def save_note(
    agent_thought: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[_State, InjectedState],
) -> Command:
    """A bookkeeping tool: it does not touch the screen."""
    return _command(tool_call_id, True, agent_thought)


def _run_stamp(tool_calls: list[dict]) -> dict:
    graph = StateGraph(_State)
    graph.add_node(
        "executor_tools",
        ExecutorToolNode(tools=[tap, save_note], messages_key=KEY),
    )
    graph.add_edge(START, "executor_tools")
    graph.add_edge("executor_tools", END)
    app = graph.compile()
    return asyncio.run(
        app.ainvoke(
            {
                KEY: [AIMessage(content="", tool_calls=tool_calls)],
                "agents_thoughts": [],
                "marker": "STATE",
            }
        )
    )


def test_a_screen_changing_tool_stamps_the_action_time():
    """tap 之后必须留下动作时刻，否则感知前的沉降等待无从判断。"""
    out = _run_stamp([_call("tap", 1)])
    assert out.get("last_action_at") is not None
    assert out["last_action_at"] > 0


def test_a_bookkeeping_tool_does_not_stamp_the_action_time():
    """记笔记不动屏幕 —— 记了会让感知白等一次。"""
    out = _run_stamp([_call("save_note", 1)])
    assert out.get("last_action_at") is None
