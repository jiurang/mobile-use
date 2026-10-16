"""Tests for the mobile-use terminal UI.

Two layers:
- unit tests for the event bridge and the half-block renderer, which are the
  parts with real logic;
- one end-to-end test that boots the actual Textual app headlessly, submits a
  goal, and asserts the timeline and device panel update. The UI is only
  trustworthy if the whole app runs, not just its helpers.
"""

from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from textual.widgets import Input

from minitap.mobile_use.tui.app import MobileUseApp, StatusLine
from minitap.mobile_use.tui.events import (
    AgentEventBridge,
    EventKind,
    _extract_usage,
    extract_thought,
    format_tool_input,
)
from minitap.mobile_use.tui.session import DemoSession, DeviceType, TuiSession


def screen_text(app: MobileUseApp) -> str:
    """Flatten the rendered screen to plain text for assertions."""
    strips = app.screen._compositor.render_strips()
    return "\n".join(strip.text for strip in strips)


# ----------------------------------------------------------------------
# event bridge
# ----------------------------------------------------------------------


def test_drain_returns_events_in_order():
    bridge = AgentEventBridge()
    bridge.emit(EventKind.NODE_START, "planner")
    bridge.emit(EventKind.THOUGHT, "thinking")

    events = bridge.drain()

    assert [e.title for e in events] == ["planner", "thinking"]
    assert bridge.drain() == []


def test_unknown_graph_nodes_are_ignored():
    """LangGraph fires chain events for the graph itself and internal sub-chains."""
    bridge = AgentEventBridge()
    bridge.on_chain_start({}, {}, run_id="1", metadata={"langgraph_node": "LangGraph"})
    bridge.on_chain_start({}, {}, run_id="2", metadata={})
    bridge.on_chain_start({}, {}, run_id="3", metadata=None)

    assert bridge.drain() == []


def test_node_lifecycle_emits_start_then_end_with_duration():
    bridge = AgentEventBridge()
    bridge.on_chain_start({}, {}, run_id="7", metadata={"langgraph_node": "cortex"})
    bridge.on_chain_end({"agents_thoughts": ["decide to tap"]}, run_id="7")

    events = bridge.drain()
    kinds = [e.kind for e in events]

    assert kinds == [EventKind.NODE_START, EventKind.NODE_END, EventKind.THOUGHT]
    assert events[1].duration_ms is not None
    assert events[2].title == "decide to tap"


def test_repeated_thought_is_not_emitted_twice():
    """The state update carries the whole thought list, so the tail repeats."""
    bridge = AgentEventBridge()
    for run_id in ("1", "2"):
        bridge.on_chain_start({}, {}, run_id=run_id, metadata={"langgraph_node": "executor"})
        bridge.on_chain_end({"agents_thoughts": ["same"]}, run_id=run_id)

    kinds = [e.kind for e in bridge.drain() if e.kind is EventKind.THOUGHT]
    assert kinds == [EventKind.THOUGHT]


def test_tool_events_name_the_tool_and_time_it():
    bridge = AgentEventBridge()
    bridge.on_tool_start({"name": "tap"}, "", run_id="9", inputs={"x": 1, "y": 2})
    bridge.on_tool_end("tapped", run_id="9")

    start, end = bridge.drain()

    assert (start.kind, start.title, start.detail) == (EventKind.ACTION_START, "tap", "x=1, y=2")
    assert end.kind is EventKind.ACTION_END
    assert end.ok is True
    assert end.duration_ms is not None


def test_tool_error_is_reported_as_a_failed_action():
    bridge = AgentEventBridge()
    bridge.on_tool_start({"name": "swipe"}, "", run_id="4", inputs={})
    bridge.on_tool_error(RuntimeError("device offline"), run_id="4")

    _, end = bridge.drain()

    assert end.ok is False
    assert "device offline" in end.detail


def test_extract_thought_ignores_junk_shapes():
    assert extract_thought(None) is None
    assert extract_thought({"agents_thoughts": []}) is None
    assert extract_thought({"agents_thoughts": ["  ", ""]}) is None
    assert extract_thought({"agents_thoughts": ["ok"]}) == "ok"


def test_format_tool_input_caps_the_line():
    detail = format_tool_input({"a": "x" * 200, "b": 2, "c": 3, "d": 4})

    assert detail.count("=") == 3
    assert "…" in detail


# ----------------------------------------------------------------------
# session wiring against the real SDK (no device needed)
# ----------------------------------------------------------------------


def test_session_builds_its_agent_config():
    """Guards the keyword names handed to the SDK config builders.

    Regression: the WDA overrides were once passed as
    ``WdaClientConfig.with_overrides(**{"url": ...})``. The SDK spells that
    parameter ``wda_url``, so the TUI died at startup with
    `TypeError: with_overrides() got an unexpected keyword argument 'url'` --
    and because a dict expansion erases signature checking, neither pyright
    nor any test caught it. Builds the config for real; needs no device.
    """
    import minitap.mobile_use.config as config_module

    session = TuiSession(
        bridge=AgentEventBridge(),
        wda_url="http://localhost:8100",
        wda_timeout=42.0,
        wda_auto_start_iproxy=False,
        wda_auto_start_wda=True,
        wda_project_path="/tmp/WebDriverAgent.xcodeproj",
        wda_startup_timeout=120.0,
        idb_host="127.0.0.1",
        idb_port=10882,
    )
    # Only the LLM provider selection is environment dependent; the defaults
    # file that ships with the repo is enough to build a config here.
    monkey = pytest.MonkeyPatch()
    monkey.setattr(config_module, "initialize_llm_config", config_module.parse_llm_config)
    try:
        config = session._build_config().build()
    finally:
        monkey.undo()

    ios = config.ios_client_config
    assert ios is not None
    wda = ios.wda
    assert wda.wda_url == "http://localhost:8100"
    assert wda.timeout == 42.0
    assert wda.auto_start_wda is True
    assert wda.wda_project_path == "/tmp/WebDriverAgent.xcodeproj"
    assert wda.wda_startup_timeout == 120.0
    # The callback hook the whole TUI depends on is actually wired in, and it
    # is our bridge rather than some other handler.
    callbacks = config.graph_config_callbacks
    assert isinstance(callbacks, list)
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], AgentEventBridge)


def test_cloud_device_type_requires_a_platform():
    import minitap.mobile_use.config as config_module

    session = TuiSession(bridge=AgentEventBridge(), device_type=DeviceType.CLOUD)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(config_module, "initialize_llm_config", config_module.parse_llm_config)
    try:
        with pytest.raises(Exception, match="cloud-platform"):
            session._build_config()
    finally:
        monkey.undo()


# ----------------------------------------------------------------------
# the LangGraph callback contract the bridge depends on
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bridge_reads_events_from_a_real_langgraph_run():
    """Guards the load-bearing assumption of the whole design.

    The TUI is wired in through the pre-existing
    ``AgentConfig.graph_config_callbacks`` field, so it never touches the agents
    or the graph. That only works while LangGraph keeps reporting
    ``metadata["langgraph_node"]`` and passing each node's state update to
    ``on_chain_end`` -- with ``agents_thoughts`` on it.
    """

    class MiniState(TypedDict, total=False):
        agents_thoughts: list[str]

    async def cortex(state: MiniState) -> MiniState:
        return {"agents_thoughts": ["first", "second"]}

    async def executor(state: MiniState) -> MiniState:
        return {}

    builder = StateGraph(MiniState)
    builder.add_node("cortex", cortex)
    builder.add_node("executor", executor)
    builder.add_edge(START, "cortex")
    builder.add_edge("cortex", "executor")
    builder.add_edge("executor", END)
    graph = builder.compile()

    bridge = AgentEventBridge()
    async for _ in graph.astream(
        {"agents_thoughts": []},
        config={"callbacks": [bridge]},
        stream_mode=["messages", "custom", "updates", "values"],
    ):
        pass

    events = bridge.drain()
    kinds = [e.kind for e in events]
    nodes = [e.title for e in events if e.kind is EventKind.NODE_END]
    thoughts = [e.title for e in events if e.kind is EventKind.THOUGHT]

    assert nodes == ["cortex", "executor"]
    # Only the newest thought, not the whole accumulated list.
    assert thoughts == ["second"]
    assert EventKind.NODE_START in kinds
    assert all(e.duration_ms is not None for e in events if e.kind is EventKind.NODE_END)


# ----------------------------------------------------------------------
# token accounting
# ----------------------------------------------------------------------


class _Response:
    """Stand-in for an LLM response carrying a provider-specific usage shape."""

    def __init__(self, **attrs) -> None:
        self.usage_metadata = None
        self.llm_output = None
        self.response_metadata = None
        for key, value in attrs.items():
            setattr(self, key, value)


def test_tokens_accumulate_across_model_calls():
    bridge = AgentEventBridge()
    bridge.on_llm_end(
        _Response(usage_metadata={"input_tokens": 100, "output_tokens": 20}), run_id="1"
    )
    bridge.on_llm_end(
        _Response(usage_metadata={"input_tokens": 50, "output_tokens": 10}), run_id="2"
    )

    assert bridge.total_tokens == (150, 30)

    usage = [e for e in bridge.drain() if e.kind is EventKind.USAGE]
    assert len(usage) == 2
    # Each event carries the running total, not a per-call delta.
    assert usage[-1].payload == (150, 30)


def test_token_extraction_handles_every_shape_langchain_produces():
    # Standardised field.
    assert _extract_usage(_Response(usage_metadata={"input_tokens": 7, "output_tokens": 3})) == (
        7,
        3,
    )
    # OpenAI-style llm_output.
    assert _extract_usage(
        _Response(llm_output={"token_usage": {"prompt_tokens": 11, "completion_tokens": 5}})
    ) == (11, 5)
    # Google-style response_metadata, camelCase.
    assert _extract_usage(
        _Response(response_metadata={"usage": {"promptTokenCount": 13, "candidatesTokenCount": 6}})
    ) == (13, 6)
    # Nothing reported must not raise.
    assert _extract_usage(_Response()) is None
    assert _extract_usage(None) is None


def test_broken_usage_metadata_does_not_break_the_run():
    bridge = AgentEventBridge()
    bridge.on_llm_end(
        _Response(usage_metadata={"input_tokens": "nonsense", "output_tokens": None}), run_id="1"
    )

    assert bridge.total_tokens == (0, 0)


# ----------------------------------------------------------------------
# background logging must not corrupt the screen
# ----------------------------------------------------------------------


class _ChattySession(DemoSession):
    """A session that behaves like the SDK: logger calls plus bare prints."""

    async def run_goal(self, goal: str) -> str:
        from minitap.mobile_use.utils.logger import get_logger

        log = get_logger("minitap.mobile_use.test_tui_chatter")
        log.info("QUIET-INFO-LINE from the sdk")
        log.warning("LOUD-WARNING-LINE from the sdk")
        print("BARE-PRINT-LINE from the sdk")
        return "ok"


@pytest.mark.asyncio
async def test_sdk_logs_are_captured_instead_of_hitting_the_terminal(capsys):
    """Regression: a full-screen app is destroyed by stray stdout writes.

    The SDK logs through stdlib `logging` handlers that captured `sys.stdout`
    at construction time, plus bare `print()` calls. Both used to land on top
    of the rendered TUI. They must be routed into the event stream instead.
    """
    bridge = AgentEventBridge()
    session = _ChattySession(bridge=bridge, script=[])
    app = MobileUseApp(session, bridge)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one("#prompt", Input).value = "make some noise"
        await pilot.press("enter")
        for _ in range(60):
            await pilot.pause(0.05)
            if not app._goal_running and app._goal_count:
                break
        await pilot.pause(0.1)
        rendered = screen_text(app)

    leaked = capsys.readouterr()
    for marker in ("QUIET-INFO-LINE", "LOUD-WARNING-LINE", "BARE-PRINT-LINE"):
        assert marker not in leaked.out
        assert marker not in leaked.err

    # Nothing reached the terminal, and the app's own pump consumed the events
    # (so they are asserted via what got rendered, not via bridge.drain()).
    # Warnings always surface; INFO chatter -- including bare print() -- stays
    # hidden until the toggle is on.
    assert "LOUD-WARNING-LINE" in rendered
    assert "QUIET-INFO-LINE" not in rendered
    assert "BARE-PRINT-LINE" not in rendered


@pytest.mark.asyncio
async def test_log_toggle_reveals_the_routine_chatter():
    bridge = AgentEventBridge()
    session = _ChattySession(bridge=bridge, script=[])
    app = MobileUseApp(session, bridge)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one("#prompt", Input).value = "make some noise"
        await pilot.press("enter")
        for _ in range(60):
            await pilot.pause(0.05)
            if not app._goal_running and app._goal_count:
                break

        assert app._show_logs is False
        await pilot.press("ctrl+o")
        await pilot.pause()
        assert app._show_logs is True

        app.query_one("#prompt", Input).value = "make some noise again"
        await pilot.press("enter")
        for _ in range(60):
            await pilot.pause(0.05)
            if not app._goal_running and app._goal_count == 2:
                break
        await pilot.pause(0.1)

        revealed = screen_text(app)
        # Both the logger's INFO line and the bare print() come back.
        assert "QUIET-INFO-LINE" in revealed
        assert "BARE-PRINT-LINE" in revealed


@pytest.mark.asyncio
async def test_log_capture_restores_stdout_on_unmount():
    """A second run in the same process must not stay muted."""
    import sys

    before = sys.stdout
    bridge = AgentEventBridge()
    session = DemoSession(bridge=bridge, script=[])
    app = MobileUseApp(session, bridge)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()

    assert sys.stdout is before


# ----------------------------------------------------------------------
# whole app, headless
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_app_runs_a_goal_and_updates_the_stats_panel():
    bridge = AgentEventBridge()
    session = DemoSession(
        bridge=bridge,
        script=[
            {"kind": "node", "name": "planner", "think": 0.01},
            {"kind": "thought", "text": "拆解目标"},
            {"kind": "action", "name": "tap", "detail": "x=1, y=2", "take": 0.01},
            {"kind": "tokens", "totals": (1234, 567)},
            {"kind": "result", "output": "all good"},
        ],
    )
    app = MobileUseApp(session, bridge)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # Pilot.press only takes key names, so the goal text is set directly.
        app.query_one("#prompt", Input).value = "send a message to Zhang San"
        await pilot.press("enter")

        for _ in range(60):
            await pilot.pause(0.05)
            if not app._goal_running and app._goal_count:
                break
        await pilot.pause(0.2)

        assert app._goal_count == 1
        assert app._goal_running is False

        rendered = screen_text(app)
        # Action, reasoning and node all reached the timeline.
        assert "tap" in rendered
        assert "拆解目标" in rendered
        assert "planner" in rendered
        assert "all good" in rendered

        # The side panel reports device, tokens and progress.
        assert "android" in rendered
        assert "demo-0000" in rendered
        assert "1220×2712" in rendered
        assert "1,801" in rendered  # 1234 + 567
        assert "1,234" in rendered
        assert "567" in rendered
        assert "steps" in rendered
        assert "actions" in rendered


@pytest.mark.asyncio
async def test_app_accepts_a_second_goal_on_the_same_session():
    """The whole point of the REPL: one session, many goals."""
    bridge = AgentEventBridge()
    session = DemoSession(bridge=bridge, script=[{"kind": "result", "output": "done"}])
    app = MobileUseApp(session, bridge)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        for goal in ("first goal", "second goal"):
            app.query_one("#prompt", Input).value = goal
            await pilot.press("enter")
            for _ in range(60):
                await pilot.pause(0.05)
                if not app._goal_running and app._goal_count:
                    break
            await pilot.pause(0.05)

        assert app._goal_count == 2


@pytest.mark.asyncio
async def test_status_line_says_it_is_thinking_while_a_goal_runs():
    """The agents stream no tokens, so the running state must be stated.

    Regression guard for the requirement that the TUI tells the user the agent
    is thinking rather than looking frozen between two graph nodes.
    """
    bridge = AgentEventBridge()
    # Long enough that the 0.15s status clock is guaranteed to tick at least
    # once while the goal is in flight.
    session = DemoSession(
        bridge=bridge,
        script=[{"kind": "result", "output": "done", "sleep": 0.6}],
    )
    app = MobileUseApp(session, bridge)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one("#prompt", Input).value = "think about it"
        await pilot.press("enter")

        # Sample while the goal is in flight.
        seen_thinking = False
        for _ in range(80):
            await pilot.pause(0.05)
            status = str(app.query_one(StatusLine).content)
            if "thinking" in status:
                seen_thinking = True
                break
            if not app._goal_running and app._goal_count:
                break

        assert seen_thinking, "status line never showed a thinking state"

        # And it settles back to ready once the goal is done. The status clock
        # ticks every 0.15s, so wait for that tick rather than the goal.
        for _ in range(80):
            await pilot.pause(0.05)
            if "ready" in str(app.query_one(StatusLine).content):
                break
        assert not app._goal_running
        assert "ready" in str(app.query_one(StatusLine).content)


@pytest.mark.asyncio
async def test_slash_commands_do_not_start_a_goal():
    bridge = AgentEventBridge()
    session = DemoSession(bridge=bridge, script=[])
    app = MobileUseApp(session, bridge)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one("#prompt", Input).value = "/clear"
        await pilot.press("enter")
        await pilot.pause()

        assert app._goal_count == 0
