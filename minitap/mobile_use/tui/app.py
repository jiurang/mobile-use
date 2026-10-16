"""The mobile-use terminal UI.

Layout::

    ┌ mobile-use ─────────────────────────────────┬─ device ─────────┐
    │ GOAL  给张三发一条消息                        │                  │
    │                                            │   ▀▀▀▀▀▀▀▀▀▀      │
    │  ▏planner                              1.2s │   ▀▀▀▀▀▀▀▀▀▀      │
    │  › 拆解目标：打开微信 → 找到张三 → 发消息      │                  │
    │  ├ tap(x=540, y=1180)                        │                  │
    │  └ ok · 0.8s                                 │                  │
    │                                            │  Redmi · android  │
    ├────────────────────────────────────────────┴──────────────────┤
    │ > next goal…                                          ctrl+t   │
    └────────────────────────────────────────────────────────────────┘

The whole UI is driven by `AgentEventBridge` events drained from a queue, so
the agent code and the graph stay untouched. `DemoSession` feeds the exact same
event stream for UI work without a device.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from rich.panel import Panel
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Input, RichLog, Static

from minitap.mobile_use.tui.events import AgentEventBridge, EventKind, TuiEvent
from minitap.mobile_use.tui.logs import LogCapture
from minitap.mobile_use.tui.session import DeviceInfo, SessionBase

KEY_HINTS = "^t stop   ^l clear   ^o logs   ^c quit   /help"
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Log levels shown without the toggle. Below these the SDK is very chatty
# ("Connecting UIAutomator2...", one line per tool, ...) and would drown the
# actual agent timeline.
LOUD_LOG_LEVELS = {"WARNING", "ERROR", "CRITICAL"}

# Palette. Kept as CSS variables so the whole app can be re-themed in one place.
ACCENT = "#4c8dff"
MUTED = "#7a8299"
OK = "#3ecf8e"
WARN = "#f5a623"
ERR = "#ff5c5c"
THOUGHT = "#b58cff"
BOLD_ERR = f"bold {ERR}"


class GoalBanner(Static):
    """Shows the goal currently being worked on."""

    def show_goal(self, goal: str) -> None:
        self.update(Text.assemble(("GOAL  ", MUTED), (goal, "bold")))

    def show_idle(self, hint: str) -> None:
        self.update(Text(hint, style=f"italic {MUTED}"))


def _fmt_count(value: int) -> str:
    """1234567 -> 1,234,567"""
    return f"{value:,}"


def _fmt_duration(seconds: float) -> str:
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m{total % 60:02d}s"
    return f"{total // 3600}h{(total % 3600) // 60:02d}m"


class StatsPanel(Static):
    """Session facts that used to live under the device screenshot.

    Device identity, token spend, elapsed time and progress counters. The
    screenshot preview was removed: at 36 columns the Chinese UI text rendered
    as unreadable grey blocks, while the numbers here stay exact.
    """

    def update_stats(
        self,
        *,
        device: DeviceInfo,
        tokens_in: int,
        tokens_out: int,
        usage_known: bool,
        steps: int,
        actions: int,
        elapsed: float,
        running: bool,
    ) -> None:
        def row(label: str, value: Text) -> Text:
            text = Text()
            text.append(f"{label:<10}", style=MUTED)
            text.append_text(value)
            return text

        lines: list[Text] = []

        # -- device ----------------------------------------------------
        lines.append(Text("DEVICE", style=f"bold {MUTED}"))
        lines.append(row("", Text(device.platform, style="#e8ecf5")))
        if device.device_id:
            lines.append(row("", Text(device.device_id, style=MUTED)))
        lines.append(row("", Text(device.screen, style=MUTED)))

        # -- tokens ----------------------------------------------------
        # Not every OpenAI-compatible endpoint reports usage (the MiniMax one
        # this project defaults to returns none), so say "n/a" rather than a
        # zero that would read as "this run was free".
        lines.append(Text())
        lines.append(Text("TOKENS", style=f"bold {MUTED}"))
        unknown = Text("n/a", style=f"italic {MUTED}")
        if usage_known:
            total = tokens_in + tokens_out
            lines.append(row("total", Text(_fmt_count(total), style="#e8ecf5")))
            lines.append(row("in", Text(_fmt_count(tokens_in), style=MUTED)))
            lines.append(row("out", Text(_fmt_count(tokens_out), style=MUTED)))
        else:
            lines.append(row("total", unknown.copy()))
            lines.append(Text("not reported by provider", style=f"dim {MUTED}"))

        # -- run -------------------------------------------------------
        lines.append(Text())
        lines.append(Text("RUN", style=f"bold {MUTED}"))
        lines.append(row("elapsed", Text(_fmt_duration(elapsed), style="#e8ecf5")))
        lines.append(row("steps", Text(str(steps), style="#e8ecf5")))
        lines.append(row("actions", Text(str(actions), style="#e8ecf5")))
        lines.append(
            row("state", Text("running" if running else "idle", style=OK if running else MUTED))
        )

        self.update(Text("\n").join(lines))


class StatusLine(Static):
    """Connection state, live "thinking" indicator, device, elapsed time.

    The agents call `ainvoke()`, not `astream()`, so there is no token stream
    to render. Without this the UI looks frozen between two graph nodes, so the
    running state is stated outright: a spinner, the word 思考中 / thinking,
    the node currently executing, and a live clock.
    """

    def set_state(
        self,
        *,
        device: str,
        state: str,
        node: str | None = None,
        elapsed: float = 0.0,
        runs: int = 0,
        frame: int = 0,
    ) -> None:
        minutes, seconds = divmod(int(elapsed), 60)
        bits: list[Text] = []

        if state == "running":
            spinner = SPINNER_FRAMES[frame % len(SPINNER_FRAMES)]
            bits.append(Text(f"{spinner} ", style=f"bold {ACCENT}"))
            bits.append(Text("thinking", style=f"bold {ACCENT}"))
            if node:
                bits.append(Text(f"  {node}", style="#e8ecf5"))
            else:
                bits.append(Text("  starting", style="#e8ecf5"))
        elif state == "offline":
            bits.append(Text("✗ ", style=ERR))
            bits.append(Text("offline", style=BOLD_ERR))
        elif state == "connecting":
            spinner = SPINNER_FRAMES[frame % len(SPINNER_FRAMES)]
            bits.append(Text(f"{spinner} ", style=WARN))
            bits.append(Text("connecting", style=f"bold {WARN}"))
        else:
            bits.append(Text("● ", style=OK))
            bits.append(Text("ready", style=MUTED))

        bits.append(Text("  ·  ", style=MUTED))
        bits.append(Text(device, style=MUTED))
        if runs:
            bits.append(Text(f"  ·  run #{runs}", style=MUTED))
        if state == "running":
            bits.append(Text(f"  ·  {minutes:02d}:{seconds:02d}", style=f"dim {MUTED}"))
        self.update(Text("").join(bits))


class MobileUseApp(App[None]):
    """Interactive REPL over one long-lived agent session."""

    TITLE = "mobile-use"
    SUB_TITLE = "interactive agent terminal"

    CSS = f"""
    Screen {{
        background: #0d1017;
    }}

    GoalBanner {{
        height: auto;
        padding: 1 2 0 2;
        color: {ACCENT};
    }}

    #body {{
        height: 1fr;
    }}

    #left {{
        width: 1fr;
        padding: 0 1 0 2;
    }}

    #right {{
        width: 38;
        padding: 1 2 0 0;
    }}

    StatsPanel {{
        height: auto;
        border: round #2b3240;
        background: #0a0d14;
        padding: 1 2;
    }}

    #timeline {{
        height: 1fr;
        border: round #2b3240;
        background: #0a0d14;
        padding: 0 1;
    }}

    #prompt {{
        height: 3;
        border: round #2b3240;
        background: #0a0d14;
    }}

    #prompt:focus {{
        border: round {ACCENT};
    }}

    #bottombar {{
        dock: bottom;
        height: auto;
        padding: 0 2;
    }}

    #statusbar {{
        height: 1;
    }}

    StatusLine {{
        width: 1fr;
        height: 1;
        padding: 0 1;
        color: {MUTED};
    }}

    #hints {{
        width: auto;
        height: 1;
        color: #4d556b;
        padding: 0 1;
    }}
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+d", "quit", "Quit", show=False),
        Binding("ctrl+t", "stop", "Stop task"),
        Binding("ctrl+l", "clear_timeline", "Clear"),
        Binding("ctrl+o", "toggle_logs", "SDK logs"),
    ]

    def __init__(
        self,
        session: SessionBase,
        bridge: AgentEventBridge,
    ) -> None:
        super().__init__()
        self.session = session
        # Shared with the session: one producer, one consumer.
        self.bridge = bridge
        self._goal_running = False
        self._goal_count = 0
        self._active_node: str | None = None
        self._goal_started_at = 0.0
        self._status_state = "connecting"
        self._show_logs = False
        self._log_capture: LogCapture | None = None
        # Progress counters shown in the side panel.
        self._tokens_in = 0
        self._tokens_out = 0
        self._usage_known = False
        self._steps = 0
        self._actions = 0

    # ------------------------------------------------------------------
    # composition
    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=False, icon="▌")
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield GoalBanner(id="goal")
                yield RichLog(id="timeline", wrap=True, auto_scroll=True, markup=False)
            with Vertical(id="right"):
                yield StatsPanel(id="stats")
        # One bottom container rather than independently docked widgets: two
        # `dock: bottom` siblings (and Textual's own Footer) all claimed the
        # same last row and the status line was invisible.
        with Vertical(id="bottombar"):
            yield Input(placeholder="describe your next goal…", id="prompt")
            with Horizontal(id="statusbar"):
                yield StatusLine(id="status")
                yield Static(KEY_HINTS, id="hints")

    def on_mount(self) -> None:
        self.query_one(GoalBanner).show_idle("no goal yet — type one below and press enter")
        self.query_one("#prompt", Input).focus()
        # Must come before the session starts: the SDK logs on the very first
        # line of init, and a full-screen app cannot survive stray writes.
        self._log_capture = LogCapture(self.bridge).__enter__()
        self._pump_events()
        self._pump_clock()
        self._start_session()

    # ------------------------------------------------------------------
    # workers
    # ------------------------------------------------------------------
    @work
    async def _start_session(self) -> None:
        try:
            await self.session.start()
        except Exception as exc:
            self.bridge.emit(EventKind.ERROR, "cannot start session", detail=str(exc), ok=False)
            self._status_state = "offline"
            return
        self._status_state = "ready"
        info = self.session.device_info
        self.bridge.emit(EventKind.STATUS, f"connected · {info.platform} {info.device_id}".strip())

    @work
    async def _pump_events(self) -> None:
        """Drain agent events into the timeline."""
        while True:
            events = self.bridge.drain()
            for event in events:
                self._apply(event)
            await asyncio.sleep(0.06)

    @work
    async def _pump_clock(self) -> None:
        """Single owner of both the status line and the side panel."""
        status = self.query_one(StatusLine)
        stats = self.query_one(StatsPanel)
        frame = 0
        while True:
            # One place owns the status line, so the spinner keeps moving and
            # the clock keeps ticking even between graph nodes.
            if self._goal_running:
                frame += 1
            state = "running" if self._goal_running else self._status_state
            elapsed = time.monotonic() - self._goal_started_at if self._goal_running else 0.0
            status.set_state(
                device=self.session.device_info.label,
                state=state,
                node=self._active_node,
                elapsed=elapsed,
                runs=self._goal_count,
                frame=frame,
            )
            stats.update_stats(
                device=self.session.device_info,
                tokens_in=self._tokens_in,
                tokens_out=self._tokens_out,
                usage_known=self._usage_known,
                steps=self._steps,
                actions=self._actions,
                elapsed=elapsed,
                running=self._goal_running,
            )
            await asyncio.sleep(0.15)

    @work
    async def _run_goal(self, goal: str) -> None:
        self._goal_count += 1
        self._goal_started_at = time.monotonic()
        self._goal_running = True
        # "starting" so the indicator reads sensibly before the first node
        # fires: there is no token stream to show, this *is* the feedback.
        self._active_node = "starting"
        self._status_state = "running"
        self.query_one(GoalBanner).show_goal(goal)
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = True
        prompt.placeholder = "running… press ctrl+t to stop the current task"

        try:
            output = await self.session.run_goal(goal)
        except asyncio.CancelledError:
            self.bridge.emit(EventKind.ERROR, "stopped by user", ok=False)
        except Exception as exc:
            self.bridge.emit(EventKind.ERROR, "goal failed", detail=str(exc), ok=False)
        else:
            self.bridge.emit(EventKind.RESULT, "goal complete", detail=_render_output(output))
        finally:
            self._goal_running = False
            self._active_node = None
            self._status_state = "offline" if self._status_state == "offline" else "ready"
            prompt.disabled = False
            prompt.placeholder = "describe your next goal…"
            prompt.focus()

    # ------------------------------------------------------------------
    # event -> UI
    # ------------------------------------------------------------------
    def _apply(self, event: TuiEvent) -> None:
        timeline = self.query_one("#timeline", RichLog)

        # SDK chatter is captured (so it cannot reach the terminal) but only
        # rendered when asked for, except for problems which always show.
        if event.kind is EventKind.LOG and not self._should_show_log(event):
            return

        # Counters feeding the side panel.
        if event.kind is EventKind.USAGE:
            tokens_in, tokens_out = event.payload or (0, 0)
            self._tokens_in, self._tokens_out = int(tokens_in), int(tokens_out)
            self._usage_known = True
        elif event.kind is EventKind.THOUGHT:
            self._steps += 1
        elif event.kind is EventKind.ACTION_END:
            self._actions += 1

        if event.kind is EventKind.USAGE:
            return

        renderable = _render_event(event)
        if renderable is not None:
            timeline.write(renderable)

        if event.kind is EventKind.NODE_START:
            self._active_node = event.title
        elif event.kind is EventKind.NODE_END:
            if self._active_node == event.title:
                self._active_node = None

    def _should_show_log(self, event: TuiEvent) -> bool:
        level = str(event.payload or "INFO").upper()
        if level in LOUD_LOG_LEVELS:
            return True
        return self._show_logs

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    @on(Input.Submitted, "#prompt")
    def _submit_goal(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""

        if text.startswith("/"):
            self._run_command(text)
            return

        if self._goal_running:
            self.query_one("#timeline", RichLog).write(
                Text("a goal is already running — press ctrl+t to stop it first", style=WARN)
            )
            return

        self._run_goal(text)

    def _run_command(self, text: str) -> None:
        command = text[1:].strip().lower()
        if command in {"quit", "exit", "q"}:
            self.exit()
        elif command == "stop":
            self.action_stop()
        elif command in {"clear", "cls"}:
            self.query_one("#timeline", RichLog).clear()
        elif command in {"logs", "log"}:
            self.action_toggle_logs()
        elif command == "reset":
            self.bridge.reset_tokens()
            self._tokens_in = self._tokens_out = 0
            self._usage_known = False
            self._steps = 0
            self._actions = 0
            self.query_one("#timeline", RichLog).write(Text("counters reset", style=MUTED))
        elif command == "help":
            self.query_one("#timeline", RichLog).write(
                Text(
                    "/quit · /stop · /clear · /logs · /reset · "
                    "ctrl+t stop · ctrl+l clear · ctrl+o logs · ctrl+c quit",
                    style=MUTED,
                )
            )
        else:
            self.query_one("#timeline", RichLog).write(Text(f"unknown command: {text}", style=ERR))

    def action_stop(self) -> None:
        if not self._goal_running:
            return
        self.session.stop()

    def action_clear_timeline(self) -> None:
        self.query_one("#timeline", RichLog).clear()

    def action_toggle_logs(self) -> None:
        self._show_logs = not self._show_logs
        self.query_one("#timeline", RichLog).write(
            Text(
                "sdk logs ON" if self._show_logs else "sdk logs OFF (warnings still shown)",
                style=ACCENT if self._show_logs else MUTED,
            )
        )

    async def on_unmount(self) -> None:
        if self._log_capture is not None:
            # Restore the real stdout before anything else can print.
            self._log_capture.__exit__(None, None, None)
            self._log_capture = None
        await self.session.close()


def _fmt_ms(duration_ms: int | None) -> str:
    return f"{duration_ms / 1000:.1f}s" if duration_ms is not None else ""


def _render_output(output: Any) -> str:
    if output is None:
        return "(no structured output)"
    if isinstance(output, str):
        return output
    return str(output)


def _render_event(event: TuiEvent) -> Any:
    """Turn one agent event into what the timeline shows.

    Returns ``None`` for events that only mutate chrome (e.g. NODE_START, whose
    line is written when the node finishes so it can carry a duration).
    """
    if event.kind is EventKind.NODE_START:
        return None

    if event.kind is EventKind.LOG:
        level = str(event.payload or "INFO").upper()
        marker, style = {
            "DEBUG": ("·", f"dim {MUTED}"),
            "INFO": ("·", f"dim {MUTED}"),
            "SUCCESS": ("·", f"dim {MUTED}"),
            "WARNING": ("⚠", WARN),
            "ERROR": ("❌", ERR),
            "CRITICAL": ("❌", BOLD_ERR),
        }.get(level, ("·", f"dim {MUTED}"))
        text = Text()
        text.append(f"  {marker} ", style=style)
        text.append(event.title, style=style)
        return text

    if event.kind is EventKind.NODE_END:
        text = Text()
        text.append("▏ ", style=MUTED)
        text.append(f"{event.title:<18}", style=ACCENT)
        duration = _fmt_ms(event.duration_ms)
        if duration:
            text.append(f"{duration:>7}", style=f"dim {MUTED}")
        return text

    if event.kind is EventKind.THOUGHT:
        text = Text()
        text.append("  › ", style=THOUGHT)
        text.append(event.title, style=THOUGHT)
        return text

    if event.kind is EventKind.ACTION_START:
        text = Text()
        text.append("  ├ ", style=MUTED)
        text.append(f"{event.title}", style="bold #ffffff")
        if event.detail:
            text.append(f"  ({event.detail})", style=MUTED)
        return text

    if event.kind is EventKind.ACTION_END:
        text = Text()
        text.append("  └ ", style=MUTED)
        status = "ok" if event.ok else "failed"
        text.append(status, style=OK if event.ok else ERR)
        duration = _fmt_ms(event.duration_ms)
        if duration:
            text.append(f" · {duration}", style=f"dim {MUTED}")
        # Many tools literally return "ok"; don't print it twice.
        if event.detail and event.detail.strip().lower() != status:
            text.append(f" · {event.detail}", style=f"dim {MUTED}")
        return text

    if event.kind is EventKind.RESULT:
        return Panel(
            Text(event.detail, style="#e6e9f0"),
            title="result",
            border_style=OK,
            padding=(0, 1),
        )

    if event.kind is EventKind.ERROR:
        text = Text()
        text.append("✗ ", style=ERR)
        text.append(event.title, style=BOLD_ERR)
        if event.detail:
            text.append(f"  {event.detail}", style=ERR)
        return text

    if event.kind is EventKind.STATUS:
        return Text(f"· {event.title}", style=MUTED)

    return None
