"""Keep the agent's background logging inside the TUI.

The SDK logs through stdlib ``logging``: every module's logger is built by
``MobileUseLogger.__init__``, which attaches a ``StreamHandler(sys.stdout)``
plus occasional bare ``print()`` calls. In a full-screen Textual app that
output lands on top of the rendered screen and shreds it.

Rather than silencing the logs -- they are often the only explanation for a
failure -- this routes them into the same event stream the UI already renders,
so warnings and errors stay visible while ordinary progress chatter is hidden
behind a toggle.

Two levels of interception, because a logger's handler captures the stream
object *at construction time*:

* ``sys.stdout`` / ``sys.stderr`` are replaced, which covers bare ``print()``
  and anything else that writes to the standard streams.
* ``MobileUseLogger.__init__`` is patched. Patching only the loggers that
  already exist is not enough, and letting new loggers fall through to the
  stream is actively wrong: the sink cannot know the record's level, so a
  WARNING would be re-labelled INFO and then hidden as routine chatter.

Textual itself is unaffected: its Windows driver writes to ``sys.__stdout__``,
not ``sys.stdout``.
"""

from __future__ import annotations

import logging
import re
import sys
from types import TracebackType
from typing import TextIO

from minitap.mobile_use.tui.events import AgentEventBridge, EventKind

# Cap for a single unterminated line, so a stray write without a newline cannot
# grow the buffer without bound.
_MAX_BUFFER = 8192

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(text: str) -> str:
    """Drop SGR colour codes.

    Several SDK call sites embed `Fore.*` / `Style.RESET_ALL` in the message
    itself. Those are already-rendered ANSI bytes and Rich would print them as
    literal garbage rather than interpreting them.
    """
    return _ANSI.sub("", text)


class _StreamSink:
    """Minimal TextIO stand-in that turns writes into log events.

    Used only for output that carries no level information (bare ``print()``).
    Records routed through ``TuiLogHandler`` keep their real level.
    """

    def __init__(self, bridge: AgentEventBridge) -> None:
        self._bridge = bridge
        self._buffer = ""

    def write(self, data: str) -> int:
        self._buffer += data
        if len(self._buffer) > _MAX_BUFFER:
            self._flush_buffer()
            return len(data)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line)
        return len(data)

    def _flush_buffer(self) -> None:
        if self._buffer:
            self._emit(self._buffer)
            self._buffer = ""

    def _emit(self, line: str) -> None:
        text = strip_ansi(line).strip()
        if text:
            self._bridge.emit(EventKind.LOG, text, payload="INFO")

    def flush(self) -> None:
        self._flush_buffer()

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        raise OSError("log sink has no file descriptor")

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False


class TuiLogHandler(logging.Handler):
    """logging handler that forwards records into the TUI event stream."""

    def __init__(self, bridge: AgentEventBridge) -> None:
        # DEBUG so the toggle can reveal full detail; the UI decides what to
        # display, not the handler.
        super().__init__(level=logging.DEBUG)
        self._bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = strip_ansi(record.getMessage())
        except Exception:  # pragma: no cover - logging must never raise here
            return
        if not message.strip():
            return
        self._bridge.emit(EventKind.LOG, message.strip(), payload=record.levelname)


class LogCapture:
    """Route SDK logging into the TUI. Use as a context manager.

    ```python
    with LogCapture(bridge):
        ...
    ```
    """

    def __init__(self, bridge: AgentEventBridge) -> None:
        self._bridge = bridge
        self._handler = TuiLogHandler(bridge)
        self._stdout: TextIO | None = None
        self._stderr: TextIO | None = None
        self._detached: list[tuple[logging.Logger, logging.Handler]] = []
        self._original_init = None

    # -- internals -----------------------------------------------------
    def _redirect(self, standard_logging_logger: logging.Logger) -> None:
        for existing in list(standard_logging_logger.handlers):
            if isinstance(existing, logging.StreamHandler) and not isinstance(
                existing, logging.FileHandler
            ):
                standard_logging_logger.removeHandler(existing)
                self._detached.append((standard_logging_logger, existing))
        standard_logging_logger.addHandler(self._handler)

    # -- lifecycle -----------------------------------------------------
    def __enter__(self) -> LogCapture:
        from minitap.mobile_use.utils.logger import MobileUseLogger, _loggers

        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = _StreamSink(self._bridge)  # type: ignore[assignment]
        sys.stderr = _StreamSink(self._bridge)  # type: ignore[assignment]

        for mobile_logger in list(_loggers.values()):
            self._redirect(mobile_logger.logger)

        capture = self
        self._original_init = MobileUseLogger.__init__
        original_init = self._original_init

        def _new_init(self_, *args, **kwargs) -> None:
            original_init(self_, *args, **kwargs)
            capture._redirect(self_.logger)

        # Patch the class rather than the module-level `get_logger`: modules do
        # `from ... import get_logger`, so rebinding the module attribute would
        # not affect them.
        MobileUseLogger.__init__ = _new_init  # type: ignore[method-assign]

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        from minitap.mobile_use.utils.logger import MobileUseLogger

        if self._original_init is not None:
            MobileUseLogger.__init__ = self._original_init  # type: ignore[method-assign]
            self._original_init = None

        for standard_logger, handler in self._detached:
            if self._handler in standard_logger.handlers:
                standard_logger.removeHandler(self._handler)
            standard_logger.addHandler(handler)
        self._detached.clear()

        if self._stdout is not None:
            sys.stdout = self._stdout
        if self._stderr is not None:
            sys.stderr = self._stderr
        self._stdout = self._stderr = None
