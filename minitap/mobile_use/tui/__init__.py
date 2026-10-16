"""Interactive terminal UI for mobile-use.

Entry points:
- `mobile-use-tui` (see `tui/cli.py`) — the real thing.
- `MobileUseApp` — the Textual application, usable in tests via
  `app.run_test()`.

The UI is a pure consumer of `AgentEventBridge` events, so it can be driven by
a real agent, a demo script, or a test fixture without changes.
"""

from __future__ import annotations

from minitap.mobile_use.tui.app import MobileUseApp
from minitap.mobile_use.tui.events import AgentEventBridge, EventKind, TuiEvent
from minitap.mobile_use.tui.session import DemoSession, DeviceInfo, DeviceType, TuiSession

__all__ = [
    "AgentEventBridge",
    "DemoSession",
    "DeviceInfo",
    "DeviceType",
    "EventKind",
    "MobileUseApp",
    "TuiEvent",
    "TuiSession",
]
