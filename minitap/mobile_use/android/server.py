"""HTTP control surface for mobile-use running on an Android phone.

Why a server instead of the Textual TUI
--------------------------------------
The Textual TUI needs a real TTY and the foreground. This module exposes the
same capabilities over HTTP so the agent can live in the background while the
UI is just a browser tab -- on the phone itself, or on a laptop on the same
Wi-Fi. `AgentEventBridge` is reused verbatim: it is a plain queue, so the
streaming logic is identical to the TUI's.

Endpoints
---------
GET  /              terminal-style single-page UI
GET  /api/state     model / device / run state
POST /api/goal      submit a goal, returns immediately
POST /api/stop      stop the running task
GET  /api/events    Server-Sent Events stream of agent activity
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from minitap.mobile_use.tui.events import AgentEventBridge, EventKind, TuiEvent

LOGGER = logging.getLogger(__name__)
UI_FILE = Path(__file__).with_name("index.html")


@dataclass
class Run:
    """One submitted goal and how it went."""

    goal: str
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    result: str | None = None
    error: str | None = None

    @property
    def state(self) -> str:
        if self.error:
            return "error"
        if self.result is not None:
            return "done"
        if self.ended_at is not None:
            return "stopped"
        return "running"


def _resolve_device_id(host: str | None, port: int | None) -> str | None:
    """Pick the device to drive.

    The SDK's own auto-discovery shells out to the ``adb`` binary, which does
    not exist in a useful form on the phone (Termux's build cannot link
    against the device libc++). We already hold an adb *client*, so ask the
    server directly. ``MOBILE_USE_DEVICE_ID`` pins the choice; otherwise the
    first device the server reports wins.
    """
    import adbutils

    pinned = os.environ.get("MOBILE_USE_DEVICE_ID") or None
    try:
        client = adbutils.AdbClient(host=host, port=port or 5037) if host else adbutils.adb
        serials = [d.serial for d in client.device_list()]
    except Exception as exc:
        LOGGER.warning("could not list devices via adb server: %s", exc)
        return pinned

    if pinned:
        return pinned if pinned in serials else None
    return serials[0] if serials else None


class AndroidApp:
    """Owns the agent lifetime and the run history."""

    def __init__(self) -> None:
        self._agent: Any = None
        self._bridge = AgentEventBridge()
        self._runs: list[Run] = []
        self._active: Run | None = None
        self._task: asyncio.Task[None] | None = None
        self._device: dict[str, Any] = {}
        self._models: list[dict[str, str]] = []
        self._init_error: str | None = None

    # ------------------------------------------------------------------
    # model / device description for the header
    # ------------------------------------------------------------------
    def describe_models(self) -> list[dict[str, str]]:
        if self._models:
            return self._models
        from minitap.mobile_use.config import initialize_llm_config

        def flatten(node: Any, prefix: str = "") -> list[dict[str, str]]:
            if not isinstance(node, dict):
                return []
            if "provider" in node:
                provider = node["provider"]
                provider = getattr(provider, "value", provider)
                return [
                    {
                        "agent": prefix,
                        "provider": str(provider),
                        "model": str(node.get("model", "?")),
                    }
                ]
            rows: list[dict[str, str]] = []
            for key, value in node.items():
                rows += flatten(value, f"{prefix}.{key}" if prefix else str(key))
            return rows

        rows = flatten(initialize_llm_config().model_dump())
        rows.sort(key=lambda r: r["agent"])
        self._models = rows
        return rows

    # ------------------------------------------------------------------
    # agent lifecycle
    # ------------------------------------------------------------------
    async def ensure_agent(self) -> Any:
        if self._agent is not None:
            return self._agent
        if self._init_error:
            raise RuntimeError(self._init_error)

        from minitap.mobile_use.config import initialize_llm_config, settings
        from minitap.mobile_use.context import DevicePlatform
        from minitap.mobile_use.sdk import Agent
        from minitap.mobile_use.sdk.builders import Builders
        from minitap.mobile_use.sdk.types.task import AgentProfile

        profile = AgentProfile(name="android", llm_config=initialize_llm_config())
        config = Builders.AgentConfig.with_default_profile(profile=profile)
        # Same hook the TUI uses: zero changes to agents or graph.
        config.with_graph_config_callbacks([self._bridge])

        host, port = settings.ADB_HOST, settings.ADB_PORT
        if host:
            config.with_adb_server(host=host, port=port or 5037)

        device_id = _resolve_device_id(host, port)
        if device_id:
            config.for_device(platform=DevicePlatform.ANDROID, device_id=device_id)
            self._device["id"] = device_id

        try:
            agent = Agent(config=config.build())
            await agent.init(retry_count=3, retry_wait_seconds=2)
        except Exception as exc:  # keep the UI usable and show the reason
            self._init_error = f"{type(exc).__name__}: {exc}"
            raise

        self._agent = agent
        ctx = getattr(agent, "_device_context", None)
        if ctx is not None:
            self._device = {
                "id": ctx.device_id,
                "platform": str(ctx.mobile_platform.value),
                "width": ctx.device_width,
                "height": ctx.device_height,
            }
        return agent

    async def start(self) -> None:
        try:
            await self.ensure_agent()
        except Exception as exc:
            self._bridge.emit(EventKind.ERROR, "agent init failed", detail=str(exc)[:300], ok=False)

    async def close(self) -> None:
        self.stop()
        if self._agent is not None:
            agent, self._agent = self._agent, None
            with contextlib.suppress(Exception):
                await agent.clean()

    # ------------------------------------------------------------------
    # running goals
    # ------------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    def submit(self, goal: str) -> Run:
        if self.busy:
            raise RuntimeError("a goal is already running")
        run = Run(goal=goal)
        self._runs.append(run)
        self._active = run
        self._bridge.reset_tokens()
        self._bridge.emit(EventKind.STATUS, "goal accepted", detail=goal)
        self._task = asyncio.create_task(self._drive(run))
        return run

    async def _drive(self, run: Run) -> None:
        try:
            agent = await self.ensure_agent()
            run.result = await agent.run_task(goal=run.goal)
            self._bridge.emit(EventKind.RESULT, "goal finished", detail=str(run.result)[:300])
        except asyncio.CancelledError:
            run.ended_at = time.time()
            self._bridge.emit(EventKind.ERROR, "stopped", detail=run.goal, ok=False)
            raise
        except Exception as exc:
            run.error = f"{type(exc).__name__}: {exc}"
            self._bridge.emit(EventKind.ERROR, "goal failed", detail=run.error[:300], ok=False)
        finally:
            run.ended_at = time.time()
            if self._active is run:
                self._active = None

    def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            self._task = None
        if self._agent is not None:
            with contextlib.suppress(Exception):
                self._agent.stop_current_task()

    # ------------------------------------------------------------------
    # view model
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        tokens_in, tokens_out = self._bridge.total_tokens
        return {
            "models": self._models or self.describe_models(),
            "device": self._device,
            "running": self.busy,
            "active_goal": self._active.goal if self._active else None,
            "tokens": {"in": tokens_in, "out": tokens_out, "reported": self._bridge.usage_reported},
            "runs": [
                {
                    "goal": r.goal,
                    "state": r.state,
                    "result": r.result,
                    "error": r.error,
                    "seconds": round((r.ended_at or time.time()) - r.started_at, 1),
                }
                for r in self._runs[-20:]
            ],
            "error": self._init_error,
        }

    def drain_events(self) -> list[TuiEvent]:
        return self._bridge.drain()


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app() -> FastAPI:
    state = AndroidApp()
    api = FastAPI(title="mobile-use on Android", docs_url=None, redoc_url=None)

    @api.on_event("startup")
    async def _startup() -> None:
        asyncio.create_task(state.start())

    @api.on_event("shutdown")
    async def _shutdown() -> None:
        await state.close()

    @api.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return UI_FILE.read_text(encoding="utf-8")

    @api.get("/api/state")
    async def get_state() -> dict[str, Any]:
        return state.snapshot()

    @api.post("/api/goal")
    async def post_goal(payload: dict[str, Any]) -> dict[str, Any]:
        goal = (payload or {}).get("goal", "").strip()
        if not goal:
            return {"ok": False, "error": "goal is empty"}
        if state.busy:
            return {"ok": False, "error": "a goal is already running"}
        try:
            state.submit(goal)
        except RuntimeError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True}

    @api.post("/api/stop")
    async def post_stop() -> dict[str, Any]:
        state.stop()
        return {"ok": True}

    @api.get("/api/events")
    async def events() -> StreamingResponse:
        async def stream():
            last_ping = 0.0
            while True:
                batch = state.drain_events()
                if batch:
                    for event in batch:
                        payload: dict[str, Any] = {
                            "type": str(event.kind),
                            "title": event.title,
                            "detail": event.detail,
                            "ok": event.ok,
                            "ms": event.duration_ms,
                        }
                        if event.kind == EventKind.USAGE and event.payload:
                            payload["in"], payload["out"] = event.payload
                        yield _sse(payload)
                now = time.monotonic()
                if now - last_ping > 2.0:
                    last_ping = now
                    yield _sse({"type": "state", **state.snapshot()})
                await asyncio.sleep(0.25)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return api


app = create_app()


def main() -> None:
    import uvicorn

    host = os.environ.get("MOBILE_USE_ANDROID_HOST", "0.0.0.0")
    port = int(os.environ.get("MOBILE_USE_ANDROID_PORT", "8770"))
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
