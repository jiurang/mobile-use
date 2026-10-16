"""Agent session ownership for the interactive TUI.

`Agent.run_task` already supports several sequential tasks against one live
device session (`Agent._task_lock` only serialises *concurrent* runs), so a REPL
is a matter of holding the `Agent` open instead of exiting after one goal.

`TuiSession` owns that lifetime: build once, `init()` once, `run_goal()` many
times, `close()` once. `DemoSession` is the same surface with a fake backend so
the UI can be exercised without a phone attached.
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from minitap.mobile_use.tui.events import AgentEventBridge, EventKind


class DeviceType(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """What the status panel shows about the device."""

    label: str = "unknown"
    platform: str = "—"
    device_id: str = ""
    width: int = 0
    height: int = 0

    @property
    def screen(self) -> str:
        if not self.width or not self.height:
            return "—"
        return f"{self.width}×{self.height}"


class TuiSessionError(RuntimeError):
    pass


class SessionBase(ABC):
    """What the TUI needs from a running backend."""

    @property
    @abstractmethod
    def device_info(self) -> DeviceInfo: ...

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def run_goal(self, goal: str) -> Any: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...


class TuiSession(SessionBase):
    """A real `Agent`, kept alive across goals."""

    def __init__(
        self,
        *,
        bridge: AgentEventBridge,
        device_type: DeviceType = DeviceType.LOCAL,
        cloud_platform: Any = None,
        video_recording_tools_enabled: bool = False,
        wda_url: str | None = None,
        wda_timeout: float | None = None,
        wda_auto_start_iproxy: bool | None = None,
        wda_auto_start_wda: bool | None = None,
        wda_project_path: str | None = None,
        wda_startup_timeout: float | None = None,
        idb_host: str | None = None,
        idb_port: int | None = None,
    ) -> None:
        self._bridge = bridge
        self._device_type = device_type
        self._cloud_platform = cloud_platform
        self._video = video_recording_tools_enabled
        self._wda_url = wda_url
        self._wda_timeout = wda_timeout
        self._wda_auto_start_iproxy = wda_auto_start_iproxy
        self._wda_auto_start_wda = wda_auto_start_wda
        self._wda_project_path = wda_project_path
        self._wda_startup_timeout = wda_startup_timeout
        self._idb_host = idb_host
        self._idb_port = idb_port
        self._agent: Any = None
        self._info = DeviceInfo(label=self._fallback_label())

    def _fallback_label(self) -> str:
        if self._device_type is DeviceType.CLOUD:
            platform = self._cloud_platform.value if self._cloud_platform else "unknown"
            return f"cloud:{platform}"
        return "local device"

    @property
    def device_info(self) -> DeviceInfo:
        return self._info

    def _build_config(self):
        """Assemble the AgentConfig. Split out from start() so it can be
        exercised without a device.

        Every call site passes **explicit** keyword arguments on purpose: an
        earlier `with_overrides(**self._wda)` spelled the keys as a dict, which
        silently turned a renamed SDK keyword into a runtime TypeError
        (`with_overrides() got an unexpected keyword argument 'url'`) and also
        erased pyright's signature check. Spelling them out makes both the
        type checker and the test suite catch it.
        """
        from minitap.mobile_use.clients.ios_client_config import (
            IdbClientConfig,
            IosClientConfig,
            WdaClientConfig,
        )
        from minitap.mobile_use.config import settings
        from minitap.mobile_use.sdk.builders import Builders

        from minitap.mobile_use.config import initialize_llm_config
        from minitap.mobile_use.sdk.types.task import AgentProfile

        llm_config = initialize_llm_config()
        profile = AgentProfile(name="default", llm_config=llm_config)
        config = Builders.AgentConfig.with_default_profile(profile=profile)
        if self._video:
            config.with_video_recording_tools()

        if self._device_type is DeviceType.CLOUD:
            if self._cloud_platform is None:
                raise TuiSessionError("--cloud-platform is required with --device-type cloud")
            config.for_cloud_device(self._cloud_platform)
        else:
            wda_config = WdaClientConfig.with_overrides(
                wda_url=self._wda_url,
                timeout=self._wda_timeout,
                auto_start_iproxy=self._wda_auto_start_iproxy,
                auto_start_wda=self._wda_auto_start_wda,
                wda_project_path=self._wda_project_path,
                wda_startup_timeout=self._wda_startup_timeout,
            )
            idb_config = IdbClientConfig.with_overrides(host=self._idb_host, port=self._idb_port)
            config.with_ios_client_config(IosClientConfig(wda=wda_config, idb=idb_config))
            if settings.ADB_HOST:
                config.with_adb_server(host=settings.ADB_HOST, port=settings.ADB_PORT)

        # The hook that makes the whole TUI possible: no agent or graph change.
        config.with_graph_config_callbacks([self._bridge])
        return config

    async def start(self) -> None:
        # Imported lazily: pulling the SDK in costs ~2s and the TUI must be able
        # to start (and fail with a readable message) before that happens.
        from minitap.mobile_use.sdk import Agent

        agent = Agent(config=self._build_config().build())
        self._agent = agent
        try:
            await agent.init(
                retry_count=int(os.getenv("MOBILE_USE_HEALTH_RETRIES", 5)),
                retry_wait_seconds=int(os.getenv("MOBILE_USE_HEALTH_DELAY", 2)),
            )
        except BaseException:
            self._agent = None
            raise

        # `Agent` exposes no public accessor for the resolved device; the
        # context is populated during init(), so read it now.
        context = getattr(agent, "_device_context", None)
        if context is not None:
            self._info = DeviceInfo(
                label=self._fallback_label(),
                platform=str(context.mobile_platform.value),
                device_id=context.device_id,
                width=context.device_width,
                height=context.device_height,
            )

    async def run_goal(self, goal: str) -> Any:
        if self._agent is None:
            raise TuiSessionError("session not started")
        return await self._agent.run_task(goal=goal)

    def stop(self) -> None:
        if self._agent is not None:
            self._agent.stop_current_task()

    async def close(self) -> None:
        if self._agent is None:
            return
        agent, self._agent = self._agent, None
        await agent.clean()


class DemoSession(SessionBase):
    """Fake backend: canned agent activity and a synthetic device.

    Used by `mobile-use-tui --demo` and by the UI tests, so the TUI can be
    verified without a device or an LLM key.
    """

    def __init__(self, *, bridge: AgentEventBridge, script: list[dict[str, Any]]) -> None:
        self._bridge = bridge
        self._script = script
        self._running = False
        self._task: asyncio.Task | None = None
        self._offset = 0

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            label="demo device",
            platform="android",
            device_id="demo-0000",
            width=1220,
            height=2712,
        )

    async def start(self) -> None:
        return None

    async def run_goal(self, goal: str) -> Any:
        self._running = True
        self._offset += 1
        try:
            for beat in self._script:
                if beat.get("sleep"):
                    await asyncio.sleep(beat["sleep"])
                    if not self._running:
                        return None
                kind = beat["kind"]
                if kind == "thought":
                    self._bridge.emit(EventKind.THOUGHT, beat["text"])
                elif kind == "node":
                    self._bridge.emit(EventKind.NODE_START, beat["name"])
                    await asyncio.sleep(beat.get("think", 0.2))
                    self._bridge.emit(
                        EventKind.NODE_END,
                        beat["name"],
                        duration_ms=int(beat.get("think", 0.2) * 1000),
                    )
                elif kind == "action":
                    self._bridge.emit(
                        EventKind.ACTION_START, beat["name"], detail=beat.get("detail", "")
                    )
                    await asyncio.sleep(beat.get("take", 0.25))
                    self._bridge.emit(
                        EventKind.ACTION_END,
                        beat["name"],
                        duration_ms=int(beat.get("take", 0.25) * 1000),
                        detail=beat.get("result", "ok"),
                    )
                elif kind == "tokens":
                    self._bridge.emit(
                        EventKind.USAGE, "tokens", payload=tuple(beat.get("totals", (0, 0)))
                    )
                elif kind == "result":
                    return beat.get("output")
            return f"done: {goal}"
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False

    async def close(self) -> None:
        return None
