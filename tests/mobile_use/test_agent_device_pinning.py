"""The adb executable check must only guard the code path that needs it.

``Agent._init_internal`` used to refuse to start whenever ``which("adb")`` and
``which("xcrun")`` were both absent. That check exists for one reason:
``get_first_device()`` shells out to ``adb devices`` when the caller did not say
which device to drive.

runner-host always says — ``builder.for_device(platform=..., device_id=serial)``
and ``with_adb_server(...)`` — so on the phone the adbutils client reaches the
device over TCP and no executable is ever spawned. The check rejected a setup
that works, with ``ExecutableNotFoundError: ADB or Xcode Command Line Tools not
found in PATH``, before the agent took its first step.

Both directions are pinned here: a pinned device must not consult PATH, and an
unpinned one must still get the error rather than silently picking nothing.
"""

from __future__ import annotations

from typing import Any

import pytest

from minitap.mobile_use.sdk import Agent
from minitap.mobile_use.sdk.builders import Builders
from minitap.mobile_use.sdk.types.agent import AgentProfile
from minitap.mobile_use.sdk.types.exceptions import ExecutableNotFoundError
from minitap.mobile_use.context import DevicePlatform


def _profile() -> AgentProfile:
    # Load the project's own default profile file rather than an API key: what
    # this exercises is the pre-flight in _init_internal, which runs before any
    # model is contacted, and no credential should be needed to reach it.
    return AgentProfile(name="test", from_file="default")


@pytest.fixture
def no_which(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make PATH look empty, the way it does inside the APK."""
    import minitap.mobile_use.sdk.agent as agent_module

    seen: list[str] = []

    def fake_which(name: str) -> None:
        seen.append(name)
        return None

    monkeypatch.setattr(agent_module, "which", fake_which)
    return seen


@pytest.mark.asyncio
async def test_pinned_device_never_consults_path(no_which: list[str]) -> None:
    builder = Builders.AgentConfig.with_default_profile(profile=_profile())
    builder.for_device(platform=DevicePlatform.ANDROID, device_id="pinme-1234")
    builder.with_adb_server(host="127.0.0.1", port=5037)
    agent = Agent(config=builder.build(validate_profiles=False))

    failure: BaseException | None = None
    try:
        # Whether a device is actually reachable is not what this test is about;
        # the agent may well fail later, on the connection. What must not happen
        # is the PATH check rejecting the configuration up front.
        await agent.init(retry_count=0, retry_wait_seconds=0)
    except BaseException as exc:  # noqa: BLE001
        failure = exc

    assert not isinstance(failure, ExecutableNotFoundError), (
        "a pinned device must not require adb/xcrun on PATH; got "
        f"{type(failure).__name__ if failure else None}: {failure}"
    )
    assert no_which == [], (
        f"PATH was consulted for a pinned device: {no_which}"
    )


@pytest.mark.asyncio
async def test_unpinned_device_still_reports_the_missing_executable(
    no_which: list[str],
) -> None:
    """The check is not removed, only moved to the branch that needs it."""
    builder = Builders.AgentConfig.with_default_profile(profile=_profile())
    agent = Agent(config=builder.build(validate_profiles=False))

    with pytest.raises(ExecutableNotFoundError):
        await agent.init(retry_count=0, retry_wait_seconds=0)

    assert no_which, "the unpinned path should still check PATH before adb devices"
