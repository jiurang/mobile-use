"""Entry point for `mobile-use-tui`.

This is a separate console script rather than a subcommand of `mobile-use`:
`main.py` is a single-command Typer app (`mobile-use "goal"`), and adding a
second command would turn it into a group and break that invocation.

Run `mobile-use-tui --demo` to see the UI with no device and no LLM key.
"""

from __future__ import annotations

import sys
from typing import Annotated

import typer

from minitap.mobile_use.sdk.types.agent import CloudDevicePlatform
from minitap.mobile_use.tui.app import MobileUseApp
from minitap.mobile_use.tui.events import AgentEventBridge
from minitap.mobile_use.tui.session import DeviceType, DemoSession, TuiSession

# A canned run so the UI can be exercised without hardware or credentials.
DEMO_SCRIPT: list[dict] = [
    {"kind": "node", "name": "planner", "think": 0.6},
    {"kind": "thought", "text": "拆解目标：打开微信 → 搜索张三 → 发一条消息"},
    {"kind": "node", "name": "orchestrator", "think": 0.4},
    {"kind": "node", "name": "contextor", "think": 0.8},
    {
        "kind": "thought",
        "text": "当前在手机桌面，微信图标位于 (540, 1180)，可以点击进入",
    },
    {"kind": "node", "name": "cortex", "think": 0.7},
    {"kind": "action", "name": "tap", "detail": "x=540, y=1180", "take": 0.5},
    {"kind": "node", "name": "executor_tools", "think": 0.3},
    {"kind": "action", "name": "wait_for_delay", "detail": "seconds=2.0", "take": 0.9},
    {"kind": "node", "name": "summarizer", "think": 0.5},
    {"kind": "node", "name": "contextor", "think": 0.7},
    {
        "kind": "thought",
        "text": "已进入微信首页，顶部有搜索框，点击它并输入「张三」",
    },
    {"kind": "node", "name": "cortex", "think": 0.6},
    {"kind": "action", "name": "tap", "detail": "x=360, y=228", "take": 0.4},
    {"kind": "action", "name": "focus_and_input_text", "detail": "text=张三", "take": 0.8},
    {"kind": "node", "name": "summarizer", "think": 0.4},
    {"kind": "result", "output": "已在微信中搜索到「张三」，会话窗口打开成功。"},
]


def cli() -> None:
    typer.run(main)


def main(
    demo: Annotated[
        bool,
        typer.Option("--demo", help="Run with a fake device and no LLM key (UI preview)."),
    ] = False,
    device_type: Annotated[
        DeviceType,
        typer.Option("--device-type", "-d", help="Device type: 'local' or 'cloud'."),
    ] = DeviceType.LOCAL,
    cloud_platform: Annotated[
        CloudDevicePlatform | None,
        typer.Option("--cloud-platform", help="Cloud platform: 'android' or 'ios'."),
    ] = None,
    with_video_recording_tools: Annotated[
        bool,
        typer.Option("--with-video-recording-tools", help="Enable video recording tools."),
    ] = False,
    wda_url: Annotated[
        str | None,
        typer.Option("--wda-url", help="Override WebDriverAgent URL."),
    ] = None,
    wda_auto_start_wda: Annotated[
        bool | None,
        typer.Option(
            "--wda-auto-start-wda/--no-wda-auto-start-wda",
            help="Auto-build and run WDA if not responding.",
        ),
    ] = None,
    wda_project_path: Annotated[
        str | None,
        typer.Option("--wda-project-path", help="Path to WebDriverAgent.xcodeproj."),
    ] = None,
    idb_host: Annotated[
        str | None,
        typer.Option("--idb-host", help="IDB companion host."),
    ] = None,
    idb_port: Annotated[
        int | None,
        typer.Option("--idb-port", help="IDB companion port."),
    ] = None,
) -> None:
    """Interactive terminal UI for the mobile-use agent."""
    if not demo and not sys.stdout.isatty():
        typer.secho(
            'mobile-use-tui needs an interactive terminal. Use `mobile-use "<goal>"` for CI.',
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(1)

    bridge = AgentEventBridge()

    if demo:
        session: TuiSession | DemoSession = DemoSession(bridge=bridge, script=DEMO_SCRIPT)
    else:
        session = TuiSession(
            bridge=bridge,
            device_type=device_type,
            cloud_platform=cloud_platform,
            video_recording_tools_enabled=with_video_recording_tools,
            wda_url=wda_url,
            wda_auto_start_wda=wda_auto_start_wda,
            wda_project_path=wda_project_path,
            idb_host=idb_host,
            idb_port=idb_port,
        )

    MobileUseApp(session, bridge).run()


if __name__ == "__main__":
    cli()
