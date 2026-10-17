"""
Context variables for global state management.

Uses ContextVar to avoid prop drilling and maintain clean function signatures.
"""

from collections.abc import Callable, Coroutine
from enum import StrEnum
from pathlib import Path
from typing import Literal

from adbutils import AdbClient
from openai import BaseModel
from pydantic import ConfigDict, Field

from minitap.mobile_use.agents.planner.types import Subgoal
from minitap.mobile_use.clients.ios_client import IosClientWrapper
from minitap.mobile_use.clients.ui_automator_client import UIAutomatorClient
from minitap.mobile_use.config import AgentNode, LLMConfig
from minitap.mobile_use.controllers.cloud_device_controller import CloudAndroidController
from minitap.mobile_use.services.ocr.types import OcrConfig


class AppLaunchResult(BaseModel):
    """Result of initial app launch attempt."""

    locked_app_package: str
    locked_app_initial_launch_success: bool | None
    locked_app_initial_launch_error: str | None


class DevicePlatform(StrEnum):
    """Mobile device platform enumeration."""

    ANDROID = "android"
    IOS = "ios"


class DeviceContext(BaseModel):
    host_platform: Literal["WINDOWS", "LINUX"]
    mobile_platform: DevicePlatform
    device_id: str
    device_width: int
    device_height: int

    def to_str(self):
        return (
            f"Host platform: {self.host_platform}\n"
            f"Mobile platform: {self.mobile_platform.value}\n"
            f"Device ID: {self.device_id}\n"
            f"Device width: {self.device_width}\n"
            f"Device height: {self.device_height}\n"
        )


class ExecutionSetup(BaseModel):
    """Execution setup for a task."""

    traces_path: Path | None = None
    trace_name: str | None = None
    app_lock_status: AppLaunchResult | None = None

    def get_locked_app_package(self) -> str | None:
        """
        Get the locked app package name if app locking is enabled.

        Returns:
            The locked app package name, or None if app locking is not enabled.
        """
        if self.app_lock_status:
            return self.app_lock_status.locked_app_package
        return None


IsReplan = bool


class PerceptionConfig(BaseModel):
    """三条感知通道的开关。三个默认全开。

    关掉任何一条都是**纯粹减少**该通道带来的信息与开销，不会改变另外两条的
    行为：

    - `screenshot`：Cortex 不再收到截图。视觉模型仍能从 UI 层级树和 OCR 读文字，
      但看不到颜色、图标、图片内容、遮挡关系。
    - `ui_hierarchy`：Cortex 不再收到 UI 元素列表，也不再拿它做坐标合法性校验
      （`tools/bounds_validation.py` 会退化成只看 OCR 框）。
    - `ocr`：Contextor 不跑 OCR，Cortex 不收到 OCR 文本块，坐标校验退回只看层级树。
    """

    screenshot: bool = True
    ui_hierarchy: bool = True
    ocr: bool = True
    ocr_config: OcrConfig = Field(default_factory=OcrConfig)


class MobileUseContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    trace_id: str
    device: DeviceContext
    llm_config: LLMConfig
    adb_client: AdbClient | None = None
    ui_adb_client: UIAutomatorClient | None = None
    ios_client: IosClientWrapper | None = None
    cloud_android_controller: CloudAndroidController | None = None
    execution_setup: ExecutionSetup | None = None
    on_agent_thought: Callable[[AgentNode, str], Coroutine] | None = None
    on_plan_changes: Callable[[list[Subgoal], IsReplan], Coroutine] | None = None
    video_recording_enabled: bool = False
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)

    def get_adb_client(self) -> AdbClient:
        if self.adb_client is None:
            raise ValueError("No ADB client in context.")
        return self.adb_client  # type: ignore

    def get_ui_adb_client(self) -> UIAutomatorClient:
        if self.ui_adb_client is None:
            raise ValueError("No UIAutomator client in context.")
        return self.ui_adb_client

    def get_ios_client(self) -> IosClientWrapper:
        """Get the iOS client (IDB for simulators, WDA for physical devices)."""
        if self.ios_client is None:
            raise ValueError("No iOS client in context.")
        return self.ios_client
