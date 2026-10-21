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
    """三条感知通道的开关 + 感知前的沉降等待。

    关掉任何一条都是**纯粹减少**该通道带来的信息与开销，不会改变另外两条的
    行为：

    - `screenshot`：Cortex 不再收到截图。视觉模型仍能从 UI 层级树和 OCR 读文字，
      但看不到颜色、图标、图片内容、遮挡关系。
    - `ui_hierarchy`：Cortex 不再收到 UI 元素列表，也不再拿它做坐标合法性校验
      （`tools/bounds_validation.py` 会退化成只看 OCR 框）。
    - `ocr`：Contextor 不跑 OCR，Cortex 不收到 OCR 文本块，坐标校验退回只看层级树。

    `screenshot` 是**三态**：

    - `True` —— 每轮都把截图塞进 prompt（老行为）。
    - `False` —— 完全不发图。
    - `"on_demand"` —— **默认不发图**；Cortex 在结构化输出里把 `need_screenshot`
      置 true 时，才在**同一轮内**补一张图再问一次。

    为什么默认改成 `"on_demand"`（2026-10-21 实测）
    ------------------------------------------------
    截图进 prompt 的代价是**每轮都付、且永远付**：

    - 它是**不可缓存**的 —— 在 prompt 末尾，而 provider 的缓存单元建在「用户输入
      的末尾」，请求总长一变（base64 长度随画面变）就匹配不上。
    - 实测那 47,672 字符的 base64 占了整个易变尾部的 **92%**；砍掉它之后
      12 轮里只有 1 轮低于 80%，带它时约 1/3 的轮次掉到 6-12%
      （见 `doc/cache-hit-rate-audit-20261020.md` 第 8 节）。

    绝大多数决策靠 UI 元素表 + OCR 就够（模型报的是索引，坐标由代码给），
    图只在「颜色 / 图标 / 遮挡 / 图片内容」这些文字表达不了的地方才需要。
    """

    screenshot: bool | Literal["on_demand"] = "on_demand"
    ui_hierarchy: bool = True
    ocr: bool = True
    ocr_config: OcrConfig = Field(default_factory=OcrConfig)

    @property
    def screenshot_label(self) -> str:
        """给 prompt 模板显示用的通道状态文案。

        模板不能直接用 `perception.screenshot` 的真假判断 —— `"on_demand"` 是
        真值，模板会把它显示成「enabled」，模型就以为图已经在手里了。
        """
        if self.screenshot is True:
            return "enabled"
        if self.screenshot is False:
            return "DISABLED"
        return "on demand"

    # 动作之后的**沉降等待下限**（毫秒）。
    #
    # 含义不是「每轮固定等这么久」，而是「距离上一次动过屏幕的动作，
    # 至少要过这么久才允许再抓一次屏幕」。没动过屏幕就不等。
    #
    # 为什么要它（2026-10-20 真机实测，Redmi Turbo 4）：
    # 上滑换视频后，交互控件（like 按钮、评论区这些）是**延迟挂载**的 ——
    # 实测新页面的 like 类 resource-id 在动作后 2.0s / 2.5s 才出现在层级树里。
    # 抓早了会拿到一棵「控件还没挂上」的树，看起来像「这个按钮不存在」。
    #
    # 为什么默认 3s 而不是实测的 1~2s：控件挂载时间随 App 和机型浮动，
    # 宁可多等 1s，也不要因为差 300ms 就抓了个半成品。
    #
    # 它是**下限**不是固定延迟：真机实测决策延迟中位 19.2s，所以绝大多数轮次
    # 根本不会触发它（时间早就够了），它只在「动作后立刻就要再抓」的快路径上
    # 起作用。配 0 就是关掉这个下限。
    settle_delay_ms: int = Field(default=3000, ge=0)

    # 是否压缩发给 LLM 的 thoughts 历史。
    #
    # 关掉 = 回到压缩前的行为（逐条原样重发）。这是回退开关，也是 A/B 的
    # 对照组开关 —— 压缩必须能一键退回，否则线上出问题没法归因。
    #
    # 为什么默认开（2026-10-20 真机归档实测，exec 212，122 条真实 thoughts）：
    # thoughts 是 cortex prompt 里唯一随轮次线性增长的段，末轮占整个 prompt
    # 的 65%。压缩后末轮 prompt 降 40.3%、全链路 token 降 34.2%，且因为
    # 压缩是**位置无关**的（前缀字节不变），前缀缓存命中率仍有 91.1%
    # （基线 93.5%），满价重算反而比基线低 9.2%。
    #
    # 详见 doc/thoughts-compaction-plan.md。
    thoughts_compaction: bool = True


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
