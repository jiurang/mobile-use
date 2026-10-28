"""
Context variables for global state management.

Uses ContextVar to avoid prop drilling and maintain clean function signatures.
"""

from collections.abc import Callable, Coroutine
from datetime import datetime
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
from minitap.mobile_use.prompt_language import PromptLanguage
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


#: 稳定性判据的合法取值。
_STABILITY_MODES = ("fingerprint", "structural")


def _stability_mode_default() -> str:
    """稳定性判据的默认值 —— 环境变量 ``MOBILE_USE_STABILITY_MODE`` 优先。

    为什么走环境变量：这个开关要能**在不改代码、不改 manager 那一侧**的前提下
    做真机 A/B（``MOBILE_USE_STABILITY_MODE=structural`` 跑一次，改回去再跑一次）。
    环境变量里写了非法值就当没写（回落到现行行为），不因为一个环境变量让整轮挂掉。
    """
    import os

    value = (os.environ.get("MOBILE_USE_STABILITY_MODE") or "").strip().lower()
    return value if value in _STABILITY_MODES else "fingerprint"


class PerceptionConfig(BaseModel):
    """两条感知通道的开关 + 感知前的沉降等待。

    这两个开关是 **Manager 侧**的控制面（`AgentConfig.perception` →
    `MobileUseContext.perception`，见 `sdk/agent.py:722`）：Manager 建 agent 时
    决定哪几条通道开着，Executor 每次决策拿到的屏幕描述就按这个裁。

    关掉任何一条都是**纯粹减少**该通道带来的信息与开销，不会改变另外一条的
    行为：

    - `ui_hierarchy`（UI 元素表）：关掉后**不再有元素表**，Executor 只拿到
      OCR 块，寻址字段从 `index` 变成 `ocr_index`（`agents/executor/prompt.py`
      的 `has_element_table` 是这条的唯一判据，工具侧 `tools/utils.py`
      的 `resolve_target` 看的是同一份 `state.latest_screen_elements`）。
    - `ocr`（OCR 文本）：关掉后 Contextor **根本不跑 OCR**（真机实测 381~443ms
      /步），Executor 也拿不到 OCR 块。

    ⚠️ **原来还有第三条「图片」通道（`screenshot`），那条已下线。**
    2026-10-23 下线的是「**每轮附图**」那条通道（三态枚举
    ``always`` / ``on_demand`` / ``off`` + 输出 schema 里的 ``need_screenshot``
    字段）—— 它**仍然没有**：屏幕描述块里只有元素表或 OCR 块，
    模型也没有「举手要图」这个字段。

    2026-10-24 起截图改成了**按需工具** ``take_screenshot``
    （``tools/mobile/take_screenshot.py``）：模型自己喊一声要看图，
    框架把当屏那张图挂在**工具结果**上（多模态内容块）。
    ⇒ 它是 UI 元素表与 OCR 的**补充**，不是第三条感知通道：
    寻址仍然走元素表的 ``[i]``，图只补上图标 / 颜色 / 布局这些文字通道
    给不了的东西（provider 侧能不能吃图见该文件的实测记录）。

    ⚠️ **它不受这里的开关控制** —— Manager 侧把两条通道都关掉时，
    这个工具依然可用（那正是它能救场的时候）。
    """

    ui_hierarchy: bool = True
    ocr: bool = True
    ocr_config: OcrConfig = Field(default_factory=OcrConfig)

    # 动作之后的**沉降等待下限**（毫秒）。
    #
    # 含义不是「每轮固定等这么久」，而是「距离上一次动过屏幕的动作，
    # 至少要过这么久才允许再抓一次屏幕」。没动过屏幕就不等。
    #
    # 🔴 **2026-10-22 改：原来 3000，现在 200。**
    #
    # 原来那个 3000 的依据是「控件延迟挂载，抓早了拿到半成品」。
    # 2026-10-22 真机实测（Redmi Turbo 4 / Android 16）把「为什么 3 秒」
    # 拆开了：
    #
    # 1. **``dump_hierarchy`` 自带 waitForIdle** —— 屏幕还在动的时候它会
    #    **自己等**（实测滚动场景里它等了 609ms）。所以「防动画」这件事
    #    设备侧已经做了，外部下限在做同一件事，只是更粗。
    # 2. 6/6 次试验里，**不加外部下限时**最终内容在 274~2681ms 就位：
    #    返回键 274ms、页面导航 869~920ms、列表滚动 2015~2681ms。
    #    ⇒ 那个 3 秒下限**从来没有让树更完整**，它只是让每次抓屏晚 3 秒。
    #
    # 所以下限降到 200ms，剩下的「静 ≠ 全」（懒加载控件）交给
    # ``settle_max_retries`` 那圈**树稳定确认**兜（见
    # ``agents/contextor/contextor.py`` 的 ``_confirm_tree_stable``）。
    #
    # ⚠️ **代价要说清**：下限 200ms 意味着「动作后 ~0.55s 就抓屏」
    # （实测的工具→抓屏间隔），而页面导航场景的新页面控件要 ~0.9s 才挂全
    # ⇒ 那一类页面**可能**拿到残缺的表。**验收必须看「步数」和
    # 「失败目标数」**（表里缺控件 ⇒ 模型以为按钮不存在 ⇒ 绕路），
    # 不是「点错率」（表和屏是同一次抓的，索引契约自洽）。
    settle_delay_ms: int = Field(default=200, ge=0)

    # 沉降稳定确认的**重试上限**（次）。
    #
    # 一次确认 = 连续两次树级 dump 比指纹。相同 ⇒ 屏定下来了。
    # 不同 ⇒ 说明屏还在变 ⇒ 等 ``settle_delay_ms`` 再来一轮，最多这么多次。
    #
    # 设 1 = 只比一次（抓两棵树），这是最省的一档。
    # 设 0 = 完全不做稳定确认（只按下限等，然后直接抓）——
    # 最快，但导航场景大概率拿到残缺的表。
    settle_max_retries: int = Field(default=2, ge=0)

    # 稳定性判据用哪一套。
    #
    # * ``"fingerprint"``（默认，现行）—— 连续两次**全量树**的指纹相同才算稳。
    #   指纹含 ``text`` ⇒ 在**视频页**上点赞数/倒计时每秒都在变 ⇒ 永远不等 ⇒
    #   每次都跑满重试次数，最后还是「判定不稳，照用」。
    # * ``"structural"`` —— 只比**可交互元素的身份+位置**（不看 text）。
    #   依据是本仓库已有的结论（``tools/live_target.py``）：
    #   「OCR 的**内容会变**（点赞数在跳、时钟在走）… **位置比内容稳**」。
    #
    # 另：``structural`` 模式把**稳定性采样和建表采样合并**成两次 dump
    # （现行是 2~3 次稳定性 dump + 1 次建表 dump）。
    #
    # 默认值走环境变量 ``MOBILE_USE_STABILITY_MODE`` —— A/B 不用改代码，
    # 也不用先在 manager 那一侧加字段（runner 那边直接带环境变量就行）。
    stability_mode: Literal["fingerprint", "structural"] = Field(
        default_factory=_stability_mode_default
    )

    # ``structural`` 模式下，两次采样不一致时**额外**补采几次。
    #
    # 0 = 不补，直接用最新的一次（最省）。补采也只是「再等一个下限再采一次」，
    # 因为建表用的截图+OCR 已经和那棵树配对了（索引契约），换树就得重截+重 OCR。
    stability_extra_dumps: int = Field(default=0, ge=0)

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

    #: 提示词语言（``en`` / ``zh``）。**已经在 SDK 层定好了**（任务级 > Agent 级 >
    #: 环境变量 > 英文），这里只负责带进各个 agent。
    #:
    #: 为什么放在 context 而不是让每个 agent 自己读环境变量：同一个 Agent 上
    #: 可能同时跑中文任务和英文任务，环境变量是**进程级**的，读不了「这次任务」。
    prompt_language: PromptLanguage = PromptLanguage.EN

    #: **任务开始的时间**（带时区）。给 Executor / Cortex 看的。
    #:
    #: 口径：图**开始跑**的那一刻，不是「任务对象被创建」的那一刻 ——
    #: 创建到开跑之间还有装 APK / 锁应用这些准备动作（真机实测能花几分钟），
    #: 那段时间不是这个任务在干活。
    #:
    #: 为什么放在 context（而不是 state）：它是**一次运行**的属性，不是图状态 ——
    #: state 会被 checkpoint、被反复快照；这个时间在整场运行里只读一次。
    #:
    #: ⚠️ 它进的是提示词 ⇒ **必须整场运行逐字节不变**（否则前缀缓存被打断）。
    #: 所以只在起跑那一刻赋值，之后谁都不许改。
    task_started_at: datetime | None = None

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
