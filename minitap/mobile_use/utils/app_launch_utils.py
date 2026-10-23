"""
Utilities for handling app locking and initial app launch logic.
"""

import asyncio
import time

from minitap.mobile_use.context import AppLaunchResult, MobileUseContext
from minitap.mobile_use.controllers.platform_specific_commands_controller import (
    get_current_foreground_package_async,
)
from minitap.mobile_use.controllers.unified_controller import UnifiedMobileController
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


# 轮询节奏（2026-10-17 实测调整）
# --------------------------------
# 原来启动 App 是「先无脑 sleep(1)，再每秒轮询一次」—— 最少空等 1 秒，
# 命中之后还要按秒对齐，最坏再多等接近 1 秒。合起来每次启动白扔 1~2 秒。
#
# 冷启动一个 App 通常几百毫秒就到前台了，所以改成「立刻查一次，查不到就
# 短间隔重试，逐步退避到 1s 上限」：命中得快的那部分轮询更密，等得久的那部分
# 不会把 adb 打爆。
POLL_INTERVAL_START = 0.15
POLL_INTERVAL_MAX = 1.0
POLL_BACKOFF = 1.6


async def _poll_for_app_ready(
    ctx: MobileUseContext,
    app_package: str,
    max_poll_seconds: int = 15,
    poll_interval: float = POLL_INTERVAL_START,
) -> tuple[bool, str | None]:
    """等目标 app 切到前台。**等到了就早退，等不到也如实说等不到。**

    🔴 **不再因为「前台包名对不上」就判失败**（2026-10-24 真机实测，execution 851）

    老语义是：前台包名 != 目标包名 ⇒ **立刻**判失败。这个判断在 MIUI 上是错的，
    而且错得致命：

    851（Redmi Turbo 4 / HyperOS）里 `launch_app 设置` 的启动命令**确实成功了**
    （monkey 报 ``Events injected: 1``），屏幕上的画面也正是执行器要的那个 ——
    MIUI 的「设置 → 应用设置/应用管理」页面的宿主是 ``com.miui.securitycenter``，
    前台包名是它而不是 ``com.android.settings`` ⇒ 老逻辑当场判「启动失败」
    ⇒ 工具报失败 ⇒ 模型以为「还没验成」⇒ 在两个验证条件之间无限轮转：
    54 次 executor 调用、0 次 ``done``、59.6 万 token、13 分钟。

    ⇒ 现在的语义：
      - **包名对上了** ⇒ 成功，立刻返回（快路径，一点时间都不浪费）。
      - **包名对不上** ⇒ 只是「**还没切过去**」（或者被系统宿主包接管了），
        **继续等**，不再当场判失败 —— 启动命令已经发出去了，重发没用。
      - **等满了** ⇒ 如实返回「没能确认」，但**这不算启动失败**：
        启动命令是成功的，只是没确认它切到了前台。画面很可能已经在屏上了
        （851 就是这样）—— 判失败是撒谎。下一轮执行器会重新抓屏，
        **它看得见真相**，该由它判断。

    Returns:
        ``(confirmed, detail)``：
          - ``confirmed=True`` ⇒ 前台就是目标包，早退了，``detail`` 为 None。
          - ``confirmed=False`` ⇒ 等满了没能确认；``detail`` 是给模型看的一句话
            （说清「命令发出去了，但没确认切过去」）。
    """
    deadline = time.monotonic() + max_poll_seconds
    interval = poll_interval
    elapsed = 0.0
    polls = 0
    last_seen: str | None = None

    while True:
        polls += 1
        current_package = await get_current_foreground_package_async(ctx)
        if current_package is not None:
            last_seen = current_package

        if current_package == app_package:
            logger.success(f"App {app_package} is ready (took ~{elapsed:.2f}s, {polls} polls)")
            return True, None

        # ⚠️ **包名对不上 ⇒ 继续等，不是失败。**
        # 命令已经发出去了（monkey 报 Events injected: 1），画面很可能已经切过去了，
        # 只是前台包名不是我们要的那个（MIUI 会把「设置 → 应用设置」交给
        # com.miui.securitycenter 托管）。当场判失败 = 骗模型 = 它会重发、
        # 会重验、会在两个条件之间转圈（851）。
        logger.debug(f"Poll {polls}: foreground is {current_package!r}, waiting for {app_package}")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        await asyncio.sleep(min(interval, remaining))
        elapsed = max_poll_seconds - remaining
        interval = min(interval * POLL_BACKOFF, POLL_INTERVAL_MAX)

    detail = (
        f"启动命令已执行，但 {max_poll_seconds}s 内没能确认 {app_package} 切到前台"
        f"（前台现在是 {last_seen or '未知'}）。画面可能已经在屏上了 —— "
        f"下一轮先读屏确认，别重发启动命令。"
    )
    logger.warning(detail)
    return False, detail


async def launch_app_with_retries(
    ctx: MobileUseContext,
    app_package: str,
    max_retries: int = 3,
    max_poll_seconds: int = 15,
) -> tuple[bool, str | None]:
    """
    启动 app；**启动命令**本身没跑成就重试。

    Args:
        ctx: Mobile use context
        app_package: Package name (Android) or bundle ID (iOS) to launch
        max_retries: Maximum number of launch attempts (default: 3)
        max_poll_seconds: 启动命令成功后，等它切到前台最多等多久（default: 15s）

    Returns:
        ``(ok, detail)`` —— **detail 的含义随 ok 变**：
          - ``ok=False`` ⇒ detail 是**失败原因**（模型要知道为什么）。
          - ``ok=True``  ⇒ detail 是**成功但有保留**（目前只有一种：
            「没能确认切到前台」），可为 None。两个值都会原样进模型上下文。

    🔴 **失败只保留一种：启动命令本身没跑成**（2026-10-24 真机实测，execution 851）

    以前这里把两件事混在一起了：
      1. 「启动命令跑成了吗」—— 这是**真的失败**（adb 断了、包没有可启动的
         Activity）⇒ 该失败，该重试。
      2. 「app 起来了吗」—— 用**前台包名 == 目标包名**判定 ⇒ 这在 MIUI 上是
         **错的**：MIUI 的「设置 → 应用设置/应用管理」页面宿主是
         ``com.miui.securitycenter``，前台包名是它而不是 ``com.android.settings``
         ⇒ 启动命令明明成功了（monkey 报 ``Events injected: 1``）、画面也正是
         执行器要的那个，却被判成「启动失败」⇒ 工具报失败 ⇒ 模型以为「还没验成」
         ⇒ 在两个验证条件之间无限轮转（54 次 executor 调用、0 次 done、
         59.6 万 token、13 分钟）。

    ⇒ 现在：
      - **启动命令失败** ⇒ 失败 + 重试（最多 max_retries 次），并把**原因原文**
        带给模型。
      - **启动命令成功、只是没确认切到前台** ⇒ **不算失败**，也**不重试**
        （命令已经发出去了，重发没用）⇒ 返回成功 + 一句「没能确认」的实话，
        让模型**自己读屏判断** —— 它下一轮就会重新抓屏，看得见真相。

    ⚠️ 原因里只有包名和系统话术，没有用户数据；遥测侧另有代码化处理
    （见 ``agents/executor/executor.py`` 的 ``_rejection_code``），不外传原文。
    """
    # 最后一次 attempt 的真实原因。前面几次的也可能有价值，但模型上下文有限，
    # 留最后一次（它最接近当下的事实）就够了。
    last_reason: str | None = None

    for attempt in range(1, max_retries + 1):
        logger.info(f"Launch attempt {attempt}/{max_retries} for app {app_package}")

        controller = UnifiedMobileController(ctx)
        try:
            launch_success = await controller.launch_app(app_package)
        except Exception as exc:  # noqa: BLE001
            # 控制器**失败时抛异常**（消息里带原因原文）—— 见
            # ``controllers/device_controller.py`` 的契约说明。
            # 原因要原文带给模型：只回一句「失败了」，它没法决定下一步。
            launch_success = False
            last_reason = str(exc) or f"{type(exc).__name__}"
            logger.error(f"Launch attempt {attempt} failed: {last_reason}")
        else:
            if launch_success:
                logger.success(f"App {app_package} launched")
                # 命令发出去了 ⇒ 等它切到前台（**等，不判**）。
                confirmed, detail = await _poll_for_app_ready(
                    ctx, app_package, max_poll_seconds
                )
                if confirmed:
                    return True, None
                # 没能确认 ≠ 启动失败。命令已经成功，画面很可能已经在屏上。
                # **不重试** —— 重发启动命令没用（它已经在那个任务里了）。
                return True, detail
            last_reason = f"Failed to execute launch command for {app_package}"
            logger.error(last_reason)

        if attempt < max_retries:
            await asyncio.sleep(1)

    error_msg = _attempts_failed(app_package, max_retries, last_reason)
    logger.error(error_msg)
    return False, error_msg


def _attempts_failed(app_package: str, max_retries: int, last_reason: str | None) -> str:
    """「试了 N 次还是没起来」+ **最后一次的真实原因**。

    ⚠️ 别再把原因吞掉 —— 这个字符串会原样进模型上下文（见
    ``tools/mobile/launch_app.py`` 的 ``on_failure_fn``）。模型看不见原因，
    就只能把「失败」理解成「目标没达成」⇒ 换个条件再来一遍 ⇒ 死循环。
    """
    head = f"Failed to launch {app_package} after {max_retries} attempts"
    if not last_reason:
        return head
    return f"{head}. Last attempt failed because: {last_reason}"


async def _handle_initial_app_launch(
    ctx: MobileUseContext,
    locked_app_package: str,
) -> AppLaunchResult:
    """
    Handle initial app launch verification and launching if needed.

    If locked_app_package is set:
    1. Check if the app is already in the foreground
    2. If not, attempt to launch it (with retries)
    3. Return status with success/error information

    Args:
        ctx: Mobile use context
        locked_app_package: Package name (Android) or bundle ID (iOS) to lock to, or None

    Returns:
        AppLaunchResult with launch status and error information
    """
    if not locked_app_package:
        error_msg = f"Invalid locked_app_package: '{locked_app_package}'"
        logger.error(error_msg)
        return AppLaunchResult(
            locked_app_package=locked_app_package,
            locked_app_initial_launch_success=False,
            locked_app_initial_launch_error=error_msg,
        )

    logger.info(f"Starting initial app launch for package: {locked_app_package}")

    try:
        current_package = await get_current_foreground_package_async(ctx)
        logger.info(f"Current foreground app: {current_package}")

        if current_package == locked_app_package:
            logger.info(f"App {locked_app_package} is already in foreground")
            return AppLaunchResult(
                locked_app_package=locked_app_package,
                locked_app_initial_launch_success=True,
                locked_app_initial_launch_error=None,
            )

        logger.info(f"App {locked_app_package} not in foreground, attempting to launch")
        success, error_msg = await launch_app_with_retries(ctx, locked_app_package)

        return AppLaunchResult(
            locked_app_package=locked_app_package,
            locked_app_initial_launch_success=success,
            locked_app_initial_launch_error=error_msg,
        )

    except Exception as e:
        error_msg = f"Exception during initial app launch: {str(e)}"
        logger.error(error_msg)
        return AppLaunchResult(
            locked_app_package=locked_app_package,
            locked_app_initial_launch_success=False,
            locked_app_initial_launch_error=error_msg,
        )
