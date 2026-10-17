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
    """
    Poll for app to be ready after launch.

    Treats mCurrentFocus=null as a loading state and keeps polling.
    Only fails if we get a different (non-null) package or timeout.

    先立刻查一次（不做任何前置 sleep），之后按 poll_interval 退避重查，
    间隔从 poll_interval 涨到 POLL_INTERVAL_MAX 为止。

    Args:
        ctx: Mobile use context
        app_package: Expected package name
        max_poll_seconds: Maximum time to poll (default: 15s)
        poll_interval: 首次重试间隔，默认 POLL_INTERVAL_START

    Returns:
        Tuple of (success: bool, error_message: str | None)
    """
    deadline = time.monotonic() + max_poll_seconds
    interval = poll_interval
    elapsed = 0.0
    polls = 0

    while True:
        polls += 1
        current_package = await get_current_foreground_package_async(ctx)

        if current_package == app_package:
            logger.success(f"App {app_package} is ready (took ~{elapsed:.2f}s, {polls} polls)")
            return True, None

        if current_package is not None:
            error_msg = (
                f"Wrong app in foreground: expected '{app_package}', got '{current_package}'"
            )
            logger.warning(error_msg)
            return False, error_msg

        logger.debug(f"Poll {polls}: App loading (mCurrentFocus=null)...")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        await asyncio.sleep(min(interval, remaining))
        elapsed = max_poll_seconds - remaining
        interval = min(interval * POLL_BACKOFF, POLL_INTERVAL_MAX)

    current_package = await get_current_foreground_package_async(ctx)
    error_msg = (
        f"Timeout waiting for {app_package} to load after {max_poll_seconds}s. "
        f"Current foreground: {current_package}"
    )
    logger.error(error_msg)
    return False, error_msg


async def launch_app_with_retries(
    ctx: MobileUseContext,
    app_package: str,
    max_retries: int = 3,
    max_poll_seconds: int = 15,
) -> tuple[bool, str | None]:
    """
    Launch an app with retry logic and smart polling.

    Args:
        ctx: Mobile use context
        app_package: Package name (Android) or bundle ID (iOS) to launch
        max_retries: Maximum number of launch attempts (default: 3)
        max_poll_seconds: Maximum time to wait for app to load per attempt (default: 15s)

    Returns:
        Tuple of (success: bool, error_message: str | None)
    """
    for attempt in range(1, max_retries + 1):
        logger.info(f"Launch attempt {attempt}/{max_retries} for app {app_package}")

        controller = UnifiedMobileController(ctx)
        launch_success = await controller.launch_app(app_package)
        if not launch_success:
            error_msg = f"Failed to execute launch command for {app_package}"
            logger.error(error_msg)
            if attempt == max_retries:
                return False, error_msg
            await asyncio.sleep(1)
            continue

        # 不要再无脑 sleep(1) 了：_poll_for_app_ready 进去第一件事就是查，
        # 那一秒纯浪费。App 没起来时轮询自己会退避重试。
        success, error_msg = await _poll_for_app_ready(ctx, app_package, max_poll_seconds)

        if success:
            return True, None

        if attempt < max_retries:
            logger.warning(f"Attempt {attempt} failed: {error_msg}. Retrying...")
            await asyncio.sleep(1)

    error_msg = f"Failed to launch {app_package} after {max_retries} attempts"
    logger.error(error_msg)
    return False, error_msg


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
