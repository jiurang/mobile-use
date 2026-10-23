"""`utils/app_launch_utils.py` 的启动语义。

🔴 **2026-10-24：就绪判定的语义改了。**

以前启动命令跑完会轮询前台包名，**包名对不上就判失败**。这个判断在 MIUI 上是错的：
MIUI 的「设置 → 应用设置/应用管理」页面宿主是 ``com.miui.securitycenter``
（不是 ``com.android.settings``）⇒ 启动命令**明明把目标画面打开了**
（monkey 报 ``Events injected: 1``），轮询却判它失败 ⇒ 工具报「失败」⇒
模型以为「还没验成」⇒ 在两个验证条件之间无限轮转
（真机实测 execution 851：54 次 executor 调用、0 次 ``done``、59.6 万 token）。

现在守四件事：
1. **包名对上了** ⇒ 成功，立刻返回，一次 sleep 都不该有。
2. **包名对不上** ⇒ **继续等**，不当场判失败（851 的根因，必须守住）。
3. **等满了** ⇒ 如实说「没能确认」，但这**不算启动失败**。
4. **启动命令本身失败** ⇒ 失败 + 重试，并把**真实原因**带给模型。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from minitap.mobile_use.utils import app_launch_utils as alu


def _ctx():
    return object()


class _Clock:
    """可控的 monotonic + sleep，让轮询不真的花时间。"""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock():
    c = _Clock()
    with (
        patch.object(alu.time, "monotonic", c.monotonic),
        patch.object(alu.asyncio, "sleep", c.sleep),
    ):
        yield c


class TestPollingCadence:
    @pytest.mark.asyncio
    async def test_first_check_happens_without_sleeping(self, clock):
        """**核心断言**：App 已经在前台时，一次 sleep 都不该发生。"""
        with patch.object(
            alu, "get_current_foreground_package_async", AsyncMock(return_value="com.app")
        ):
            confirmed, detail = await alu._poll_for_app_ready(_ctx(), "com.app")

        assert confirmed is True
        assert detail is None
        assert clock.sleeps == [], f"命中即退还要 sleep？{clock.sleeps}"

    @pytest.mark.asyncio
    async def test_backoff_grows_then_caps(self, clock):
        """前几次密、后面退到 1s 上限，绝不无限增长。"""
        with patch.object(
            alu,
            "get_current_foreground_package_async",
            AsyncMock(return_value=None),  # 永远 mCurrentFocus=null
        ):
            confirmed, detail = await alu._poll_for_app_ready(
                _ctx(), "com.app", max_poll_seconds=30
            )

        assert confirmed is False and detail is not None
        assert clock.sleeps, "应该轮询过"
        assert clock.sleeps[0] == pytest.approx(alu.POLL_INTERVAL_START)
        assert clock.sleeps[1] > clock.sleeps[0], "间隔应该递增"
        assert max(clock.sleeps) <= alu.POLL_INTERVAL_MAX + 1e-9
        # 退避封顶后不再增长
        capped = [s for s in clock.sleeps if s >= alu.POLL_INTERVAL_MAX]
        assert capped and all(s == pytest.approx(alu.POLL_INTERVAL_MAX) for s in capped)

    @pytest.mark.asyncio
    async def test_null_foreground_keeps_polling_then_succeeds(self, clock):
        results = [None, None, "com.app"]
        with patch.object(
            alu,
            "get_current_foreground_package_async",
            AsyncMock(side_effect=results),
        ):
            confirmed, detail = await alu._poll_for_app_ready(_ctx(), "com.app")

        assert confirmed is True and detail is None
        assert len(clock.sleeps) == 2

    @pytest.mark.asyncio
    async def test_last_sleep_never_overshoots_the_deadline(self, clock):
        """最后一次 sleep 只能睡到 deadline，不能再睡出一个完整间隔。"""
        with patch.object(
            alu, "get_current_foreground_package_async", AsyncMock(return_value=None)
        ):
            await alu._poll_for_app_ready(_ctx(), "com.app", max_poll_seconds=0.5)

        assert clock.sleeps
        assert sum(clock.sleeps) <= 0.5 + 1e-9, f"总睡眠超出预算：{clock.sleeps}"


class TestMismatchIsNotAFailure:
    """🔴 **包名对不上 ≠ 失败** —— 这是 execution 851 的根因，必须守住。

    真实事故：MIUI 上「设置 → 应用设置/应用管理」的宿主是
    ``com.miui.securitycenter``，前台包名是它而不是 ``com.android.settings``。
    老逻辑当场判「启动失败」⇒ 工具报失败 ⇒ 模型以为「还没验成」
    ⇒ 在两个验证条件之间无限轮转（54 次调用、59.6 万 token）。
    """

    @pytest.mark.asyncio
    async def test_wrong_package_keeps_waiting_instead_of_failing(self, clock):
        """前台是别的包 ⇒ **继续等**，不是立刻失败。"""
        with patch.object(
            alu,
            "get_current_foreground_package_async",
            AsyncMock(return_value="com.other"),
        ):
            confirmed, detail = await alu._poll_for_app_ready(
                _ctx(), "com.app", max_poll_seconds=5
            )

        assert confirmed is False
        assert detail is not None
        assert clock.sleeps, "包名对不上时应该继续等，不是当场放弃"

    @pytest.mark.asyncio
    async def test_timeout_detail_says_command_ran_but_not_confirmed(self, clock):
        """等满了要说清：**命令发出去了**，只是没确认切过去。"""
        with patch.object(
            alu,
            "get_current_foreground_package_async",
            AsyncMock(return_value="com.miui.securitycenter"),
        ):
            confirmed, detail = await alu._poll_for_app_ready(
                _ctx(), "com.android.settings", max_poll_seconds=2
            )

        assert confirmed is False
        assert detail is not None
        assert "com.miui.securitycenter" in detail, f"得说清前台是谁：{detail}"
        assert "没确认" in detail or "没能确认" in detail, f"得说清是「没确认」：{detail}"

    @pytest.mark.asyncio
    async def test_timeout_reports_last_foreground(self, clock):
        with patch.object(
            alu, "get_current_foreground_package_async", AsyncMock(return_value=None)
        ):
            confirmed, detail = await alu._poll_for_app_ready(
                _ctx(), "com.app", max_poll_seconds=2
            )

        assert confirmed is False
        assert detail is not None


class TestLaunchSemantics:
    """启动命令本身的成败，和「切没切到前台」是**两件事**。"""

    @pytest.mark.asyncio
    async def test_command_failure_is_retried_and_carries_the_reason(self, clock):
        with patch.object(alu, "UnifiedMobileController") as ctor:
            ctor.return_value.launch_app = AsyncMock(return_value=False)
            ok, detail = await alu.launch_app_with_retries(
                _ctx(), "com.x", max_retries=3
            )

        assert ok is False
        assert detail is not None
        assert "Failed to execute launch command" in detail
        assert ctor.return_value.launch_app.await_count == 3, "命令失败应该重试"

    @pytest.mark.asyncio
    async def test_command_failure_reason_from_exception_reaches_the_caller(self, clock):
        """adb 层抛出来的硬错误，原文要能到模型手里。"""
        with patch.object(alu, "UnifiedMobileController") as ctor:
            ctor.return_value.launch_app = AsyncMock(
                side_effect=RuntimeError("device offline")
            )
            ok, detail = await alu.launch_app_with_retries(_ctx(), "com.x", max_retries=1)

        assert ok is False
        assert detail is not None
        assert "device offline" in detail

    @pytest.mark.asyncio
    async def test_not_confirmed_is_success_with_a_note_and_is_not_retried(self, clock):
        """🔴 **没能确认 ≠ 失败**，而且**不许重发启动命令**。

        命令已经成功发出去了（画面很可能已经在屏上），重发没用。
        """
        note = "启动命令已执行，但 2s 内没能确认 com.android.settings 切到前台（前台现在是 com.miui.securitycenter）。"
        with (
            patch.object(alu, "UnifiedMobileController") as ctor,
            patch.object(alu, "_poll_for_app_ready", AsyncMock(return_value=(False, note))),
        ):
            ctor.return_value.launch_app = AsyncMock(return_value=True)
            ok, detail = await alu.launch_app_with_retries(
                _ctx(), "com.android.settings", max_retries=3
            )

        assert ok is True, "没能确认切到前台，不该报成启动失败"
        assert detail == note
        assert ctor.return_value.launch_app.await_count == 1, "不该重发启动命令"

    @pytest.mark.asyncio
    async def test_confirmed_is_plain_success(self, clock):
        with (
            patch.object(alu, "UnifiedMobileController") as ctor,
            patch.object(alu, "_poll_for_app_ready", AsyncMock(return_value=(True, None))),
        ):
            ctor.return_value.launch_app = AsyncMock(return_value=True)
            ok, detail = await alu.launch_app_with_retries(_ctx(), "com.app")

        assert ok is True
        assert detail is None
