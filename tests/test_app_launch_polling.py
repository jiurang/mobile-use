"""`utils/app_launch_utils.py` 的轮询语义。

改动的起因（2026-10-17）：原来是「先无脑 `sleep(1)`，再每秒轮询一次」，
每次启动 App 白扔 1~2 秒。改成「立刻查一次，查不到就短间隔退避重试」。

这里守三件事：
1. 第一次查询**之前**不能有任何 sleep —— 否则前面那 1 秒白等又被请回来。
2. 退避要一路涨到 POLL_INTERVAL_MAX 就停住，不能无限涨（否则后半程几乎不查了）。
3. 语义不能变：拿到别的包名要立刻判失败，mCurrentFocus=null 才继续等。
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
            ok, err = await alu._poll_for_app_ready(_ctx(), "com.app")

        assert ok is True
        assert err is None
        assert clock.sleeps == [], f"命中即退还要 sleep？{clock.sleeps}"

    @pytest.mark.asyncio
    async def test_backoff_grows_then_caps(self, clock):
        """前几次密、后面退到 1s 上限，绝不无限增长。"""
        with patch.object(
            alu,
            "get_current_foreground_package_async",
            AsyncMock(return_value=None),  # 永远 mCurrentFocus=null
        ):
            ok, err = await alu._poll_for_app_ready(_ctx(), "com.app", max_poll_seconds=30)

        assert ok is False and err is not None
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
            ok, err = await alu._poll_for_app_ready(_ctx(), "com.app")

        assert ok is True and err is None
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


class TestSemanticsUnchanged:
    @pytest.mark.asyncio
    async def test_wrong_package_fails_immediately(self, clock):
        """前台是别的包 —— 立刻失败，不能继续等满 15 秒。"""
        with patch.object(
            alu,
            "get_current_foreground_package_async",
            AsyncMock(return_value="com.other"),
        ):
            ok, err = await alu._poll_for_app_ready(_ctx(), "com.app")

        assert ok is False
        assert "com.other" in err
        assert clock.sleeps == [], "已经确定失败就不该再等"

    @pytest.mark.asyncio
    async def test_timeout_reports_last_foreground(self, clock):
        with patch.object(
            alu, "get_current_foreground_package_async", AsyncMock(return_value=None)
        ):
            ok, err = await alu._poll_for_app_ready(_ctx(), "com.app", max_poll_seconds=2)

        assert ok is False
        assert "Timeout" in err
