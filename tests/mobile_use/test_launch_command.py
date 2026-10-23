"""`utils/launch_command.py` + 控制器 `launch_app` 的失败判定。

🔴 **2026-10-24：修「启动没成却报成功」。**

adbutils 的 `AdbDevice.app_start` 内部是 monkey，而 **monkey 失败时不抛异常**
（adbutils 的 `shell()` 只返回输出字符串、不检查错误输出）。真机实测（Redmi Turbo 4）::

    monkey -p com.definitely.not.here.xyz -c android.intent.category.LAUNCHER 1
    → ** No activities found to run, monkey aborted.

    monkey -p com.android.settings -c android.intent.category.LAUNCHER 1
    → Events injected: 1

⇒ 直接信 `app_start()` 的返回值，会把「**根本没启动**」报成「启动成功」。
⇒ 现在自己发命令、看输出；失败时**抛异常**，原因原文回灌给模型。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from minitap.mobile_use.controllers.android_controller import AndroidDeviceController
from minitap.mobile_use.utils import launch_command as lc

# 真机上抓到的两段原始输出（adb shell monkey 的真实回显）
MONKEY_OK = """bash arg: -p
bash arg: com.android.settings
bash arg: -c
bash arg: android.intent.category.LAUNCHER
bash arg: 1
args: [-p, com.android.settings, -c, android.intent.category.LAUNCHER, 1]
data="com.android.settings"
data="android.intent.category.LAUNCHER"
Events injected: 1
## Network stats: elapsed time=41ms (0ms mobile, 0ms wifi, 41ms not connected)
"""

MONKEY_ABORTED = """bash arg: -p
bash arg: com.definitely.not.here.xyz
args: [-p, com.definitely.not.here.xyz, -c, android.intent.category.LAUNCHER, 1]
** No activities found to run, monkey aborted.
"""


class TestLaunchFailureReason:
    def test_monkey_abort_is_a_failure(self):
        reason = lc.launch_failure_reason(MONKEY_ABORTED)
        assert reason is not None
        assert "No activities found" in reason

    def test_successful_monkey_is_not_a_failure(self):
        assert lc.launch_failure_reason(MONKEY_OK) is None

    def test_am_start_error_is_a_failure(self):
        reason = lc.launch_failure_reason(
            "Starting: Intent { cmp=com.x/.Foo }\n"
            "Error type 3\nError: Activity class {com.x/.Foo} does not exist."
        )
        assert reason is not None
        assert "does not exist" in reason

    def test_am_start_warning_is_not_a_failure(self):
        """⚠️ app 已经在前台时 `am start` 会打这条 Warning —— 那是**成功**。"""
        assert (
            lc.launch_failure_reason(
                "Warning: Activity not started, intent has been delivered to "
                "currently running top-most instance."
            )
            is None
        )

    def test_bytes_output_is_accepted(self):
        assert lc.launch_failure_reason(MONKEY_ABORTED.encode("utf-8")) is not None

    def test_empty_output_is_not_a_failure(self):
        assert lc.launch_failure_reason("") is None
        assert lc.launch_failure_reason(None) is None


def _controller(shell_result: str) -> AndroidDeviceController:
    """造一个只桩掉 `device.shell` 的控制器。"""
    device = MagicMock()
    device.shell = MagicMock(return_value=shell_result)
    adb_client = MagicMock()
    adb_client.device = MagicMock(return_value=device)
    controller = AndroidDeviceController(
        device_id="test-serial",
        adb_client=adb_client,
        ui_adb_client=MagicMock(),
        device_width=1220,
        device_height=2712,
    )
    return controller


class TestAndroidLaunchApp:
    @pytest.mark.asyncio
    async def test_successful_launch_returns_true(self):
        controller = _controller(MONKEY_OK)
        assert await controller.launch_app("com.android.settings") is True

    @pytest.mark.asyncio
    async def test_aborted_launch_raises_with_the_reason(self):
        """🔴 **核心断言**：没启动成功 ⇒ 抛异常，**不能**返回 True。"""
        controller = _controller(MONKEY_ABORTED)
        with pytest.raises(RuntimeError) as excinfo:
            await controller.launch_app("com.definitely.not.here.xyz")
        assert "No activities found" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_launch_does_not_use_adbutils_app_start(self):
        """不许再用 `device.app_start()` —— 它把 monkey 的输出丢掉了。"""
        controller = _controller(MONKEY_OK)
        with patch.object(controller, "_device") as device:
            device.shell = MagicMock(return_value=MONKEY_OK)
            await controller.launch_app("com.android.settings")
            assert not device.app_start.called, "app_start 会丢掉 monkey 的输出"
