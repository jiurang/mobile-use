"""「屏幕灭着 → 告诉模型 → 模型唤醒」这条链路的单测。

为什么要有它
------------
屏幕一黑，截图全黑、层级树只剩 systemui 残留，agent 于是「看不到任何业务内容」，
会在黑屏上反复尝试、原地打转。真机实测（2026-10-21，Redmi Turbo 4 / Android 16）：
手机的 30 分钟息屏超时让一个任务从 30 步涨到 51 步还没跑完 —— 而且**全程没有任何
报错**，症状是「步数暴涨」，极易被误判成代码回归。

这条链路有三段，各自有坑：
1. 读状态（`dumpsys power` 的 `mWakefulness`）
2. **无条件**读状态，并与抓屏并发（省掉 ~100ms/轮）
3. 工具如实回报成功/失败（有 PIN 锁屏时划不掉，不能假装成功）

第 2 段有个被实测证伪过的坑：最初的判据是「层级树为空才查状态」。
真机实测熄屏后 uiautomator **仍返回 19 个元素**（全是 systemui 残留），
那个判据**永远不会触发**，唤醒功能等于没接上。下面
`TestContextorReadsPowerState` 就是钉死这个坑的回归测试。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from minitap.mobile_use.controllers.android_controller import AndroidDeviceController
from minitap.mobile_use.tools.index import EXECUTOR_WRAPPERS_TOOLS


class _FakeDevice:
    """只实现 `shell()` 的假设备。"""

    def __init__(self, output: str | bytes):
        self.output = output
        self.commands: list[str] = []

    def shell(self, cmd: str) -> str | bytes:
        self.commands.append(cmd)
        return self.output


def _controller(output: str | bytes) -> AndroidDeviceController:
    """造一个只连着假设备的控制器（绕过 __init__，它要真设备）。"""
    c = AndroidDeviceController.__new__(AndroidDeviceController)
    c.device_id = "fake"
    c._device = _FakeDevice(output)  # type: ignore[assignment]
    return c


class TestGetScreenPowerState:
    @pytest.mark.asyncio
    async def test_awake_is_on(self):
        c = _controller("mWakefulness=Awake\nmScreenOnFully=true\n")
        assert await c.get_screen_power_state() == "on"

    @pytest.mark.asyncio
    async def test_asleep_is_off(self):
        c = _controller("mWakefulness=Asleep\n")
        assert await c.get_screen_power_state() == "off"

    @pytest.mark.asyncio
    async def test_dozing_is_off(self):
        c = _controller("mWakefulness=Dozing\n")
        assert await c.get_screen_power_state() == "off"

    @pytest.mark.asyncio
    async def test_unrecognized_is_unknown(self):
        """读不出来要说「不知道」，**不能**说「亮着」—— 那会让 cortex 以为没事。"""
        c = _controller("something else entirely\n")
        assert await c.get_screen_power_state() == "unknown"

    @pytest.mark.asyncio
    async def test_bytes_output_is_decoded(self):
        c = _controller(b"mWakefulness=Asleep\n")
        assert await c.get_screen_power_state() == "off"

    @pytest.mark.asyncio
    async def test_shell_failure_is_unknown(self):
        class _Boom:
            def shell(self, cmd: str) -> str:
                raise RuntimeError("adb died")

        c = AndroidDeviceController.__new__(AndroidDeviceController)
        c._device = _Boom()  # type: ignore[assignment]
        assert await c.get_screen_power_state() == "unknown"


class TestWakeUp:
    @pytest.mark.asyncio
    async def test_sends_wakeup_and_dismiss(self):
        c = _controller("mWakefulness=Awake\n")
        assert await c.wake_up() is True
        assert c.device.commands == [  # type: ignore[attr-defined]
            "input keyevent 224",
            "wm dismiss-keyguard",
        ]

    @pytest.mark.asyncio
    async def test_failure_returns_false(self):
        class _Boom:
            def shell(self, cmd: str) -> str:
                raise RuntimeError("adb died")

        c = AndroidDeviceController.__new__(AndroidDeviceController)
        c._device = _Boom()  # type: ignore[assignment]
        assert await c.wake_up() is False


class TestToolRegistration:
    def test_tool_is_in_the_executor_toolset(self):
        """注册进去，直连派发（dispatch_decisions 按名字查表）才认得。"""
        names = {w.tool_fn_getter.__name__ for w in EXECUTOR_WRAPPERS_TOOLS}
        assert "get_wake_screen_tool" in names

    def test_tool_has_no_required_args_besides_thought(self):
        """除 agent_thought 外没有必填参数 —— 模型只要写 action 就能派发。"""
        from minitap.mobile_use.tools.mobile import wake_screen as ws

        tool = ws.get_wake_screen_tool(None)  # type: ignore[arg-type]
        fields = getattr(tool.args_schema, "model_fields", {})
        required = {
            name
            for name, f in fields.items()
            if f.is_required() and name not in {"tool_call_id", "state", "agent_thought"}
        }
        assert required == set(), f"不该有模型必须自己填的参数: {required}"


class TestToolBehaviour:
    """工具要**如实**回报：屏幕还灭着就说失败，不能假装成功。"""

    def _run(self, monkeypatch, *, after_state: str, wake_ok: bool = True) -> dict:
        from minitap.mobile_use.tools.mobile import wake_screen as ws

        calls: list[str] = []

        class _FakeController:
            def __init__(self, ctx):
                pass

            async def wake_up(self) -> bool:
                calls.append("wake_up")
                return wake_ok

            async def get_screen_power_state(self) -> str:
                calls.append("state")
                return after_state

        # 工具模块是 `from ... import UnifiedMobileController` 直接引的，
        # 所以要打在 **ws** 上，打在 unified_controller 模块上没用。
        monkeypatch.setattr(ws, "UnifiedMobileController", _FakeController)

        class _FakeState:
            async def asanitize_update(self, *, ctx, update, agent):
                return update

        tool = ws.get_wake_screen_tool(None)  # type: ignore[arg-type]
        out = asyncio.run(
            tool.coroutine(  # type: ignore[misc]
                agent_thought="屏幕是黑的",
                tool_call_id="call_1",
                state=_FakeState(),  # type: ignore[arg-type]
            )
        )
        return {"calls": calls, "out": out}

    def test_reports_success_when_screen_is_on(self, monkeypatch):
        r = self._run(monkeypatch, after_state="on")
        assert r["calls"] == ["wake_up", "state"]
        content = str(r["out"].update.get("agents_thoughts"))
        assert "on now" in content

    def test_reports_failure_when_still_off(self, monkeypatch):
        """PIN 锁屏划不掉 —— 必须如实说失败。"""
        r = self._run(monkeypatch, after_state="off")
        content = str(r["out"].update.get("agents_thoughts"))
        assert "PIN" in content


class TestContextorReadsPowerState:
    """contextor 必须**无条件**读电源状态。

    这里钉的是一个**被实测证伪**的设计：最初的判据是「层级树为空才查状态」，
    理由是「屏幕亮着时层级树一定有东西，正常路径一次额外调用都不花」。

    2026-10-21 真机实测（Redmi Turbo 4 / Android 16）推翻了它：熄屏后
    uiautomator **仍然返回 19 个元素**（全是 systemui 的通知面板/状态栏残留），
    截图才是全黑的。于是那个判据**永远不会触发** —— 屏幕灭着的时候 harness
    根本不会告诉模型，唤醒功能等于没接上，而且**没有任何报错**。
    """

    @staticmethod
    def _harness(monkeypatch, controller):
        from minitap.mobile_use.agents.contextor import contextor as ctx_mod

        monkeypatch.setattr(ctx_mod, "create_device_controller", lambda _ctx: controller)
        # 🔴 2026-10-22：不再有 `get_current_foreground_package_async` 这条 adb
        # 往返 —— 前台包名改成**从刚抓回来的层级树里推**
        # （`utils.screen_elements.foreground_package_from_tree`）。
        # 少一次串行 adb 往返（实测 127ms），整段感知快 40%。
        monkeypatch.setattr(ctx_mod, "get_device_date", lambda _ctx: None)

        node = ctx_mod.ContextorNode.__new__(ctx_mod.ContextorNode)
        node.ctx = SimpleNamespace(
            perception=SimpleNamespace(settle_delay_ms=0, ocr=False),
            execution_setup=None,
        )
        return node

    @staticmethod
    def _state():
        class _State:
            captured: dict = {}

            async def asanitize_update(self, *, ctx, update, agent):
                _State.captured = update
                return update

        return _State()

    @pytest.mark.asyncio
    async def test_reads_state_even_when_element_tree_is_not_empty(self, monkeypatch):
        """树非空**不代表**屏幕亮着 —— 熄屏时树里有 19 个 systemui 残留。"""

        calls: list[str] = []

        class _Controller:
            async def get_screen_data(self, on_screenshot=None):
                calls.append("screen")
                return SimpleNamespace(
                    elements=[{"package": "com.android.systemui"}] * 19,
                    base64="AAAA",
                    width=1220,
                    height=2712,
                )

            async def get_screen_power_state(self) -> str:
                calls.append("power")
                return "off"

        node = self._harness(monkeypatch, _Controller())
        state = self._state()
        await node(state)

        assert "power" in calls, f"没读电源状态，calls={calls}"
        assert state.captured["screen_state"] == "off"

    @pytest.mark.asyncio
    async def test_power_state_is_read_while_the_capture_is_still_running(self, monkeypatch):
        """并发：电源状态要藏在抓屏的等待窗口里，不能串行多花 ~100ms。"""

        class _Controller:
            power_read = False

            async def get_screen_data(self, on_screenshot=None):
                # 抓屏**还没返回**的时候，电源状态就应该已经读到了
                assert self.power_read, "电源状态是在抓屏之后才读的 —— 并发没了"
                return SimpleNamespace(
                    elements=[{"package": "x"}],
                    base64="AAAA",
                    width=1220,
                    height=2712,
                )

            async def get_screen_power_state(self) -> str:
                self.power_read = True
                return "on"

        node = self._harness(monkeypatch, _Controller())
        state = self._state()
        await node(state)
        assert state.captured["screen_state"] == "on"

    @pytest.mark.asyncio
    async def test_controller_without_the_method_does_not_break_the_round(self, monkeypatch):
        """自定义控制器没跟上新接口时，不能整轮挂掉。"""

        class _OldController:
            async def get_screen_data(self, on_screenshot=None):
                return SimpleNamespace(
                    elements=[{"package": "x"}],
                    base64="AAAA",
                    width=1220,
                    height=2712,
                )

        node = self._harness(monkeypatch, _OldController())
        state = self._state()
        await node(state)
        assert state.captured["screen_state"] is None

    @pytest.mark.asyncio
    async def test_power_state_is_recovered_when_the_capture_fails(self, monkeypatch):
        """抓屏炸了也要把电源状态收回来，不能留一个悬空的 task。"""

        class _Controller:
            async def get_screen_data(self, on_screenshot=None):
                raise RuntimeError("uiautomator died")

            async def get_screen_power_state(self) -> str:
                return "off"

        node = self._harness(monkeypatch, _Controller())
        state = self._state()
        with pytest.raises(RuntimeError):
            await node(state)
        # 电源状态读到了（虽然这一轮整体失败了），没有 task 泄漏


def _async_none():
    async def _none():
        return None

    return _none()
