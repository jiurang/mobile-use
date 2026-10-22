"""图片按需取（③）的行为锁定。

🔴 **2026-10-22 变更：Executor 的截图通道整条关掉了。**

原来 Executor 能说「我需要看图」→ 同轮补一张 → **再问一次 LLM**。
实测那次二问就是整整一次 LLM 调用（executor 单次决策中位 5,276ms），
换来的只是「模型偶尔能看见颜色/图标/遮挡」。现在 Executor 只靠
**UI 元素表 + OCR** 决策。

这个文件锁的是三件事：

1. **通道三态** —— `True`（每轮带图）/ `False`（不带）/ `"on_demand"`（按需）。
   配置项本身还在（`PerceptionConfig.screenshot`），但 Executor 不用它了。
2. **截图通道已关** —— `ExecutorOutput` 没有 `need_screenshot`；模型硬吐这个
   键也被静默忽略。
3. **prompt 渲染** —— `executor.md` 必须说清「**没有截图**」+ 息屏唤醒。

⚠️ 改造前这个文件测的是 cortex 的 `should_request_screenshot`，
那套机制随 Cortex 不再看屏一起删掉了（`cortex.py` 里不再有这个函数）。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import Environment, StrictUndefined

from minitap.mobile_use.agents.executor.types import ExecutorOutput
from minitap.mobile_use.context import PerceptionConfig

EXECUTOR_MD = (
    Path(__file__).resolve().parents[2]
    / "minitap"
    / "mobile_use"
    / "agents"
    / "executor"
    / "executor.md"
)


class TestScreenshotMode:
    def test_default_is_on_demand(self):
        """默认按需 —— 截图是最贵且永不缓存的一块。"""
        assert PerceptionConfig().screenshot == "on_demand"

    def test_bool_values_stay_bool(self):
        """别让 pydantic 把 True/False 变成字符串。"""
        assert PerceptionConfig(screenshot=True).screenshot is True
        assert PerceptionConfig(screenshot=False).screenshot is False

    def test_on_demand_stays_string(self):
        assert PerceptionConfig(screenshot="on_demand").screenshot == "on_demand"

    @pytest.mark.parametrize(
        ("value", "label"),
        [
            (True, "enabled"),
            (False, "DISABLED"),
            ("on_demand", "on demand"),
        ],
    )
    def test_label(self, value, label):
        assert PerceptionConfig(screenshot=value).screenshot_label == label

    def test_other_channels_untouched(self):
        p = PerceptionConfig(screenshot="on_demand")
        assert p.ui_hierarchy and p.ocr


class TestExecutorScreenshotChannelIsOff:
    """🔴 2026-10-22：Executor 的截图通道**整条关掉**。

    原来是「要图 → 同轮补一张 → **再问一次 LLM**」。实测那次二问就是
    整整一次 LLM 调用（executor 单次决策中位 5,276ms），换来的只是
    「模型偶尔能看见颜色/图标/遮挡」。

    关掉之后：
    * `ExecutorOutput` 没有 `need_screenshot` ⇒ 模型不会再提这件事；
    * 模型仍然吐这个键 ⇒ pydantic 静默忽略 ⇒ 不报错、不触发额外动作。
    """

    def test_executor_output_has_no_need_screenshot(self):
        """Executor 的输出模型**不带** `need_screenshot`。"""
        assert "need_screenshot" not in ExecutorOutput.model_fields

    def test_stray_need_screenshot_is_ignored_not_fatal(self):
        """模型要是仍然吐这个键，必须**静默忽略**，不能让整轮挂掉。

        2026-10-21 真机踩出来的教训反过来用在这里：解析层的任何一次
        硬失败都会让 agent 在原地白烧步数（实测 51 步没跑完）。
        """
        out = ExecutorOutput(actions=[], need_screenshot=True)  # type: ignore[call-arg]
        assert out.actions == []
        assert not hasattr(out, "need_screenshot")

    def test_cortex_no_longer_owns_the_screenshot_channel(self):
        """Cortex 已经不看屏了，不该再有 `need_screenshot` 字段。"""
        from minitap.mobile_use.agents.cortex.types import CortexOutput

        assert "need_screenshot" not in CortexOutput.model_fields
        assert "decisions" not in CortexOutput.model_fields


class TestPromptRendering:
    def _render(self) -> str:
        env = Environment(undefined=StrictUndefined, autoescape=False)
        return env.from_string(EXECUTOR_MD.read_text(encoding="utf-8")).render(
            platform="android",
            executor_tools_list="tap",
            goal="读出当前连接的 Wi-Fi 名称",
        )

    def test_prompt_says_there_is_no_screenshot(self):
        """必须说清「**没有截图**」—— 否则模型会以为图在手里。"""
        out = self._render()
        assert "There is no screenshot" in out
        assert "need_screenshot" not in out

    def test_prompt_keeps_the_two_remaining_channels(self):
        """UI 元素 + OCR 这两个通道必须留着。"""
        out = self._render()
        assert "UI elements" in out
        assert "OCR" in out

    def test_prompt_explains_the_screen_may_be_asleep(self):
        """息屏时截图全黑、元素表只剩残留 —— 必须告诉模型怎么应对。"""
        out = self._render()
        assert "wake_screen" in out
        assert "OFF" in out or "asleep" in out

    def test_output_table_does_not_list_the_removed_field(self):
        out = self._render()
        assert "need_screenshot" not in out
