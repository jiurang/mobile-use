"""图片按需取（③）的行为锁定。

三条链路都要锁：

1. **通道三态** —— `True`（每轮带图）/ `False`（不带）/ `"on_demand"`（按需）。
   默认是 `"on_demand"`：截图是最贵、且**永远不可缓存**的那一块。
2. **谁来要图** —— 2026-10-21「Executor 自主执行」改造后，要图的是
   **Executor**（`ExecutorOutput.need_screenshot`），不再是 Cortex。
   Cortex 已经不看屏了（感知全部下沉给 Executor），所以它没有这个字段。
3. **prompt 渲染** —— `executor.md` 必须说清「什么时候要图」和「要了会给图」。

⚠️ 改造前这个文件测的是 cortex 的 `should_request_screenshot`，
那套机制随 Cortex 不再看屏一起删掉了（`cortex.py` 里不再有这个函数）。
截图按需的能力**没有丢**，只是换了主人 —— 现在在 Executor 侧。
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


class TestExecutorOwnsTheScreenshotChannel:
    """要图的能力搬到了 Executor 侧 —— 锁住它没丢。"""

    def test_executor_output_has_need_screenshot(self):
        """Executor 的输出模型必须带 `need_screenshot`。"""
        assert "need_screenshot" in ExecutorOutput.model_fields

    def test_need_screenshot_defaults_to_false(self):
        """默认不要图 —— 图是最贵的一块，不能每轮都带。"""
        assert ExecutorOutput(actions=[]).need_screenshot is False

    def test_need_screenshot_alone_is_a_legal_answer(self):
        """「我要图」是**合法应答**，不是错误。

        2026-10-21 真机踩出来的：模型只置 `need_screenshot`、没给动作时，
        旧 cortex 的空响应校验会直接抛 "empty response" → 整轮失败 →
        agent 在原地白烧步数（实测 51 步没跑完）。

        新架构里 `ExecutorOutput` 是 pydantic 模型，`need_screenshot=True`
        + 空 actions 是**合法构造**，不会抛。
        """
        out = ExecutorOutput(actions=[], need_screenshot=True)
        assert out.need_screenshot is True
        assert out.actions == []

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

    def test_prompt_explains_the_on_demand_mechanism(self):
        """必须说清：什么时候置 need_screenshot、要了会给图。"""
        out = self._render()
        assert "need_screenshot" in out
        assert "asked again" in out or "asked once more" in out or "given one screenshot" in out

    def test_prompt_explains_the_screen_may_be_asleep(self):
        """息屏时截图全黑、元素表只剩残留 —— 必须告诉模型怎么应对。"""
        out = self._render()
        assert "wake_screen" in out
        assert "OFF" in out or "asleep" in out

    def test_prompt_tells_the_model_not_to_ask_every_round(self):
        """「Ask for it only when you really cannot decide」—— 不能每轮都要图。"""
        out = self._render()
        assert "Ask for it only" in out or "only when" in out

    def test_output_table_lists_the_field(self):
        out = self._render()
        assert "need_screenshot" in out
