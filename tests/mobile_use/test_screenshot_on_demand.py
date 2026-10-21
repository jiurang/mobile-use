"""图片按需取（③）的行为锁定。

三条链路都要锁：

1. **通道三态** —— `True`（每轮带图）/ `False`（不带）/ `"on_demand"`（按需）。
   默认是 `"on_demand"`：截图是最贵、且**永远不可缓存**的那一块。
2. **要不要补图** —— 纯函数 `should_request_screenshot`。
3. **prompt 渲染** —— 三种模式下给模型看的「当前通道」文案必须对；
   `"on_demand"` 不能显示成 `enabled`（模型会以为图已经在手里）。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import Environment, StrictUndefined

from minitap.mobile_use.agents.cortex.cortex import should_request_screenshot
from minitap.mobile_use.agents.cortex.types import CortexOutput
from minitap.mobile_use.context import PerceptionConfig

CORTEX_MD = (
    Path(__file__).resolve().parents[2]
    / "minitap"
    / "mobile_use"
    / "agents"
    / "cortex"
    / "cortex.md"
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


class TestShouldRequestScreenshot:
    def _resp(self, *, need: bool = True, decisions: str | None = None) -> CortexOutput:
        return CortexOutput(need_screenshot=need, decisions=decisions)

    def test_asks_again_when_model_cannot_decide(self):
        assert should_request_screenshot(self._resp(), on_demand=True, has_screenshot=True)

    def test_no_ask_when_model_already_decided(self):
        """模型给出了动作就别补图 —— 补了只会打乱它的判断。"""
        assert not should_request_screenshot(
            self._resp(decisions='[{"action":"tap"}]'), on_demand=True, has_screenshot=True
        )

    def test_no_ask_when_model_did_not_request(self):
        assert not should_request_screenshot(
            self._resp(need=False), on_demand=True, has_screenshot=True
        )

    def test_no_ask_in_always_screenshot_mode(self):
        """老模式每轮都带图，没有「再问一次」这回事。"""
        assert not should_request_screenshot(self._resp(), on_demand=False, has_screenshot=True)

    def test_no_ask_without_a_screenshot(self):
        """contextor 没抓到图，补不了。"""
        assert not should_request_screenshot(self._resp(), on_demand=True, has_screenshot=False)


class TestEmptyResponseIsNotAnErrorWhenAskingForScreenshot:
    """「我要图」是**合法应答**，不是空响应错误。

    这条是 2026-10-21 真机踩出来的：模型只置 `need_screenshot`、没给决策也没给
    理由时，`_invoke_cortex` 的空响应校验会直接抛 "Cortex LLM returned an empty
    response" → 整轮失败 → agent 在原地白烧步数（实测 51 步没跑完）。
    """

    def test_need_screenshot_alone_is_not_empty(self):
        from minitap.mobile_use.agents.cortex.cortex import _invoke_cortex

        class _Ctx:
            trace_id = "t1"

        class _Resp:
            decisions = None
            decisions_reason = None
            goals_completion_reason = None
            complete_subgoals_by_ids = []
            need_screenshot = True

        class _Llm:
            async def ainvoke(self, messages, **kwargs):
                return _Resp()

        import asyncio

        result = asyncio.run(
            _invoke_cortex(_Llm(), [], _Ctx(), set())  # type: ignore[arg-type]
        )
        assert result.need_screenshot is True

    def test_truly_empty_response_still_raises(self):
        """真·空响应（连图都不要）仍然要报错。"""
        from minitap.mobile_use.agents.cortex.cortex import _invoke_cortex

        class _Ctx:
            trace_id = "t1"

        class _Resp:
            decisions = None
            decisions_reason = None
            goals_completion_reason = None
            complete_subgoals_by_ids = []
            need_screenshot = False

        class _Llm:
            async def ainvoke(self, messages, **kwargs):
                return _Resp()

        import asyncio
        import pytest as _pytest

        with _pytest.raises(ValueError, match="empty response"):
            asyncio.run(
                _invoke_cortex(_Llm(), [], _Ctx(), set())  # type: ignore[arg-type]
            )


class TestPromptRendering:
    def _render(self, perception) -> str:
        env = Environment(undefined=StrictUndefined, autoescape=False)
        return env.from_string(CORTEX_MD.read_text(encoding="utf-8")).render(
            platform="android",
            executor_tools_list="tap",
            locked_app_package=None,
            perception=perception,
        )

    def test_on_demand_is_labelled_correctly(self):
        out = self._render(PerceptionConfig(screenshot="on_demand"))
        assert "Screenshot: on demand" in out
        # 绝不能显示成 enabled —— 模型会以为图已经在手里
        assert "Screenshot: enabled" not in out

    def test_enabled_and_disabled_labels(self):
        assert "Screenshot: enabled" in self._render(PerceptionConfig(screenshot=True))
        assert "Screenshot: DISABLED" in self._render(PerceptionConfig(screenshot=False))

    def test_on_demand_explains_the_mechanism(self):
        out = self._render(PerceptionConfig(screenshot="on_demand"))
        assert "Need Screenshot" in out
        assert "on demand" in out

    def test_output_table_lists_the_new_field(self):
        out = self._render(PerceptionConfig(screenshot="on_demand"))
        assert "**Need Screenshot**" in out

    def test_always_mode_does_not_mention_on_demand_section(self):
        out = self._render(PerceptionConfig(screenshot=True))
        assert "You will then be given\none screenshot" not in out
