"""图片通道：**Executor 自己决定要不要看图**（2026-10-22 改）。

🔴 **2026-10-22 语义整个换掉了。**

* 改造前：Executor 整条通道**没有图**（`need_screenshot` 被删，提示词写着
  "There is no screenshot"）。理由是原来的「模型要图 → 同轮补一张 →
  **再问一次 LLM**」要烧整整一次 LLM 调用（真机实测 executor 单次决策中位
  5,276ms），收益只有「模型偶尔看见颜色/图标/遮挡」。
* 现在：`ExecutorOutput.need_screenshot` **回来了**，但用法不一样 ——
  **Executor 自己举手**，框架在**下一轮**把当屏截图挂进 prompt。
  **零额外 LLM 调用**（循环本来就要继续下一轮），而且图是 contextor 刚抓的
  那一帧 ⇒ 图和元素表同屏，`[i]` 编号契约不受影响。

🔴 **2026-10-22 再改：三态显式枚举 + 「关掉就一个字都不提」。**

* `PerceptionConfig.screenshot` 从混型 `bool | "on_demand"` 改成字符串枚举
  `"always" / "on_demand" / "off"`（老 bool 在校验器里被收：`True` →
  `"on_demand"`、`False` → `"off"`）。
* **模型不能看图时，提示词里一个图相关的词都不许有，schema 里也不留
  `need_screenshot` 字段** —— 模型看不到的东西，不该让它以为有。

这个文件锁住四件事：

1. **通道三态** —— 显式枚举，归一化在字段校验里做（不再有「`"on_demand"` 是
   真值」的真假判断陷阱）。
2. **决策字段在** —— 能看图时 `ExecutorOutput.need_screenshot` 存在；
   **不能看图时连字段都没有**。
3. **提示词按开关渲染** —— 能看图才写图怎么用；不能看图时**零图词**。
4. **OCR / UI 开关也生效** —— 关掉的通道不出现（不是「关了还偷偷给」）。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import Environment, StrictUndefined

from minitap.mobile_use.agents.executor.types import ExecutorOutput, executor_output_schema
from minitap.mobile_use.context import PerceptionConfig

EXECUTOR_MD = (
    Path(__file__).resolve().parents[2]
    / "minitap"
    / "mobile_use"
    / "agents"
    / "executor"
    / "executor.md"
)

#: 「跟图有关的词」—— 通道关掉时，渲染出来的 prompt 里一个都不该有。
IMAGE_WORDS = (
    "screenshot",
    "picture",
    "image",
    "need_screenshot",
    "color",
    "icon",
)


class TestScreenshotMode:
    def test_default_is_on_demand(self):
        """默认按需 —— **Executor 自己决定**要不要看图。"""
        assert PerceptionConfig().screenshot == "on_demand"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("always", "always"),
            ("on_demand", "on_demand"),
            ("off", "off"),
        ],
    )
    def test_three_states(self, value, expected):
        assert PerceptionConfig(screenshot=value).screenshot == expected

    def test_legacy_bool_true_means_executor_decides(self):
        """老代码里 ``True`` 是「每轮都带图」（Cortex 时代的老行为）。

        Manager-web 的开关是「**能不能看图**」⇒ ``True`` 现在映射成
        ``"on_demand"``（能看图，**由 Executor 自己决定**什么时候看）。
        """
        assert PerceptionConfig(screenshot=True).screenshot == "on_demand"

    def test_legacy_bool_false_means_off(self):
        assert PerceptionConfig(screenshot=False).screenshot == "off"

    def test_can_see_screenshot(self):
        assert PerceptionConfig(screenshot="always").can_see_screenshot is True
        assert PerceptionConfig(screenshot="on_demand").can_see_screenshot is True
        assert PerceptionConfig(screenshot="off").can_see_screenshot is False

    def test_other_channels_untouched(self):
        p = PerceptionConfig(screenshot="on_demand")
        assert p.ui_hierarchy and p.ocr

    def test_screenshot_channel_is_the_normalized_value(self):
        """归一化在字段校验里做完了 —— 属性直接返回，不会各判各的。"""
        assert PerceptionConfig(screenshot=True).screenshot_channel == "on_demand"
        assert PerceptionConfig(screenshot=False).screenshot_channel == "off"


class TestSchemaFollowsTheSwitch:
    """不能看图 ⇒ schema 里连字段都没有。"""

    def test_schema_has_the_field_when_the_model_can_see(self):
        schema = executor_output_schema(can_see_screenshot=True)
        assert "need_screenshot" in schema.model_fields

    def test_schema_drops_the_field_when_the_model_cannot_see(self):
        schema = executor_output_schema(can_see_screenshot=False)
        assert "need_screenshot" not in schema.model_fields
        # 其它字段一个都不能少
        for field in ("actions", "agent_thought", "done", "done_reason"):
            assert field in schema.model_fields

    def test_tool_name_stays_executor_output(self):
        """工具名不能变 —— 模型被教的是「你只能调 ExecutorOutput」，
        解析层也拿这个名字对账（喊错名字会被拒）。"""
        from langchain_core.utils.function_calling import convert_to_openai_tool

        for can_see in (True, False):
            schema = executor_output_schema(can_see_screenshot=can_see)
            assert convert_to_openai_tool(schema)["function"]["name"] == "ExecutorOutput"


class TestExecutorDecidesTheScreenshot:
    """Executor 自己举手要图。"""

    def test_need_screenshot_exists(self):
        assert "need_screenshot" in ExecutorOutput.model_fields

    def test_need_screenshot_defaults_to_false(self):
        """默认不举手 —— 绝大多数轮次靠元素表 + OCR 就够。"""
        assert ExecutorOutput(actions=[]).need_screenshot is False

    def test_model_can_raise_the_hand(self):
        out = ExecutorOutput(actions=[], need_screenshot=True)
        assert out.need_screenshot is True

    def test_cortex_still_does_not_own_the_channel(self):
        """Cortex 不看屏，截图通道归 Executor。"""
        from minitap.mobile_use.agents.cortex.types import CortexOutput

        assert "need_screenshot" not in CortexOutput.model_fields
        assert "decisions" not in CortexOutput.model_fields


class TestPromptRendering:
    def _render(self, perception: PerceptionConfig | None = None) -> str:
        env = Environment(undefined=StrictUndefined, autoescape=False)
        return env.from_string(EXECUTOR_MD.read_text(encoding="utf-8")).render(
            platform="android",
            executor_tools_list="tap",
            goal="读出当前连接的 Wi-Fi 名称",
            perception=perception or PerceptionConfig(),
        )

    def test_on_demand_tells_the_model_it_has_to_ask(self):
        """默认档：模型必须知道「图不会自己来，要举手」。"""
        out = self._render(PerceptionConfig(screenshot="on_demand"))
        assert "you have to ask for it" in out
        assert "need_screenshot" in out

    def test_always_says_the_picture_is_attached_every_turn(self):
        out = self._render(PerceptionConfig(screenshot="always"))
        assert "attached to **every**" in out

    def test_off_has_no_image_words_at_all(self):
        """🔴 **模型不能看图 ⇒ 提示词里一个图相关的词都不许有。**

        留一个词，模型就会以为自己能看到：要么在 `agent_thought` 里说
        「我看到有个红色的按钮」（它根本看不到），要么每轮举手要一张
        永远不会来的图。
        """
        out = self._render(PerceptionConfig(screenshot="off")).lower()
        for word in IMAGE_WORDS:
            assert word not in out, f"关掉图片通道后，提示词里不该出现「{word}」"

    def test_off_does_not_teach_the_field(self):
        out = self._render(PerceptionConfig(screenshot="off"))
        assert "`need_screenshot`" not in out

    def test_ui_off_is_declared(self):
        """UI 通道关掉 ⇒ 提示词要说清「没有元素表、只能按 ocr_index 寻址」。"""
        out = self._render(PerceptionConfig(ui_hierarchy=False))
        assert "UI elements are OFF" in out
        assert "ocr_index" in out

    def test_ocr_off_is_declared(self):
        out = self._render(PerceptionConfig(ocr=False))
        assert "OCR is OFF" in out

    def test_keeps_the_two_text_channels(self):
        """默认档：UI 元素 + OCR 这两个通道必须留着。"""
        out = self._render()
        assert "UI elements" in out
        assert "OCR" in out

    def test_prompt_explains_the_screen_may_be_asleep(self):
        """息屏时元素表只剩残留 —— 必须告诉模型怎么应对。"""
        out = self._render()
        assert "wake_screen" in out
        assert "asleep" in out
