"""把「提示词模板能渲染」变成 pytest 门禁，而不是一个要人记得手跑的脚本。

实测事故：cortex.md 末尾多写了 `{% if locked_app_package %}` 却漏掉 `{% endif %}`，
真机上跑完整条任务才抛 TemplateSyntaxError —— agent 已经启动、已经在点屏幕，
最后一步渲染 system prompt 时崩。这个测试就是为了让它发生在 0 秒、0 成本的时候。

真正的检查逻辑在 scripts/check_prompt_templates.py（可单独当 CLI 跑），
这里只做 import + 断言，让 `pytest` 顺带把它跑掉。

Executor / Reporter 的提示词契约（方案 §3.1、§7.1、§7.2）也钉在这里：
模板变量集合、元素索引契约、输出字段表、以及「不出现工具 JSON schema」这条
硬约束 —— Executor 一旦 bind_tools，每轮就多背 4,008 token 且不进缓存键。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from jinja2 import Environment, StrictUndefined

ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = ROOT / "scripts" / "check_prompt_templates.py"
_AGENTS_DIR = ROOT / "minitap" / "mobile_use" / "agents"
_EXECUTOR_MD = _AGENTS_DIR / "executor" / "executor.md"
_REPORTER_MD = _AGENTS_DIR / "reporter" / "reporter.md"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_prompt_templates", _SCRIPT)
    assert spec and spec.loader, f"无法加载 {_SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render(template_path: Path, variables: dict[str, object]) -> str:
    env = Environment(undefined=StrictUndefined, autoescape=False)
    return env.from_string(template_path.read_text(encoding="utf-8")).render(**variables)


def _perception(
    ui_hierarchy: bool = True,
    ocr: bool = True,
    screenshot: bool | str = "on_demand",
):
    """感知开关的替身（和 `scripts/check_prompt_templates.py` 里的是同一个类）。"""
    return _load()._Perception(ui_hierarchy, ocr, screenshot)


def test_prompt_templates_render():
    """每个 agent 提示词模板在生产变量下都能编译并渲染出结果。"""
    checked, failures = _load().check()
    assert not failures, "以下提示词模板渲染失败：\n" + "\n".join(failures)
    assert checked > 0, "一个模板都没检查到，守卫本身失效了"


def test_every_agent_prompt_is_registered():
    """守卫自身的覆盖度：磁盘上的 .md 都必须在变量表里登记。

    新增一个 .md 却忘了登记 → 「全过」是假的，这条挡的就是这种情况。
    """
    template_vars = _load().TEMPLATE_VARS
    on_disk = {p.relative_to(_AGENTS_DIR).as_posix() for p in _AGENTS_DIR.rglob("*.md")}
    assert on_disk - set(template_vars) == set(), "有 .md 没登记渲染变量，无法校验"


def test_executor_template_declares_exactly_four_vars():
    """executor.md 的模板变量恰好四个：platform / executor_tools_list / goal / perception。

    多一个 → 渲染端要多传一个（调用点会对不上）；少一个 → 模板引用了没传的变量。

    `perception` 是**感知通道开关状态**（Manager 侧）：三条通道开着哪几条，
    提示词里必须写清楚 —— 否则模型会一直要一张不会来的图，或者按一个
    被关掉的通道去寻址。一次运行里它是恒定的 ⇒ 放 system prompt 不伤缓存。
    """
    template_vars = _load().TEMPLATE_VARS
    assert set(template_vars["executor/executor.md"]) == {
        "platform",
        "executor_tools_list",
        "goal",
        "perception",
    }
    # goal 必须登记成「取值可能为空」的变量，否则 {% if goal %} 的 else 分支
    # 永远编不到 —— 写坏了也能「全过」。
    assert "goal" in _load().OPTIONAL_VARS


def test_executor_prompt_renders_without_leftover_placeholders():
    """渲染后不能残留 Jinja 占位符 —— 残留在 prompt 里等于给了模型一句废话。"""
    rendered = _render(
        _EXECUTOR_MD,
        {
            "platform": "android",
            "executor_tools_list": "tap, back",
            "goal": "打开快手赚钱",
            "perception": _perception(),
        },
    )
    assert "{{" not in rendered and "{%" not in rendered
    assert "打开快手赚钱" in rendered
    assert "Your goal:" in rendered


def test_executor_prompt_survives_an_empty_goal():
    """goal 传空串时 prompt 仍然完整。

    目标有两种放法：渲染进 system prompt（契约 W1.1），或者留空、由调用方
    放在紧随其后的消息块里（Executor 节点的选择 —— 目标每子目标变，进
    system prompt 会让整块缓存每换一次子目标就重算）。两种都得能用。
    """
    rendered = _render(
        _EXECUTOR_MD,
        {
            "platform": "android",
            "executor_tools_list": "tap, back",
            "goal": "",
            "perception": _perception(),
        },
    )
    assert "{{" not in rendered and "{%" not in rendered
    # 不能留下一句空的 "Your goal:" 让模型以为渲染坏了
    assert "Your goal: \n" not in rendered
    assert "Your goal:" in rendered


def test_executor_prompt_is_an_independent_decider():
    """旧版「Cortex 说什么我做什么」的转写器规则必须已经删掉。

    新架构里 Executor 是唯一的决策者（方案 §2.1），留着那条规则会让它
    等着别人给指令，而没人会给。
    """
    text = _EXECUTOR_MD.read_text(encoding="utf-8")
    assert "Don't reason about strategy" not in text
    assert "just execute what Cortex decided" not in text
    assert "Cortex" not in text
    # 自己决定：目标和循环都在自己手里
    assert "exactly **one** goal" in text
    assert "## Your Loop" in text


def test_executor_prompt_keeps_element_targeting_contract():
    """元素索引契约（`[i]` 是 1-based 行号、直接用）不能删。

    这节是索引契约的说明书：工具按 `[i]` 还原当屏那一行，模型抄 id 或坐标
    反而会点到别的东西。
    """
    text = _EXECUTOR_MD.read_text(encoding="utf-8")
    assert "## Element targeting" in text
    assert "1-based, used verbatim" in text
    assert "never have to copy an id or coordinates" in text
    assert "An index belongs to one screen" in text
    # 失败的目标不重复上报
    assert "do not re-report a target that already failed" in text


def test_executor_prompt_keeps_text_tools_table():
    """文本工具表要留着（含 `{{ executor_tools_list }}` 那行）。

    Executor 不 bind_tools（方案 §3.1：schema 每轮多背 4,008 token 且不进缓存键），
    工具怎么用全靠这张表教。
    """
    text = _EXECUTOR_MD.read_text(encoding="utf-8")
    assert "## Tools" in text
    assert "Available tools: {{ executor_tools_list }}" in text
    for tool in (
        "`launch_app`",
        "`tap`",
        "`long_press_on`",
        "`focus_and_input_text`",
        "`focus_and_clear_text`",
        "`swipe_percentages`",
        "`press_key`",
        "`wake_screen`",
    ):
        assert tool in text, f"工具表少了 {tool}"


def test_executor_prompt_keeps_perception_channels():
    """感知通道不能丢：**三条通道**（UI 元素 + OCR + 图片）+ 设备状态（息屏）。

    🔴 **2026-10-22 改**：图片通道**回来了**，但语义换成了
    「**Executor 自己举手**」（`need_screenshot`）—— 上一轮举手，下一轮
    才附当屏截图（不再同轮二问，省一次 LLM 调用）。

    三条通道都由 Manager 侧开关控制（`perception.ui_hierarchy` /
    `perception.ocr` / `perception.screenshot`），提示词必须**按开关渲染**：
    通道关着却还在教模型用那个通道，模型就会一直要一张不会来的图。

    丢了 UI/OCR 通道 ⇒ 模型看不见任何东西；丢了设备状态就会在黑屏上
    反复空转（真机教训：30 分钟息屏让一个任务从 30 步涨到 51 步）。
    """
    text = _EXECUTOR_MD.read_text(encoding="utf-8")
    assert "## Perception" in text
    # 三个通道都在，且**按开关分支**渲染
    assert "perception.ui_hierarchy" in text
    assert "perception.ocr" in text
    assert "perception.screenshot_channel" in text
    # 两个文本通道
    assert "UI elements" in text
    assert "OCR" in text
    assert "device state line" in text
    assert "wake_screen" in text


def test_executor_prompt_teaches_the_model_to_ask_for_a_picture():
    """图片通道开着时，提示词必须教它**怎么要**（`need_screenshot`）。

    通道关着时不能教 —— 否则模型会一直要一张不会来的图，而且它会以为自己
    能看到画面。**关掉时模板里一个图相关的词都不该有**（见
    `tests/mobile_use/test_screenshot_on_demand.py` 的零图词断言）。
    """
    text = _EXECUTOR_MD.read_text(encoding="utf-8")
    assert "need_screenshot" in text
    # 三种通道状态都有分支
    assert 'perception.screenshot_channel == "always"' in text
    assert 'perception.screenshot_channel == "on_demand"' in text
    # 「off」档 = 整段图的内容**不渲染**（模板里没有第三个分支）
    assert 'perception.screenshot_channel == "off"' not in text


def test_executor_prompt_has_no_tool_json_schema():
    """整份 prompt 里不许出现工具的完整参数定义（方案 §3.1）。"""
    text = _EXECUTOR_MD.read_text(encoding="utf-8")
    for marker in (
        "bind_tools",
        '"input_schema"',
        '"properties"',
        '"type": "object"',
        '"type": "array"',
        '"enum"',
        '"anyOf"',
    ):
        assert marker not in text, f"executor.md 里出现了不该有的东西：{marker}"


def test_executor_prompt_output_fields_match_executor_output():
    """Output 字段表要和 `agents/executor/types.py:ExecutorOutput` 逐字对齐。"""
    text = _EXECUTOR_MD.read_text(encoding="utf-8")
    for field in ("actions", "agent_thought", "done", "done_reason", "need_screenshot"):
        assert f"`{field}`" in text, f"Output 字段表少了 {field}"
    # agent_thought：≤1 句，且不许把目标陈述写进去（方案 §3.1.1 顺带说明）
    assert "one short sentence" in text
    assert "Never put the goal statement in it" in text


def test_executor_prompt_keeps_the_anchor_rules():
    """一次一个动作 / 不重复失败目标 / 锚在本次目标上 / 走不动如实说。"""
    text = _EXECUTOR_MD.read_text(encoding="utf-8")
    assert "One action at a time" in text
    assert "Never repeat a failed target" in text
    assert "Stay on this goal" in text
    assert "still advance *this* goal" in text
    assert "report the truth" in text
    assert "Never invent success" in text


def test_reporter_template_is_static():
    """reporter.md 完全静态：没有模板变量，渲染结果 == 原文。"""
    template_vars = _load().TEMPLATE_VARS
    assert template_vars["reporter/reporter.md"] == {}
    text = _REPORTER_MD.read_text(encoding="utf-8")
    assert "{{" not in text and "{%" not in text
    # Jinja 默认不留末尾换行（keep_trailing_newline=False），比对时归一化掉
    assert _render(_REPORTER_MD, {}).rstrip("\n") == text.rstrip("\n")


def test_reporter_prompt_only_writes_summary_and_evidence():
    """Reporter 只压缩、不判断：outcome 由代码给，事实字段也由代码填。"""
    text = _REPORTER_MD.read_text(encoding="utf-8")
    assert "## You are the **Reporter**" in text
    assert "outcome is already decided by code" in text
    assert "Report, don't judge" in text
    assert "Never invent" in text
    assert "No narration of every step" in text
    assert "`summary`" in text and "`evidence`" in text
    # Output 表里不许出现 outcome 行 —— 那是代码的字段，LLM 写了也会被覆盖
    assert "| `outcome` |" not in text
