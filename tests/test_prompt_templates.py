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
_EXECUTOR_ZH_MD = _AGENTS_DIR / "executor" / "executor-zh.md"
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


def _render_executor_system(variables: dict[str, object], path: Path = _EXECUTOR_MD) -> str:
    """Executor 的模板是宏库：正文的渲染入口是 `system_message` 宏。"""
    env = Environment(undefined=StrictUndefined, autoescape=False)
    template = env.from_string(path.read_text(encoding="utf-8"))
    return template.module.system_message(**variables)


def _perception(
    ui_hierarchy: bool = True,
    ocr: bool = True,
    vision: bool = True,
):
    """感知开关的替身（和 `scripts/check_prompt_templates.py` 里的是同一个类）。"""
    return _load()._Perception(ui_hierarchy, ocr, vision)


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


def test_executor_template_declares_exactly_three_vars():
    """executor.md 的 system_message 变量恰好三个：platform / executor_tools_list / perception。

    多一个 → 渲染端要多传一个（调用点会对不上）；少一个 → 模板引用了没传的变量。

    `perception` 是**感知通道开关状态**（Manager 侧）：两条通道开着哪几条，
    提示词里必须写清楚 —— 否则模型会按一个被关掉的通道去寻址。
    一次运行里它是恒定的 ⇒ 放 system prompt 不伤缓存。

    目标**不是** system prompt 的变量：它每子目标变，只能走独立消息块。
    """
    template_vars = _load().TEMPLATE_VARS
    assert set(template_vars["executor/executor.md"]) == {
        "platform",
        "executor_tools_list",
        "perception",
    }
    assert set(template_vars["executor/executor-zh.md"]) == {
        "platform",
        "executor_tools_list",
        "perception",
    }


def test_executor_templates_own_every_message_block():
    """所有进 prompt 的块都必须是模板宏 —— 代码不再自带文案。

    宏名就是「块契约」：prompt.py / executor.py 按名字取，提示词作者只改
    ``.md``。中英两份必须逐个对齐，缺一个就会在 zh 运行时回落成空串。
    """
    macros = (
        "system_message",
        "goal_message",
        "device_state_message",
        "screen_elements_message",
        "dispatch_rejection_message",
        "vision_subjects_message",
        "retry_message",
        "extra_actions_message",
        "step_limit_message",
        "no_goal_message",
        "done_fallback_message",
    )
    for path in (_EXECUTOR_MD, _EXECUTOR_ZH_MD):
        text = path.read_text(encoding="utf-8")
        for macro in macros:
            assert f"macro {macro}(" in text, f"{path.name} 缺少宏 {macro}"


def test_executor_prompt_renders_without_leftover_placeholders():
    """渲染后不能残留 Jinja 占位符 —— 残留在 prompt 里等于给了模型一句废话。"""
    rendered = _render_executor_system(
        {
            "platform": "android",
            "executor_tools_list": "tap, back",
            "perception": _perception(),
        }
    )
    assert "{{" not in rendered and "{%" not in rendered
    assert "You are the **Executor**" in rendered
    assert "## Tools" in rendered


def test_executor_system_prompt_does_not_carry_the_goal():
    """目标每子目标变 → 只能走独立消息块，绝不能进 system prompt。

    一旦内联，换子目标时整块 system 缓存从目标那一行起全部重算。
    """
    text = _EXECUTOR_MD.read_text(encoding="utf-8")
    system_part, marker, blocks_part = text.partition("{%- macro goal_message")
    assert marker, "executor.md 缺少 goal_message 宏"
    assert "{{ goal }}" not in system_part
    assert "{% if goal" not in system_part
    # 目标只能从消息块进来
    assert "{{ goal }}" in blocks_part


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

    ⚠️ 2026-10-23：提示词由作者本人重写过（`## Element targeting` →
    `## Screen composition`，整段措辞被压缩）。所以这条断言的是**契约本身**
    —— 行格式、1-based、`index` 寻址 ——
    而不是某一段标题或某一句措辞：标题和措辞还能再变，契约不能丢。
    """
    text = _EXECUTOR_MD.read_text(encoding="utf-8")
    # 行格式说明书：`[i] "text" id=... Type (x,y,WxH) flags`
    assert "[i]" in text
    assert "id=" in text
    # 编号是 1-based（OCR 块行首的 `[i]`）
    assert "1-based" in text
    # 唯一寻址方式：元素表（UI 行 + OCR 行）统一用 `index`
    assert "`index`" in text


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
    """感知通道不能丢：元素表（UI 行 + OCR 行）+ 设备状态（息屏）。

    2026-10-24：与 zh 版对齐后，system prompt 不再按开关分支渲染 ——
    两条通道改成一句静态描述，开关带来的差异由**每轮消息块**自己说清：
    两条通道全关时 `device_state_message` 明确告知「没有感知通道」。

    2026-10-28：UI 行和 OCR 行已经是同一张元素表，不再有独立的 OCR 块。

    所以这里断言的是「通道信息没有丢」，而不是「必须在 system prompt 里分支」。
    """
    text = _EXECUTOR_MD.read_text(encoding="utf-8")
    assert "## Perception" in text
    # 同一张表里包含两种行
    assert "element table" in text
    assert "OCR text rows" in text
    # 开关带来的状态由运行时块承载
    assert "NO perception channel is available" in text
    # 息屏告知（真机教训：30 分钟息屏让任务从 30 步涨到 51 步）
    assert "Screen is OFF" in text
    assert "wake_screen" in text


def test_executor_prompt_has_no_trace_of_the_picture_channel():
    """截图这条通道**整条移除**（2026-10-27）：提示词里不许再有图的痕迹。

    下线过两次：先是「每轮附图」那条通道（2026-10-23，`need_screenshot`
    这种「举手要图」的字段），后是「按需工具」那条（2026-10-27，
    `take_screenshot` —— 模型自己喊一声要看图）。

    ⇒ 这里断言：提示词里**一个**截图痕迹都不许有 ——
    工具表行、说明段落、通道标题，全部不许出现。

    ⚠️ 断言的是**「不许有」**，不是「必须有什么」：加了新工具再改一遍
    这条断言的成本，比留一条「模型得能叫出这个名字」的断言低得多 ——
    截图通道已经**整条**没了，没有模型需要叫的名字。
    """
    for path in (_EXECUTOR_MD, _EXECUTOR_ZH_MD):
        text = path.read_text(encoding="utf-8")
        for marker in (
            "need_screenshot",
            "`take_screenshot`",
            "take_screenshot",
            "Image perception",
            "图片感知",
        ):
            assert marker not in text, f"{path.name} 里还有截图痕迹：{marker}"
    # 两条**文字**通道还在：元素表 / 视觉主体列表
    text = _EXECUTOR_MD.read_text(encoding="utf-8")
    assert "element table" in text
    assert "visual-subjects" in text


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
    for field in ("actions", "agent_thought", "done", "done_reason"):
        assert f"`{field}`" in text, f"Output 字段表少了 {field}"
    # agent_thought：**先说上一步的结果，再说本动作的原因**（与 zh 版字段表对齐）
    #
    # 2026-10-24 改的：以前是「One sentence explaining this action」——
    # 纯前视的。实测（execution 853）：执行器把 Home 按了 13 次，每一次都以为
    # Home 会回到桌面，因为历史里「按了 Home」后面跟的永远是「按成功了」，
    # **没有任何东西说「屏幕还是那个」**。
    # ⇒ 框架现在把「上一步的结果」写进历史，提示词要求它**先读那个结果、
    # 对照自己要的东西**，再决定下一步。
    assert "First state the result of the previous step" in text
    assert "One sentence explaining this action" not in text


def test_executor_schema_description_agrees_with_the_prompt():
    """🔴 **schema 的字段描述是模型真正看到的东西** —— 它必须和提示词说同一件事。

    2026-10-24 踩过：`agent_thought` 的 schema 描述还写着「Max 1 sentence:
    why this action」，而提示词已经改成「先说上一步的结果，再说本动作的原因」
    ⇒ **两边对着干**，模型会照着 schema 那个走，历史里的「上一步的结果」白记。

    ⇒ 这条测试盯住 schema 描述：改了它、忘了改提示词（或反过来），这里会红。
    """
    from minitap.mobile_use.agents.executor.types import ExecutorOutput

    description = ExecutorOutput.model_fields["agent_thought"].description or ""
    assert "result of the previous step" in description
    assert "Max 1 sentence" not in description


def test_executor_prompt_tells_the_model_to_use_the_previous_result():
    """提示词必须明说「历史里有上一步的结果，要先对照它」。

    框架只负责把**客观事实**搬进历史（见 `contextor.render_action_result`）；
    「这个动作没用」是**相对于目标**的判断，框架做不了 ⇒ 必须由模型做。
    ⇒ 提示词不写清楚，模型就会忽略历史里那几行，继续原地打转。
    """
    en = " ".join(_EXECUTOR_MD.read_text(encoding="utf-8").split())
    zh = " ".join(_EXECUTOR_ZH_MD.read_text(encoding="utf-8").split())

    assert "the result of the previous step" in en
    assert "上一步的结果" in zh
    # 两边都要说清「结果不对就换动作，别重复验」
    assert "take a different action" in en
    assert "换动作" in zh
    # 也要说清「结果对了就 done，别再验一遍」
    #
    # ⚠️ 2026-10-24：断言的是**契约**，不是某一句措辞 ——
    # 模板压缩过措辞（"instead of verifying it again" → "must not verify again"，
    # 「别再验一遍」→「禁止再验一遍」），但「不许再验一遍」这件事必须还在。
    assert "must not verify again" in en
    assert "禁止再验一遍" in zh


def test_executor_prompt_keeps_the_anchor_rules():
    """一次一个动作 / 目标不变、做不到就如实说。

    2026-10-24：与 zh 版对齐后，规则不再单列一节：
    「目标不变、做不到就如实说」写在开头的目标陈述里。

    2026-10-28：Executor 的失败事实块已移除。
    """
    text = _EXECUTOR_MD.read_text(encoding="utf-8")
    assert "One action at a time" in text
    assert "reach it, or report truthfully that you could not" in text


def test_executor_prompt_demands_done_when_the_goal_is_reached():
    """目标达成**必须** `done: true` —— 中英两版都要有。

    2026-10-24 真机实测（mobile-manager execution 851）：cortex 下发了一个
    「A **并且** B」的验证型子目标，executor 两个条件都验过了（每一步 thought
    都写着「已确认」），却**一次都没发过 `done: true`** —— 该子目标跑了 54 次
    调用、51 个动作步、59.6 万 token，任务永远收不了尾。

    ⇒ 这条不是风格偏好，是**「没有出口」的兜底**：executor 循环唯一的出口
    就是模型说 done（``graph.py`` 的 ``post_executor_gate``）。模型不说，
    就永远在里面转。
    """
    en = _EXECUTOR_MD.read_text(encoding="utf-8")
    zh = _EXECUTOR_ZH_MD.read_text(encoding="utf-8")

    # 归一化空白：提示词是按行宽排版的，断言不能被换行炸掉
    flat_en = " ".join(en.split())
    flat_zh = " ".join(zh.split())

    assert "you **MUST** set `done: true`" in flat_en
    assert "`done: true`" in flat_zh

    # 死循环的另一只手：已经验过的条件不要再验。
    #
    # ⚠️ 2026-10-24：措辞被压缩过（断言的是**契约**，不是原句）。
    assert "re-checking produces no progress" in flat_en
    assert "重复验证不产生进展" in flat_zh


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
