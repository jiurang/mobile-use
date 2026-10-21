"""编译并渲染所有 agent 提示词模板，把「Jinja 语法错」挡在真机跑之前。

起因：cortex.md 末尾多写了一个 `{% if locked_app_package %}` 却没写 `{% endif %}`，
代价是真机上跑完整条任务才炸 TemplateSyntaxError —— agent 已经启动、已经在点屏幕，
最后一步渲染 prompt 时才崩。模板必须能在不接设备的情况下先编一遍。

这里的变量表逐条抄自各 agent 真实的 `.render(...)` 调用点（cortex.py / planner.py /
executor.py / contextor.py / orchestrator.py / hopper.py / video_analyzer.py），
所以它既能挡住语法错，也能挡住「模板引用了代码从没传过的变量」。

可用变量取值要覆盖两种分支：{% if x %} 只跑一种取值时，少写的 {% else %} / {% endif %}
依然编得过，所以 locked_app_package / video_recording_enabled 都要试 None 与真值。

用法：
    python scripts/check_prompt_templates.py
退出码 0 = 全通过。
"""

from __future__ import annotations

import sys
from pathlib import Path

from jinja2 import Environment, StrictUndefined
from jinja2.exceptions import TemplateError

ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = ROOT / "minitap" / "mobile_use" / "agents"


class _Perception:
    """planner/cortex 里 {{ perception.xxx }} 读的就是这三个字段。

    `screenshot` 是三态（True / False / "on_demand"），模板用
    `screenshot_label` 显示状态文案 —— "on_demand" 是真值，直接用真假判断
    会把它显示成 "enabled"，模型就以为图已经在手里了。
    """

    def __init__(
        self,
        ui_hierarchy: bool = True,
        ocr: bool = True,
        screenshot: bool | str = True,
    ) -> None:
        self.ui_hierarchy = ui_hierarchy
        self.ocr = ocr
        self.screenshot = screenshot

    @property
    def screenshot_label(self) -> str:
        if self.screenshot is True:
            return "enabled"
        if self.screenshot is False:
            return "DISABLED"
        return "on demand"


def _perception_variants() -> list[_Perception]:
    return [_Perception(True, True, True), _Perception(False, False, False)]


# 每个模板一条：渲染时调用方真实提供的变量。
# 键是相对 AGENTS_DIR 的路径，值是基础变量表。
TEMPLATE_VARS: dict[str, dict[str, object]] = {
    # cortex.py:164 —— 整段静态：目标/计划/feedback 已移到 messages，不再入模板。
    "cortex/cortex.md": {
        "platform": "android",
        "executor_tools_list": "tools",
        "locked_app_package": "com.ss.android.ugc.aweme",
        "perception": _Perception(),
    },
    # 同上，load_cortex_prompt() 可以把模板换成 cortex.zh.md（提示词 A/B）；
    # zh 与 en 结构保持逐块对应，变量也一致。
    "cortex/cortex.zh.md": {
        "platform": "android",
        "executor_tools_list": "tools",
        "locked_app_package": "com.ss.android.ugc.aweme",
        "perception": _Perception(),
    },
    # planner.py:46 / planner.py:55
    "planner/planner.md": {
        "platform": "android",
        "executor_tools_list": "tools",
        "locked_app_package": "com.ss.android.ugc.aweme",
        "current_foreground_app": "com.ss.android.ugc.aweme",
        "video_recording_enabled": False,
    },
    "planner/human.md": {
        "action": "plan",
        "initial_goal": "打开快手赚钱",
        "previous_plan": "subgoal-0",
        "agent_thoughts": "thought",
    },
    # executor.py:126
    "executor/executor.md": {"platform": "android"},
    # contextor.py:218
    "contextor/contextor.md": {
        "task_goal": "打开快手赚钱",
        "subgoal_plan": "subgoal-0",
        "locked_app_package": "com.ss.android.ugc.aweme",
        "current_app_package": "com.ss.android.ugc.aweme",
        "agents_thoughts": ["thought"],
    },
    # orchestrator.py:61 / orchestrator.py:64
    "orchestrator/orchestrator.md": {"platform": "android"},
    "orchestrator/human.md": {
        "initial_goal": "打开快手赚钱",
        "subgoal_plan": "subgoal-0",
        "subgoals_to_examine": "subgoal-0",
        "agent_thoughts": "thought",
    },
    # hopper.py:29 —— .render() 无参
    "hopper/hopper.md": {},
    # video_analyzer.py:58 / video_analyzer.py:62
    "video_analyzer/video_analyzer.md": {},
    "video_analyzer/human.md": {"prompt": "把视频里的数字报出来"},
    # outputter.py:28 —— system prompt 是硬编码字符串，没有 .md 模板
    "outputter/human.md": {
        "initial_goal": "打开快手赚钱",
        "agents_thoughts": "thought",
        "structured_output": {"x": 1},
        "output_description": "结构化输出",
        "last_ai_message": "done",
    },
}

# 这些变量在生产里取值可能为 None，每个取值都要编一遍。
OPTIONAL_VARS = ("locked_app_package", "video_recording_enabled")


def _branches(base: dict[str, object]) -> list[dict[str, object]]:
    """基础变量 + 每个可选变量的 None/真值组合。"""
    variants: list[dict[str, object]] = [dict(base)]
    for name in OPTIONAL_VARS:
        if name not in base:
            continue
        for value in (None, base[name]):
            variants.append({**base, name: value})
    # perception 有三个独立开关，至少要覆盖「全开」和「全关」。
    if isinstance(base.get("perception"), _Perception):
        variants.append({**base, "perception": _Perception(False, False, False)})
    return variants


def check() -> tuple[int, list[str]]:
    """返回 (渲染次数, 失败描述列表)。空列表 = 全通过。"""
    env = Environment(undefined=StrictUndefined, autoescape=False)
    failures: list[str] = []
    checked = 0

    # 守卫自身的覆盖度：新增 .md 没登记就直接失败，否则「全过」是假的。
    on_disk = {p.relative_to(AGENTS_DIR).as_posix() for p in AGENTS_DIR.rglob("*.md")}
    uncovered = sorted(on_disk - set(TEMPLATE_VARS))
    for rel in uncovered:
        failures.append(f"{rel}: 磁盘上存在但未登记渲染变量，无法校验")

    for rel in sorted(set(TEMPLATE_VARS) & on_disk):
        path = AGENTS_DIR / rel
        source = path.read_text(encoding="utf-8")
        for branch in _branches(TEMPLATE_VARS[rel]):
            detail = ", ".join(f"{k}={v!r}" for k, v in branch.items() if k in OPTIONAL_VARS)
            label = f"{rel} [{detail or '默认'}]"
            checked += 1
            try:
                env.from_string(source).render(**branch)
            except Exception as e:  # noqa: BLE001 - 渲染期任何异常都算坏模板
                failures.append(f"{label}: {type(e).__name__}: {e}")
    return checked, failures


def main() -> int:
    checked, failures = check()
    for f in failures:
        print(f"FAIL {f}")
    print(f"\n{checked - len(failures)}/{checked} 渲染通过" if failures else f"{checked}/{checked} 渲染通过")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
