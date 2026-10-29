"""编译并渲染所有 agent 提示词模板，把「Jinja 语法错」挡在真机跑之前。

起因：cortex.md 末尾多写了一个 `{% if locked_app_package %}` 却没写 `{% endif %}`，
代价是真机上跑完整条任务才炸 TemplateSyntaxError —— agent 已经启动、已经在点屏幕，
最后一步渲染 prompt 时才崩。模板必须能在不接设备的情况下先编一遍。

这里的变量表：其他 agent 逐条抄自真实的 `.render(...)` 调用点（cortex.py /
planner.py / contextor.py / orchestrator.py / hopper.py / video_analyzer.py /
reporter.py）；Executor 已按 `prompt_manifest.toml` 拆段，变量表由清单生成
（40 多个小文件逐条登记容易漏）。所以它既能挡住语法错，也能挡住「模板引用了
代码从没传过的变量」。

可用变量取值要覆盖两种分支：{% if x %} 只跑一种取值时，少写的 {% else %} / {% endif %}
依然编得过，所以 locked_app_package / video_recording_enabled 都要试 None 与真值。

用法：
    python scripts/check_prompt_templates.py
退出码 0 = 全通过。
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from jinja2 import Environment, StrictUndefined

ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = ROOT / "minitap" / "mobile_use" / "agents"

# Executor 的提示词按 manifest 拆成 system / messages 两组文件；变量表由清单生成，
# 新增一段只要在 toml 里登记，这里的覆盖度检查会自动跟上。
EXECUTOR_DIR = AGENTS_DIR / "executor"
EXECUTOR_PROMPTS = EXECUTOR_DIR / "prompts"
EXECUTOR_MANIFEST = EXECUTOR_DIR / "prompt_manifest.toml"

EXECUTOR_SYSTEM_VARS: dict[str, object] = {
    "platform": "android",
    "executor_tools_table": "| `tap` | `target` | Tap |",
    "vision_enabled": True,
}

EXECUTOR_MESSAGE_VARS: dict[str, dict[str, object]] = {
    "task_started": {"task_started_at": "2026-10-28 10:00"},
    "goal": {"goal": "进入 WLAN 页面"},
    "previous_attempt": {
        "outcome": "failed",
        "summary": "点了搜索框，没进到结果页",
        "reason": "目标不在当前屏幕上",
        "failed_targets": '"搜索" search_box',
    },
    "device_state": {
        "device_date": "2026-10-28",
        "focused_app_info": "设置",
        "screen_off": True,
        "no_perception": True,
    },
    "screen_elements": {"rows": '[1] "WLAN" | Class | (0,0,10,10) | C'},
    "vision_subjects": {"rows": '[1] "齿轮图标" (0,0,10,10)'},
    "dispatch_rejection": {"reason": "tap is not a known tool"},
    "retry": {"reason": "the answer was not an ExecutorOutput object"},
    "extra_actions": {"count": 2},
    "stall_notice": {
        "kind": "screen",
        "repeats": 3,
        "window": 8,
        "fingerprint": "abc123",
        "vision_enabled": True,
    },
    "step_limit": {"limit": 30},
    "no_goal": {},
    "done_fallback": {},
    "failed_fallback": {},
}


def _language_variants(rel: str) -> list[Path]:
    """英文原版 + 存在的中文兄弟文件。缺中文版只回落那一段，不算错误。"""
    base = EXECUTOR_PROMPTS / rel
    zh = base.with_name(f"{base.stem}-zh{base.suffix}")
    return [base, zh] if zh.is_file() else [base]


def _executor_prompt_vars() -> dict[str, dict[str, object]]:
    """把 manifest 里的每一段映射成「文件 -> 渲染变量」。"""
    manifest = tomllib.loads(EXECUTOR_MANIFEST.read_text(encoding="utf-8"))
    entries: dict[str, dict[str, object]] = {}
    for row in manifest.get("system", []):
        for path in _language_variants(row["file"]):
            entries[path.relative_to(AGENTS_DIR).as_posix()] = dict(EXECUTOR_SYSTEM_VARS)
    for row in manifest.get("message", []):
        for path in _language_variants(row["file"]):
            entries[path.relative_to(AGENTS_DIR).as_posix()] = dict(
                EXECUTOR_MESSAGE_VARS[row["name"]]
            )
    return entries


class _Perception:
    """感知开关的替身（和 `context.PerceptionConfig` 同形）。

    三条通道：UI 元素表 + OCR + 按需视觉扫描。后者的工具表行是条件渲染的，
    所以变体里必须既有开、也有关。图片通道已于 2026-10-23 下线 —— 屏幕截图
    不再进模型，视觉扫描返回的是文字列表。
    """

    def __init__(
        self,
        ui_hierarchy: bool = True,
        ocr: bool = True,
        vision: bool = True,
    ) -> None:
        self.ui_hierarchy = ui_hierarchy
        self.ocr = ocr
        self.vision = vision


def _perception_variants() -> list[_Perception]:
    """感知开关的典型组合 —— 模板里每种都有一条分支，都要编到。

    特别要编到**全关**：那时提示词里必须如实告诉模型「没有任何视觉感知」，
    那条分支必须能正常渲染。视觉扫描也要编到关闭态，否则工具表行的
    ``{% if perception.vision %}`` 分支永远走不到。
    """
    return [
        _Perception(True, True, True),  # 默认：三条全开
        _Perception(True, False, True),  # 只留 UI 元素表 + 视觉
        _Perception(False, True, True),  # 只留 OCR + 视觉
        _Perception(False, False, False),  # 全关
    ]


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
    # Executor 的 system / messages 在下方由 manifest 生成，这里不逐条登记。
    # reporter.md **完全静态**：没有任何模板变量。Reporter 只写 summary + 挑
    # evidence，outcome / actions_taken / failed_targets 全由代码填（见
    # agents/executor/types.py:ExecutorReport），所以它不需要模板变量。
    "reporter/reporter.md": {},
    # reporter 的中文兄弟文件：同样完全静态（变量表为空）。
    "reporter/reporter-zh.md": {},
    # contextor.py:218
    "contextor/contextor.md": {
        "task_goal": "打开快手赚钱",
        "subgoal_plan": "subgoal-0",
        "locked_app_package": "com.ss.android.ugc.aweme",
        "current_app_package": "com.ss.android.ugc.aweme",
        "agents_thoughts": ["thought"],
    },
    # contextor 的中文兄弟文件：变量表与英文版一致（zh 只翻散文，字面量不动）。
    "contextor/contextor-zh.md": {
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
    # orchestrator 的中文兄弟文件（system + human 两块）。
    "orchestrator/orchestrator-zh.md": {"platform": "android"},
    "orchestrator/human-zh.md": {
        "initial_goal": "打开快手赚钱",
        "subgoal_plan": "subgoal-0",
        "subgoals_to_examine": "subgoal-0",
        "agent_thoughts": "thought",
    },
    # hopper.py:29 —— .render() 无参
    "hopper/hopper.md": {},
    # hopper 的中文兄弟文件：同样无参渲染。
    "hopper/hopper-zh.md": {},
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
    # outputter 的中文兄弟文件：变量表一致。
    "outputter/human-zh.md": {
        "initial_goal": "打开快手赚钱",
        "agents_thoughts": "thought",
        "structured_output": {"x": 1},
        "output_description": "结构化输出",
        "last_ai_message": "done",
    },
}

TEMPLATE_VARS.update(_executor_prompt_vars())

# 这些变量在生产里取值可能为 None，每个取值都要编一遍。
OPTIONAL_VARS = (
    "locked_app_package",
    "video_recording_enabled",
    # Executor system：只控制 vision 段装不装，模板本身不读这个变量。
    "vision_enabled",
    # Executor device-state：每个标志为真/为假两条路径都要编到。
    "device_date",
    "focused_app_info",
    "screen_off",
    "no_perception",
    # Executor retry / dispatch：失败原因可能缺省。
    "reason",
    # Executor stall-notice：两种句式（screen / action）都要编到。
    "kind",
)


def _branches(base: dict[str, object]) -> list[dict[str, object]]:
    """基础变量 + 每个可选变量的 None/真值组合。"""
    variants: list[dict[str, object]] = [dict(base)]
    for name in OPTIONAL_VARS:
        if name not in base:
            continue
        for value in (None, base[name]):
            variants.append({**base, name: value})
    # perception 有两个独立开关，模板里每种组合都有分支 —— 都要编一遍。
    if isinstance(base.get("perception"), _Perception):
        for perception in _perception_variants():
            variants.append({**base, "perception": perception})
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
    for rel in sorted(set(TEMPLATE_VARS) - on_disk):
        failures.append(f"{rel}: 变量表登记了但磁盘上不存在")

    for rel in sorted(set(TEMPLATE_VARS) & on_disk):
        path = AGENTS_DIR / rel
        source = path.read_text(encoding="utf-8")
        # Executor 的模板是「宏库」：正文住在 system_message 宏里，直接 render
        # 只会输出空串（等于什么都没校验）。宏调用才是它真实的渲染入口。
        # 用源码判断而不是 `hasattr(template.module, ...)`：后者会为了探测而
        # 渲染**所有**模板的正文，别的模板缺变量时会在这里误报。
        template = env.from_string(source)
        is_macro_library = "macro system_message" in source
        for branch in _branches(TEMPLATE_VARS[rel]):
            detail = ", ".join(f"{k}={v!r}" for k, v in branch.items() if k in OPTIONAL_VARS)
            label = f"{rel} [{detail or '默认'}]"
            checked += 1
            try:
                if is_macro_library:
                    template.module.system_message(**branch)
                else:
                    template.render(**branch)
            except Exception as e:  # noqa: BLE001 - 渲染期任何异常都算坏模板
                failures.append(f"{label}: {type(e).__name__}: {e}")
    return checked, failures


def main() -> int:
    checked, failures = check()
    for f in failures:
        print(f"FAIL {f}")
    passed = checked - len(failures)
    print(f"\n{passed}/{checked} 渲染通过" if failures else f"{checked}/{checked} 渲染通过")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
