"""验证 Reporter 的 outcome 判定链路（纯代码，不调 LLM）。

要锁的：
1. `executor_done=True` → completed
2. 有失败记录 + 未 done → failed
3. 步数超限 → blocked（**不是** completed！executor_done 在熔断时也是 True，
   那个 True 只表示「收尾」，不表示「达成」）
4. 事实字段从 state 直搬，逐字
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "minitap"))

from minitap.mobile_use.agents.executor.executor import MAX_STEPS_PER_SUBGOAL  # noqa: E402
from minitap.mobile_use.agents.reporter.reporter import (  # noqa: E402
    build_actions_taken,
    build_failed_targets,
    determine_outcome_for,
)
from minitap.mobile_use.graph.state import State  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from typing import Any, cast  # noqa: E402


def _state(**kw) -> State:
    # `base` 的值类型是异构的（State 的每个字段类型都不一样），直接 `State(**base)`
    # 会让 pyright 逐字段报「类型不对」。这里只做一次 cast —— 探针脚本要的是
    # 「把字段塞进去」，不是逐字段做静态校验。
    base = dict(
        messages=[],
        initial_goal="g",
        subgoal_plan=[],
        latest_ui_hierarchy=None,
        latest_ocr_results=None,
        latest_screenshot=None,
        focused_app_info=None,
        device_date=None,
        structured_decisions=None,
        complete_subgoals_by_ids=[],
        agents_thoughts=[],
        executor_messages=[],
        cortex_last_thought=None,
    )
    base.update(kw)
    return State(**cast(Any, base))


def main() -> int:
    failures: list[str] = []

    def check(label: str, got, expected) -> None:
        ok = got == expected
        if not ok:
            failures.append(f"{label}: 期望 {expected!r}，实际 {got!r}")
        print(f"  {'✓' if ok else '✗'} {label}: {got!r}")

    print("=" * 70)
    print("1. outcome 判定")
    print("=" * 70)
    check("done=True → completed", determine_outcome_for(_state(executor_done=True)), "completed")
    check(
        "有失败 + 未 done → failed",
        determine_outcome_for(
            _state(executor_done=False, executor_failures=[{"key": "index=1|id=x"}])
        ),
        "failed",
    )
    # ⚠️ 步数上限**默认不限**（`MAX_STEPS_PER_SUBGOAL = None`，2026-10-22 起）。
    # ⇒ 默认下 outcome **永远不是 blocked**；下面两条要测 blocked，得先临时
    # 配一个上限（monkeypatch 那个模块常量）。
    if MAX_STEPS_PER_SUBGOAL is None:
        check(
            "默认不限 ⇒ 步数再多也不会 blocked",
            determine_outcome_for(_state(executor_done=True, executor_steps=999)),
            "completed",
        )
    else:
        check(
            f"步数超限({MAX_STEPS_PER_SUBGOAL + 1}) → blocked",
            determine_outcome_for(
                _state(executor_done=True, executor_steps=MAX_STEPS_PER_SUBGOAL + 1)
            ),
            "blocked",
        )
        check(
            "步数恰好等于上限 → 不算超限",
            determine_outcome_for(
                _state(executor_done=True, executor_steps=MAX_STEPS_PER_SUBGOAL)
            ),
            "completed",
        )

    print()
    print("=" * 70)
    print("2. 事实字段直搬")
    print("=" * 70)
    facts = [
        {
            "key": "index=48|id=com.x:id/like",
            "index": 48,
            "resource_id": "com.x:id/like",
            "label": "点赞",
            "reason": "no element with that id",
        }
    ]
    built = build_failed_targets(facts)
    check("失败目标条数", len(built), 1)
    check("失败目标 index", built[0].index, 48)
    check("失败目标 label", built[0].label, "点赞")
    check("失败目标 reason 逐字", built[0].reason, "no element with that id")

    msgs = [
        AIMessage(
            content="",
            tool_calls=[{"name": "tap", "args": {"index": 7}, "id": "c1"}],
        ),
        ToolMessage(content="Tapped successfully.", tool_call_id="c1"),
    ]
    actions = build_actions_taken(msgs)
    check("动作条数", len(actions), 1)
    print(f"     动作描述: {actions[0]!r}")

    print()
    print("=" * 70)
    print("3. 失败目标为空时不应报错")
    print("=" * 70)
    check("空事实账本", len(build_failed_targets([])), 0)
    check("无历史时的动作", build_actions_taken([]), [])

    print()
    print("=" * 70)
    print("结果：" + ("全部通过" if not failures else f"{len(failures)} 项失败"))
    for f in failures:
        print(f"  ✗ {f}")
    print("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
