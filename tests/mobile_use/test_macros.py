"""宏（脚本）能力的测试：数据模型、降级链、前置条件、指纹算法。

全是纯逻辑，不碰设备。
"""

from __future__ import annotations

from typing import Any

import pytest

from minitap.mobile_use.macros.model import Macro, MacroRunResult, MacroStep
from minitap.mobile_use.macros.runner import MacroEnv, ToolOutcome, run_macro
from minitap.mobile_use.macros.store import MacroCandidate, MacroStore
from minitap.mobile_use.utils.screen_fingerprint import (
    align_window,
    build_context,
    element_token,
    format_context,
    normalize_resource_id,
    parse_context,
    tree_tokens,
)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


def test_step_parses_tool_and_args():
    step = MacroStep.from_dict({"tool": "tap", "args": {"target": {"resource_id": "a:id/b"}}})
    assert step.tool == "tap"
    assert step.args["target"]["resource_id"] == "a:id/b"
    assert step.optional is False


def test_step_without_tool_is_rejected():
    with pytest.raises(ValueError):
        MacroStep.from_dict({"args": {}})
    with pytest.raises(ValueError):
        MacroStep.from_dict({"tool": "  "})


def test_step_args_must_be_object():
    """空列表是 falsy —— 早先用 ``or {}`` 会把它静默变成 {}，参数校验就形同虚设。"""
    with pytest.raises(ValueError):
        MacroStep.from_dict({"tool": "tap", "args": []})


def test_fallback_is_not_in_args():
    """降级数据**不能**进 args —— args 会原样发给工具，多余键会被参数校验拒掉。"""
    raw = {
        "tool": "tap",
        "args": {"target": {"resource_id": "a:id/b"}},
        "fallback": {
            "bounds": {"x": 1, "y": 2, "width": 3, "height": 4},
            "context": {"t": "b", "i": 0, "w": ["b"]},
        },
    }
    step = MacroStep.from_dict(raw)
    assert step.fallback_bounds == {"x": 1, "y": 2, "width": 3, "height": 4}
    assert step.fallback_context == {"t": "b", "i": 0, "w": ["b"]}
    assert "fallback" not in step.args
    assert "_recorded_bounds" not in step.args


def test_macro_roundtrip():
    raw = {
        "name": "ks_like",
        "app_package": "com.kuaishou.nebula",
        "device_serial": "SERIAL1",
        "when_to_use": "给当前视频点赞",
        "steps": [{"tool": "tap", "args": {"target": {"resource_id": "a:id/b"}}}],
    }
    macro = Macro.from_dict(raw)
    assert macro.app_package == "com.kuaishou.nebula"
    assert macro.device_serial == "SERIAL1"
    assert len(macro.steps) == 1
    assert macro.to_dict()["steps"][0]["tool"] == "tap"


# ---------------------------------------------------------------------------
# 指纹算法
# ---------------------------------------------------------------------------


def test_normalize_resource_id_drops_noise_suffixes():
    """id 改名后仍然能对上：``like_button`` 和 ``like_element_click_layout`` 都是 ``like``。"""
    assert normalize_resource_id("com.ks:id/like_button") == "like"
    assert normalize_resource_id("com.ks:id/like_element_click_layout") == "like"
    assert normalize_resource_id("com.ks:id/comment_element_click_layout") == "comment"
    assert normalize_resource_id("") == ""


def test_element_token_prefers_text_then_id():
    assert element_token({"text": "搜索", "resource_id": "a:id/x"}) == "搜索"
    assert element_token({"text": "", "resource_id": "a:id/like_button"}) == "like"
    assert element_token({"text": "", "resource_id": "", "class_name": "ImageView"}) == "ImageView"


def test_build_context_window_and_offset():
    elements = [
        {"kind": "ui", "text": "热门", "resource_id": "a:id/t1", "flags": "C"},
        {"kind": "ui", "text": "同城", "resource_id": "a:id/t2", "flags": "C"},
        {"kind": "ui", "text": "搜索", "resource_id": "a:id/search", "flags": "C"},
        {"kind": "ui", "text": "发现", "resource_id": "a:id/t4", "flags": "C"},
    ]
    context = build_context(elements, "a:id/search")
    assert context is not None
    assert context["t"] == "搜索"
    assert context["i"] == 2
    assert context["w"][context["i"]] == "搜索"


def test_build_context_returns_none_for_unknown_target():
    assert build_context([{"kind": "ui", "resource_id": "a:id/x", "flags": "C"}], "a:id/y") is None


def test_context_format_parse_roundtrip():
    context = {"t": "搜索", "i": 2, "w": ["热门", "同城", "搜索"]}
    text = format_context(context)
    assert text.startswith("ctx:")
    assert parse_context(text) == context
    assert parse_context("no fingerprint here") is None


def test_align_window_finds_shifted_anchor():
    """整屏平移（元素位置全变）不影响对齐 —— 对齐的是**序列**不是坐标。"""
    recorded = ["热门", "同城", "搜索", "发现", "生活"]
    current = ["推荐", "热门", "同城", "搜索", "发现", "生活", "关注"]
    anchor = align_window(recorded, current)
    assert anchor is not None
    # 录制的窗口第 0 个（热门）应对到当屏的第 1 个
    assert current[anchor] == "热门"
    # 目标在窗口里的偏移 2 ⇒ 当屏里就是「搜索」
    assert current[anchor + 2] == "搜索"


def test_align_window_handles_renamed_id():
    """id 改名（like_button → like_element_click_layout）归一化后仍能对上。"""
    recorded = ["collect", "like", "comment"]
    current = ["collect", "like", "comment"]
    assert align_window(recorded, current) == 0


def test_tree_tokens_uses_text_then_normalized_id():
    tree = [
        {"text": "搜索", "resource-id": "a:id/right_tv", "clickable": "true"},
        {"text": "", "resource-id": "a:id/like_element_click_layout", "clickable": "true"},
        {"text": "装饰", "resource-id": "a:id/label", "clickable": "false"},
    ]
    tokens = tree_tokens(tree)
    assert "搜索" in tokens
    assert "like" in tokens
    assert "装饰" not in tokens  # 不可交互的不进序列


# ---------------------------------------------------------------------------
# 引擎：前置条件
# ---------------------------------------------------------------------------


def _env(
    *,
    device_id: str | None = "SERIAL1",
    tree: list[dict] | None = None,
    outcomes: list[ToolOutcome] | None = None,
) -> tuple[MacroEnv, list[tuple[str, dict[str, Any]]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    queue = list(outcomes or [])

    async def dispatch(tool: str, args: dict[str, Any]) -> ToolOutcome:
        calls.append((tool, args))
        if queue:
            return queue.pop(0)
        return ToolOutcome(ok=True, message="ok")

    async def read_tree() -> list[dict]:
        return tree or []

    env = MacroEnv(dispatch=dispatch, read_tree=read_tree, device_id=device_id)
    return env, calls


@pytest.mark.asyncio
async def test_device_mismatch_refuses_to_run():
    """设备对不上必须拒跑 —— 真机上连着两台设备时，执行面会自动挑「第一台」。"""
    env, calls = _env(device_id="OTHER")
    macro = Macro.from_dict(
        {
            "name": "s",
            "device_serial": "SERIAL1",
            "steps": [{"tool": "tap", "args": {"target": {"resource_id": "a:id/b"}}}],
        }
    )
    result = await run_macro(env, macro)
    assert result.ok is False
    assert "SERIAL1" in result.reason
    assert calls == []


@pytest.mark.asyncio
async def test_app_mismatch_refuses_to_run():
    env, calls = _env(
        tree=[{"package": "com.android.settings", "text": "设置", "resource-id": "a:id/x"}]
    )
    macro = Macro.from_dict(
        {
            "name": "s",
            "app_package": "com.kuaishou.nebula",
            "steps": [{"tool": "tap", "args": {"target": {"resource_id": "a:id/b"}}}],
        }
    )
    result = await run_macro(env, macro)
    assert result.ok is False
    assert "com.kuaishou.nebula" in result.reason
    assert calls == []


# ---------------------------------------------------------------------------
# 引擎：降级链
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_target_falls_back_to_position():
    """id 失效 → 按录制位置兜底 → 成功，并**上报降级**。"""
    tree = [
        {
            "package": "com.app",
            "text": "点赞",
            "resource-id": "com.app:id/like_element_click_layout",
            "class": "android.widget.FrameLayout",
            "bounds": "[100,1000][200,1100]",
            "clickable": "true",
        }
    ]
    env, calls = _env(
        tree=tree,
        outcomes=[
            ToolOutcome(ok=False, message="the element ... is no longer on the screen"),
            ToolOutcome(ok=True, message="ok", data={"selector": "coordinates (150, 1050)"}),
        ],
    )
    macro = Macro.from_dict(
        {
            "name": "s",
            "app_package": "com.app",
            "steps": [
                {
                    "tool": "tap",
                    "args": {"target": {"resource_id": "com.app:id/like_button"}},
                    "fallback": {"bounds": {"x": 100, "y": 1000, "w": 100, "h": 100}},
                }
            ],
        }
    )
    result = await run_macro(env, macro)
    assert result.ok is True
    assert result.steps_executed == 1
    assert len(result.degradations) == 1
    assert "位置" in result.degradations[0]["got"]


@pytest.mark.asyncio
async def test_missing_target_falls_back_to_context():
    """id 失效 + 位置也平移了 → 靠上下文指纹（邻居序列）对齐。"""
    tree = [
        {
            "package": "com.app",
            "text": "热门",
            "resource-id": "com.app:id/t1",
            "class": "android.widget.FrameLayout",
            "bounds": "[0,100][100,200]",
            "clickable": "true",
        },
        {
            "package": "com.app",
            "text": "同城",
            "resource-id": "com.app:id/t2",
            "class": "android.widget.FrameLayout",
            "bounds": "[100,100][200,200]",
            "clickable": "true",
        },
        {
            "package": "com.app",
            "text": "搜索",
            "resource-id": "com.app:id/thanos_home_top_search",
            "class": "android.widget.ImageView",
            "bounds": "[200,100][300,200]",
            "clickable": "true",
        },
    ]
    env, calls = _env(
        tree=tree,
        outcomes=[
            ToolOutcome(ok=False, message="no longer on the screen"),
            # 第 2 次 dispatch 是**上下文兜底**发的那一次（位置故意对不上）
            ToolOutcome(ok=True, message="ok", data={"selector": "coordinates (250, 150)"}),
        ],
    )
    macro = Macro.from_dict(
        {
            "name": "s",
            "app_package": "com.app",
            "steps": [
                {
                    "tool": "tap",
                    "args": {"target": {"resource_id": "com.app:id/right_tv"}},
                    "fallback": {
                        "bounds": {"x": 5000, "y": 5000, "w": 10, "h": 10},  # 位置故意对不上
                        "context": {"t": "搜索", "i": 2, "w": ["热门", "同城", "搜索"]},
                    },
                }
            ],
        }
    )
    result = await run_macro(env, macro)
    assert result.ok is True
    assert result.steps_executed == 1
    assert any("上下文指纹" in d["got"] for d in result.degradations)


@pytest.mark.asyncio
async def test_failure_reports_which_step_and_why():
    """定位不到就**停**，报清「第几步、哪个工具、为什么」—— 绝不瞎点。"""
    env, _calls = _env(
        outcomes=[ToolOutcome(ok=False, message="no longer on the screen")] * 12,
    )
    macro = Macro.from_dict(
        {
            "name": "s",
            "steps": [
                {"tool": "tap", "args": {"target": {"resource_id": "a:id/b"}}, "max_retries": 1},
            ],
        }
    )
    result = await run_macro(env, macro)
    assert result.ok is False
    assert result.failed_index == 1
    assert result.failed_tool == "tap"
    assert "no longer on the screen" in result.reason


# ---------------------------------------------------------------------------
# 从历史里提取步骤（save_macro 走的就是这条路）
# ---------------------------------------------------------------------------


def _assistant(actions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "toolCalls": [{"name": "ExecutorOutput", "args": {"actions": actions, "agent_thought": "t"}}],
    }


def _tool(text: str) -> dict[str, Any]:
    return {"role": "tool", "name": "ExecutorOutput", "blocks": [{"type": "text", "text": text}]}


def test_steps_from_history_keeps_only_replayable_steps():
    """带 resource_id 的才录；index 定位的**丢掉**（跨运行不可靠）。"""
    from minitap.mobile_use.macros.compiler import steps_from_history

    messages = [
        _assistant([{"action": "launch_app", "app_name": "快手极速版"}]),
        _tool("App 'com.kuaishou.nebula' launched successfully."),
        _assistant([{"action": "tap", "target": {"index": 13}}]),
        _tool(
            "Tap on element with resource_id='com.kuaishou.nebula:id/right_tv' "
            "-> coordinates (1120, 201) was successful."
        ),
        _assistant([{"action": "tap", "target": {"index": 102}}]),
        _tool("Tap on element with index=102 coordinates (1086, 1934) was successful."),
        _assistant([{"action": "swipe_percentages", "start_x_percent": 50, "start_y_percent": 70,
                     "end_x_percent": 50, "end_y_percent": 35}]),
        _tool("Swipe is successful."),
    ]
    steps = steps_from_history(messages)
    tools = [s["tool"] for s in steps]
    # index 定位的那一步没有 resource_id ⇒ 不录
    assert tools == ["launch_app", "tap", "swipe"]
    assert steps[1]["args"]["target"]["resource_id"] == "com.kuaishou.nebula:id/right_tv"
    assert steps[2]["args"]["use_percentages"] is True
    assert steps[2]["args"]["end_y_percent"] == 35


def test_steps_from_history_picks_up_the_context_fingerprint():
    """工具结果里的 ``ctx:`` 指纹要被带进降级链。"""
    from minitap.mobile_use.macros.compiler import steps_from_history

    messages = [
        _assistant([{"action": "tap", "target": {"index": 7}}]),
        _tool(
            "Tap on element with resource_id='com.ks:id/like_element_click_layout' "
            "-> coordinates (1111, 1078) was successful. "
            'ctx:{"t":"like","i":2,"w":["collect","like","comment"]}'
        ),
    ]
    steps = steps_from_history(messages)
    assert len(steps) == 1
    context = steps[0]["fallback"]["context"]
    assert context["t"] == "like"
    assert context["w"] == ["collect", "like", "comment"]


# ---------------------------------------------------------------------------
# 脚本库客户端
# ---------------------------------------------------------------------------


def test_candidate_summary_is_short():
    """候选摘要会进 context，必须短。"""
    candidate = MacroCandidate(
        id=1,
        name="ks_like_and_comment",
        when_to_use="在快手极速版给当前视频点赞并评论" * 10,
        app_package="com.kuaishou.nebula",
        step_count=12,
    )
    assert len(candidate.summary()) <= 160


def test_store_url_from_env(monkeypatch):
    from minitap.mobile_use.macros import store

    monkeypatch.setenv("MOBILE_USE_MANAGER_URL", "http://example:1234/")
    assert store.manager_url() == "http://example:1234"
    monkeypatch.delenv("MOBILE_USE_MANAGER_URL", raising=False)
    assert store.manager_url() == store.DEFAULT_MANAGER_URL
