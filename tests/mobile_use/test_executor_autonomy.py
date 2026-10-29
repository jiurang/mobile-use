"""Executor 自主执行改造的回归测试。

覆盖方案 `doc/executor-autonomy-plan.md` §11.1 的 T1~T13 里能离线验证的那些。

几个最关键的（都是审计里判过「致命」的）：

* **T12 元素表在 prompt 末尾** —— 元素表每轮变，放中间会把它之后的全部历史每轮重算，
  前缀缓存命中率**递减**（实测 53.8%，低于现状 74.6%）。放末尾才递增（96.6%）。
  一行断言就能锁住。
* **T3 outcome 由代码判定** —— `determine_outcome` 不经 LLM，所以 Reporter
  没有机会把失败说成完成。
* **T5 失败事实跨子目标** —— 熔断账本 `failed_targets` 成功即清、换键即替换，
  跨子目标的事实必须走 append-only 的 `executor_failures`。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "minitap"))

import pytest
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langgraph.errors import GraphRecursionError  # noqa: E402
from langgraph.graph.message import REMOVE_ALL_MESSAGES  # noqa: E402
from pytest import MonkeyPatch  # noqa: E402

from minitap.mobile_use.agents.cortex.dispatcher import (  # noqa: E402
    dispatch_decisions,
    dispatch_decisions_with_reason,
)
from minitap.mobile_use.agents.executor.executor import (  # noqa: E402
    MAX_STEPS_PER_SUBGOAL,
    ExecutorNode,
    _rejection_code,
)
from minitap.mobile_use.agents.executor.prompt import (  # noqa: E402
    EXECUTOR_PROMPTS_DIR,
    build_executor_messages,
    render_executor_system,
)
from minitap.mobile_use.agents.executor.types import (  # noqa: E402
    ExecutorOutput,
    ExecutorReport,
    determine_outcome,
)
from minitap.mobile_use.agents.planner.types import Subgoal, SubgoalStatus  # noqa: E402
from minitap.mobile_use.agents.planner.utils import (  # noqa: E402
    get_current_subgoal,
)
from minitap.mobile_use.config import LLM, LLMConfig, LLMConfigUtils, LLMWithFallback  # noqa: E402
from minitap.mobile_use.context import (  # noqa: E402
    DeviceContext,
    DevicePlatform,
    MobileUseContext,
)
from minitap.mobile_use.graph.state import State  # noqa: E402
from minitap.mobile_use.sdk.agent import (  # noqa: E402
    compute_recursion_limit,
    count_action_steps,
)
from minitap.mobile_use.utils.failed_targets import (  # noqa: E402
    append_failure,
    breaker_reason,
    make_key,
    record_failure,
)
from minitap.mobile_use.utils.screen_elements import ScreenElement  # noqa: E402
from minitap.mobile_use.utils.message_content import has_image_block  # noqa: E402
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds  # noqa: E402


def _ctx(platform: DevicePlatform = DevicePlatform.ANDROID) -> MobileUseContext:
    def llm() -> LLMWithFallback:
        return LLMWithFallback(
            provider="openai", model="gpt-5", fallback=LLM(provider="openai", model="o4-mini")
        )

    return MobileUseContext(
        trace_id="test",
        device=DeviceContext(
            host_platform="WINDOWS",
            mobile_platform=platform,
            device_id="test",
            device_width=1220,
            device_height=2712,
        ),
        llm_config=LLMConfig(
            planner=llm(),
            orchestrator=llm(),
            contextor=llm(),
            cortex=llm(),
            executor=llm(),
            utils=LLMConfigUtils(outputter=llm(), hopper=llm()),
        ),
    )


def _elements(n: int = 5) -> list[ScreenElement]:
    return [
        ScreenElement(
            index=i,
            kind="ui",
            label=f"Row {i}",
            bounds=ElementBounds(x=0, y=0, width=100, height=50),
            flags="C",
        )
        for i in range(1, n + 1)
    ]


def _static_prompt_text() -> str:
    """Executor 的英文提示词全文（所有段拼在一起）。

    断言的是「这条契约还在」，不关心它住在哪个文件；拆段后用这个 helper
    保持断言不变。
    """
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((EXECUTOR_PROMPTS_DIR / "en").rglob("*.md"))
    )


# ---------------------------------------------------------------------------
# T12 —— 元素表必须放在局部历史之后（缓存机制的核心）
# ---------------------------------------------------------------------------


def test_t12_screen_elements_come_after_local_history():
    """元素表每轮变 → 必须放在局部历史**之后**。

    放中间的话，它之后的全部历史每轮都要重算，前缀缓存命中率递减
    （实测 53.8%，低于现状 74.6%）。
    """
    msgs = build_executor_messages(
        ctx=_ctx(),
        goal="进入 WLAN 页面",
        screen_elements=_elements(),
        local_messages=[
            AIMessage(content="", tool_calls=[{"name": "tap", "args": {}, "id": "c1"}]),
            ToolMessage(content="Tapped.", tool_call_id="c1"),
        ],
        device_state_lines=["Device date: 2026-10-21"],
    )

    history_idx = next(i for i, m in enumerate(msgs) if isinstance(m, ToolMessage))
    screen_idx = next(
        i for i, m in enumerate(msgs) if isinstance(m, HumanMessage) and "Row 1" in str(m.content)
    )
    assert screen_idx > history_idx, (
        f"元素表位置 {screen_idx} 必须在局部历史 {history_idx} 之后 —— "
        "否则前缀缓存命中率会随步数递减"
    )


def test_t12b_goal_comes_before_history():
    """目标在局部历史之前（它是子目标内恒定的，放前面才能被缓存覆盖）。"""
    msgs = build_executor_messages(
        ctx=_ctx(),
        goal="读出 Wi-Fi 名称",
        screen_elements=_elements(),
        local_messages=[ToolMessage(content="x", tool_call_id="c1")],
        device_state_lines=[],
    )
    goal_idx = next(i for i, m in enumerate(msgs) if "读出 Wi-Fi 名称" in str(m.content))
    history_idx = next(i for i, m in enumerate(msgs) if isinstance(m, ToolMessage))
    assert goal_idx < history_idx


# ---------------------------------------------------------------------------
# T18 —— 截图通道已整条移除：prompt 里**永远**不该有图片块
# ---------------------------------------------------------------------------


def _message_texts(msgs) -> list[str]:
    """每条消息的**文字**（多模态 content 只取 text 块）。"""
    out = []
    for m in msgs:
        content = m.content
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            out.append(
                "\n".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            )
        else:
            out.append(str(content))
    return out


def test_t18_the_prompt_never_carries_an_image_block():
    """🔴 **prompt 里永远不该有图片块**（2026-10-27 截图通道整条移除）。

    下线过两条：先是「每轮附图」那条（`need_screenshot` 举手字段），
    后是「按需工具」那条（`take_screenshot` + 末尾那条「图片感知」消息）。

    ⇒ 现在模型只有两条**文字**通道：UI 元素表 / OCR 块。
    """
    msgs = build_executor_messages(
        ctx=_ctx(),
        goal="进入 WLAN 页面",
        screen_elements=_elements(),
        local_messages=[ToolMessage(content="Tapped.", tool_call_id="c1")],
        device_state_lines=["Device date: 2026-10-21"],
    )

    for i, m in enumerate(msgs):
        assert not has_image_block(getattr(m, "content", None)), f"第 {i} 条消息里不该有图"


def test_t18b_the_last_message_is_the_element_table():
    """最后一条消息是**元素表** —— 屏幕描述到这儿就结束，没有「图片感知」。"""
    msgs = build_executor_messages(
        ctx=_ctx(),
        goal="进入 WLAN 页面",
        screen_elements=_elements(),
        local_messages=[],
        device_state_lines=[],
    )

    texts = _message_texts(msgs)
    assert "[1]" in texts[-1], "最后一条消息应该是元素表"
    assert "Image perception" not in texts[-1]
    assert "图片感知" not in texts[-1]


def test_t18c_the_static_prompt_has_no_image_contract():
    """静态提示词里不许再讲「图会进模型 / 自己喊看截图」那套契约。

    视觉主体是**文字列表**，不是一张图：截图工具、图片感知、内联 data URI
    都必须不存在；模型改用 ``scan_visual_subjects`` + ``vision_index``。
    """
    system = render_executor_system(_ctx())

    for marker in ("take_screenshot", "Image perception", "data:image", '"vision":'):
        assert marker not in system, f"静态提示词里还有旧图片契约：{marker}"
    # 视觉工具的教学仍在工具表里
    assert "scan_visual_subjects" in system
    # 两条**文字**通道都在：元素表（UI 行 + OCR 行）/ 视觉主体列表
    text = _static_prompt_text()
    assert "Screen elements" in text
    assert "OCR rows" in text
    assert "Visual subjects" in text
    assert "vision_index" in text


def test_t18d_tools_table_rows_match_the_registered_tools():
    """工具表的行 == 这次真正注册的工具，一行不多、一行不少。

    手写表会随开关漂移：录屏工具默认**不注册**，视觉工具默认注册。
    表里留着调不到的工具，模型就会照着它喊，dispatch 只能回一条
    "not a known tool"，白烧一轮。
    """
    from minitap.mobile_use.agents.executor.prompt import format_executor_tools_table
    from minitap.mobile_use.prompt_language import PromptLanguage
    from minitap.mobile_use.tools.index import executor_wrappers, get_tools_from_wrappers

    def _table_names(ctx) -> set[str]:
        table = format_executor_tools_table(ctx, language=PromptLanguage.EN)
        return {line.split("`")[1] for line in table.splitlines() if line.startswith("| `")}

    ctx = _ctx()
    actual = {tool.name for tool in get_tools_from_wrappers(ctx, executor_wrappers(ctx))}
    names = _table_names(ctx)
    assert names == actual
    # 默认：视觉开、录屏关 —— 表里必须如实反映
    assert "scan_visual_subjects" in names
    assert "start_video_recording" not in names
    assert "stop_video_recording" not in names

    # 打开录屏之后，两个工具必须出现在表里
    video_ctx = ctx.model_copy(update={"video_recording_enabled": True})
    video_names = _table_names(video_ctx)
    assert {"start_video_recording", "stop_video_recording"} <= video_names

    table = format_executor_tools_table(ctx, language=PromptLanguage.EN)
    # agent_thought 是 ExecutorOutput 的顶层字段，不是动作参数
    assert "`agent_thought`" not in table
    assert "optional `package_name`" in table
    assert "*(none)*" in table

    zh_table = format_executor_tools_table(ctx, language=PromptLanguage.ZH)
    assert "可选 `package_name`" in zh_table
    assert "点击" in zh_table


# ---------------------------------------------------------------------------
# T7 —— outcome 由代码判定
# ---------------------------------------------------------------------------


def test_t7_outcome_is_decided_by_code():
    assert determine_outcome(done=True) == "completed"
    assert determine_outcome(done=False, failures=[{"key": "x"}]) == "failed"
    assert determine_outcome(done=False, step_limit_reached=True) == "blocked"
    assert determine_outcome(done=False) == "failed"


def test_t7b_completed_wins_over_failures():
    """done=True 时即使有失败记录也是 completed（目标确实达成了）。"""
    assert determine_outcome(done=True, failures=[{"key": "x"}]) == "completed"


def test_t7c_step_limit_wins_over_failures():
    """步数超限 → blocked（比 failed 更具体）。"""
    assert (
        determine_outcome(done=False, failures=[{"key": "x"}], step_limit_reached=True) == "blocked"
    )


# ---------------------------------------------------------------------------
# T3 —— 汇报的事实字段由代码直搬
# ---------------------------------------------------------------------------


def test_t3_report_fact_fields_are_typed():
    report = ExecutorReport(
        outcome="failed",
        summary="点不到那个按钮",
        failed_targets=[],
    )
    assert report.outcome == "failed"
    assert report.failed_targets == []


def test_t3b_dispatcher_returns_rejection_reason():
    """dispatcher 失败时必须给出**可回灌给模型**的理由。

    旧架构退回 LLM 重转写就行；新架构的 LLM 就是决策者，
    没有理由它会输出同样的坏动作 → 死循环。
    """
    calls, reason = dispatch_decisions_with_reason(
        decisions=json.dumps([{"action": "teleport"}]),
        default_thought="thinking",
        tools=[],
    )
    assert calls is None
    assert reason is not None
    assert "teleport" in reason
    assert "not a known tool" in reason


def test_t3c_dispatcher_reason_is_actionable():
    """理由要指出**缺了什么**，不能只说「不合法」。"""
    _, reason = dispatch_decisions_with_reason(
        decisions=json.dumps([{"action": "tap"}]),
        default_thought="thinking",
        tools=[],
    )
    # 没有工具时先撞「未知工具」；这里只验证理由非空且含可操作信息
    assert reason


def test_t3d_dispatcher_rejects_missing_arg():
    """缺必填参数 → 理由里要点名缺哪个。"""
    from minitap.mobile_use.tools.index import EXECUTOR_WRAPPERS_TOOLS, get_tools_from_wrappers

    tools = get_tools_from_wrappers(_ctx(), EXECUTOR_WRAPPERS_TOOLS)
    calls, reason = dispatch_decisions_with_reason(
        decisions=json.dumps([{"action": "tap"}]),
        default_thought="thinking",
        tools=tools,
    )
    assert calls is None
    assert reason is not None
    assert "target" in reason


def test_t3e_dispatcher_backward_compatible():
    """旧的 `dispatch_decisions` 签名和行为不能变（16/16 命中的路径）。"""
    from minitap.mobile_use.tools.index import EXECUTOR_WRAPPERS_TOOLS, get_tools_from_wrappers

    tools = get_tools_from_wrappers(_ctx(), EXECUTOR_WRAPPERS_TOOLS)
    calls = dispatch_decisions(
        json.dumps([{"action": "tap", "target": {"index": 7}}]),
        "thinking",
        tools,
    )
    assert calls is not None
    assert calls[0]["name"] == "tap"


def test_t3g_empty_actions_is_not_reported_as_parse_failure():
    """**空 actions ≠ 解析失败** —— 回灌给模型的理由必须区分。

    真机实测（2026-10-21）：Executor 上一轮动作失败后，第二轮返回了空
    `actions`，dispatcher 却报「could not be parsed as a JSON array」——
    模型的格式**没问题**（它就是返回了空数组），被告知「格式错」会让它
    去改格式，而它本来就没格式问题，只能原地打转。

    ⇒ 空数组要说「你没决定任何动作」，解析失败才说「格式不对」。
    """
    calls, reason = dispatch_decisions_with_reason(
        decisions=json.dumps([]),
        default_thought="thinking",
        tools=[],
    )
    assert calls is None
    assert reason is not None
    assert "empty action list" in reason
    assert "could not be parsed" not in reason


def test_t3h_real_parse_failure_still_says_parse_failure():
    """真正的解析失败，理由不能被上面那条误伤。"""
    calls, reason = dispatch_decisions_with_reason(
        decisions="not json at all",
        default_thought="thinking",
        tools=[],
    )
    assert calls is None
    assert reason is not None
    assert "could not be parsed" in reason


def test_t3f_rejection_code_is_short_and_leaky_free():
    """遥测只放短代码，不放理由原文（理由里可能有屏幕文字 = 用户数据）。"""
    code = _rejection_code("'teleport' is not a known tool; available tools are: tap, back")
    assert code == "unknown_tool"
    assert len(code) < 32
    assert "teleport" not in code


# ---------------------------------------------------------------------------
# T14 —— 解析失败不能杀掉整个任务（真机踩出来的）
# ---------------------------------------------------------------------------


def test_t14_executor_prompt_demands_a_tool_call():
    """prompt 必须明确说「必须调用工具」，否则模型会写散文。

    真机实测（2026-10-21，Redmi Turbo 4）：Executor 明明已经完成任务，
    却输出了一段散文（"The WLAN page is now open... done: true"）而**不调用工具**。
    那段文字里没有 `{`，连 JSON 兜底提取都做不了 → 解析失败。
    而 `with_fallback` 兜不住 —— 本机 override 里 fallback 和主模型是同一个模型，
    重试同样失败 → **整个任务挂掉**。
    """
    text = _static_prompt_text()
    assert "call the" in text.lower()
    assert "plain text" in text.lower() or "as plain text" in text.lower()


def test_t14b_executor_prompt_shows_a_json_example():
    """要有 JSON 示例 —— 表格描述不如示例管用（模型照着抄）。"""
    text = _static_prompt_text()
    assert '"actions"' in text
    assert '"agent_thought"' in text
    # done 的形态也要有示例：actions 为空 + done=true
    assert '"done": true' in text


def test_t14c_executor_degrades_instead_of_raising():
    """解析失败 → 重试 → 仍失败 → 降级成空动作，**不抛异常**。

    抛出去会让 `with_fallback` 切到 fallback 模型，而本机 fallback 和主模型
    相同 → 同样失败 → 任务顶层崩掉。降级成空动作则让 Reporter 看到、
    汇报成 failed、Cortex 换招重来 —— 任务能继续。
    """
    import inspect

    from minitap.mobile_use.agents.executor.executor import ExecutorNode

    src = inspect.getsource(ExecutorNode._invoke)
    # _invoke 捕获所有异常并返回 None，不往上抛
    assert "except Exception" in src
    assert "return None" in src

    # _decide 里解析失败有降级分支
    decide_src = inspect.getsource(ExecutorNode._decide)
    assert "degrading to an empty" in decide_src


# ---------------------------------------------------------------------------
# T15 —— 汇报构建（纯代码，Cortex 入口调用）
# ---------------------------------------------------------------------------


def test_t15b_code_summary_produces_usable_text():
    """代码摘要要真的能用 —— 不能是空串或「模型不可用」这种误导措辞。"""
    from minitap.mobile_use.agents.reporter.reporter import code_summary

    for outcome, failed in (("completed", 0), ("failed", 2), ("blocked", 0), ("failed", 0)):
        s = code_summary(outcome=outcome, failed_count=failed)
        assert s and len(s) > 8
        # 免 LLM 模式下这句话是错的（模型不是"不可用"，是没被调用）
        assert "不可用" not in s


def test_t15c_evidence_can_be_picked_without_llm():
    """免 LLM 时 evidence 也要能从执行日志里挑出来。"""
    from langchain_core.messages import AIMessage, ToolMessage

    from minitap.mobile_use.agents.reporter.reporter import pick_evidence

    msgs = [
        AIMessage(content="", tool_calls=[{"name": "tap", "args": {"index": 27}, "id": "c1"}]),
        ToolMessage(
            content="Tapped. The action bar title reads WLAN and the network list is visible.",
            tool_call_id="c1",
        ),
    ]
    log = "tap -> Tapped. The action bar title reads WLAN and the network list is visible."
    ev = pick_evidence(None, log_text=log, outcome="completed", messages=msgs)
    assert ev is not None
    assert "WLAN" in ev


# ---------------------------------------------------------------------------
# T19 —— 失败账本的键必须认 ocr_index
# ---------------------------------------------------------------------------


def test_t19_target_key_uses_ocr_index_when_no_table():
    """无元素表时（iOS / 云设备），模型填的是 ``ocr_index``。

    账本键如果只读 ``index``，那条路径下 index 恒为 None → 键退化成
    ``text=...`` 或 ``"unknown"``：
    * 熔断账本把「同一个 ocr_index 连着失败」判成「换了目标」→ 连续失败计数失效；
    * 失败清单渲染成 ``[unknown] "点赞"`` → 模型看不到是第几行。
    """
    from minitap.mobile_use.tools.types import Target
    from minitap.mobile_use.tools.utils import target_key, target_label

    k = target_key(Target(ocr_index=7, text="点赞"))
    assert "index=7" in k
    assert "unknown" not in k
    assert "[7]" in target_label(Target(ocr_index=7, text="点赞"))


def test_t19b_target_key_prefers_index_when_present():
    """有元素表时（index 和 ocr_index 都填），键用 index。

    两条路径下键的形状不同也没关系 —— 熔断账本是**当前屏内**的即时保护，
    不会跨屏比较。
    """
    from minitap.mobile_use.tools.types import Target
    from minitap.mobile_use.tools.utils import target_key

    k = target_key(Target(index=3, ocr_index=9, text="x"))
    assert "index=3" in k
    assert "index=9" not in k


# ---------------------------------------------------------------------------
# T5 —— 失败事实跨子目标保留
# ---------------------------------------------------------------------------


def test_t5_failure_facts_survive_across_subgoals():
    """append-only 的事实账本跨子目标保留；熔断账本不保留（它成功即清）。"""
    key = make_key(1, "com.x:id/a", "A")

    # 熔断账本：换键即替换、成功即清 —— 事实拿不到
    breaker = record_failure(None, key=key, index=1, resource_id="com.x:id/a", label="A")
    breaker = record_failure(breaker, key=key, index=1, resource_id="com.x:id/a", label="A")
    other = make_key(2, "com.x:id/b", "B")
    breaker = record_failure(breaker, key=other, index=2, resource_id="com.x:id/b", label="B")
    assert len(breaker) == 1
    assert breaker[0].key == other
    assert breaker_reason(breaker, key) is None  # 旧的熔断也失效了

    # 事实账本：只增不减
    facts = append_failure(
        None, key=key, index=1, resource_id="com.x:id/a", label="A", reason="gone"
    )
    facts = append_failure(
        facts, key=other, index=2, resource_id="com.x:id/b", label="B", reason="also gone"
    )
    assert len(facts) == 2
    assert facts[0]["key"] == key


def test_t5b_failure_facts_render_for_the_model():
    """事实清单要渲染成给模型看的一行（让它避开试过的目标）。"""
    from minitap.mobile_use.utils.failed_targets import render_failure_facts

    facts = append_failure(
        None, key="index=48|id=x", index=48, resource_id="x", label="点赞", reason="gone"
    )
    line = render_failure_facts(facts)
    assert "48" in line
    assert "点赞" in line
    assert render_failure_facts([]) == ""


# ---------------------------------------------------------------------------
# T11 —— dispatcher 只吃数组
# ---------------------------------------------------------------------------


def test_t11_dispatcher_rejects_single_object():
    """单个对象（不是数组）→ dispatcher 认不出来。"""
    from minitap.mobile_use.tools.index import EXECUTOR_WRAPPERS_TOOLS, get_tools_from_wrappers

    tools = get_tools_from_wrappers(_ctx(), EXECUTOR_WRAPPERS_TOOLS)
    calls = dispatch_decisions(
        json.dumps({"action": "tap", "target": {"index": 7}}),
        "thinking",
        tools,
    )
    assert calls is None


def test_t11b_single_element_array_works():
    """单元素数组能直连 —— Executor 每轮只出一个动作时的形态。"""
    from minitap.mobile_use.tools.index import EXECUTOR_WRAPPERS_TOOLS, get_tools_from_wrappers

    tools = get_tools_from_wrappers(_ctx(), EXECUTOR_WRAPPERS_TOOLS)
    calls = dispatch_decisions(
        json.dumps([{"action": "tap", "target": {"index": 7}}]),
        "thinking",
        tools,
    )
    assert calls is not None
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# T13 —— 感知通道没丢
# ---------------------------------------------------------------------------


def test_t13_executor_output_has_no_picture_field():
    """🔴 2026-10-23：图片通道整条下线 ⇒ 输出 schema 里没有「我要看图」。

    模型看不到的东西，就不该让它以为有：只要 `need_screenshot` 还在 schema
    里，模型就会学会举手要图，而框架已经没有图可以给了。
    """
    assert "need_screenshot" not in ExecutorOutput.model_fields
    # 剩下的字段就是全部。
    #
    # ⚠️ 2026-10-26：多了 `failed` / `failed_reason` —— 「做不到」的出口。
    # 在此之前 executor 唯一能结束子目标的方式是 `done: true`，而 `done`
    # 的语义是「目标达成」、`determine_outcome` 把它**无条件**映射成
    # ``completed`` ⇒ **「做不到」在契约里不可表达** ⇒ 模型只能一直试
    # ⇒ 原地打转 ⇒ 撞 `recursion_limit` 崩掉（真机 execution 880，70 步）。
    assert set(ExecutorOutput.model_fields) == {
        "actions",
        "agent_thought",
        "done",
        "done_reason",
        "failed",
        "failed_reason",
    }


def test_t13b_executor_prompt_has_no_picture_channel_but_keeps_wake_screen():
    """截图通道**整条没了**，但息屏唤醒必须留着。

    息屏唤醒是另一回事（真机教训：30 分钟息屏让一个任务从 30 步涨到 51 步），
    它和图片通道无关，不能一起删掉 —— 现在模型只有两条文字通道，
    息屏时两条都读不到东西，「先唤醒」就更得保住。

    2026-10-27：按需截图**工具**也下线了 ⇒ 提示词里不许再有 `take_screenshot`，
    `need_screenshot` 这种「举手要图」的字段同样不许有。
    """
    text = _static_prompt_text()
    # 通道没回来：模型不该再学会「举手要图」
    assert "need_screenshot" not in text
    # 工具也没了：模型没有可喊的名字
    assert "take_screenshot" not in text
    # 息屏唤醒必须留着
    assert "wake_screen" in text


def test_t13c_executor_prompt_has_no_tool_schema():
    """Executor 的提示词 **不能**含工具的 JSON schema（那是 4,008 token 且不进缓存键）。"""
    text = _static_prompt_text()
    for marker in ("bind_tools", '"input_schema"', '"properties"', "additionalProperties"):
        assert marker not in text, f"Executor 的提示词不该出现 {marker}"


def test_t13d_executor_prompt_has_element_index_contract():
    """索引契约的说明书必须在（`[i]` 直接用，不抄 id 和坐标）。"""
    text = _static_prompt_text()
    assert "1-based" in text
    assert "index" in text


# ---------------------------------------------------------------------------
# T2 —— 局部历史生命周期
# ---------------------------------------------------------------------------


def test_t2_state_has_executor_fields():
    """State 必须有这 4 个 executor 字段，名字和契约一致。"""
    from minitap.mobile_use.graph.state import State

    fields = State.model_fields
    for name in (
        "executor_goal",
        "executor_done",
        "executor_steps",
        "executor_failures",
    ):
        assert name in fields, f"State 缺少字段 {name}"
    assert "executor_report" not in fields, (
        "汇报不再是 state 字段：Cortex 在入口用 build_executor_report 构建"
    )


def test_t2b_executor_steps_default_is_zero():
    from minitap.mobile_use.graph.state import State

    assert State.model_fields["executor_steps"].default == 0


# ---------------------------------------------------------------------------
# T4 —— 熔断
# ---------------------------------------------------------------------------


def test_t4_step_limit_is_configurable_and_generous():
    """步数上限**默认不限**（2026-10-22 改）。

    原来是 12。两条实测证据说明它是错的：

    * 它**腰斩正常子目标** —— 真机跑一个「往上滚到结果页顶部，确认第一条是否带
      『广告』标识」的子目标，12 步不够 → 被截断成 ``outcome=blocked``
      → 换招 → 重发。**那个子目标本来是做得完的**。
    * 它**掩盖了真因** —— 执行器原地打转时，它把「打转」腰斩成「这个子目标
      走不通」，于是换招重来、清空局部历史、更瞎（限制-症状倒挂）。

    ⇒ 现在默认不限（None）。要恢复上限：把它改成整数（代码路径都还在）。
    """
    assert MAX_STEPS_PER_SUBGOAL is None, "默认必须不限子目标步数"


def test_t4b_default_limit_means_outcome_is_never_blocked():
    """T4b：默认不限 ⇒ ``outcome`` 不会是 ``blocked``（那条分支只在配了上限时可达）。"""
    from minitap.mobile_use.agents.reporter.reporter import determine_outcome_for

    state = State(
        **{
            **_initial_state(),
            # 步数远超任何可能的子目标上限
            "executor_steps": 999,
            "executor_done": True,
            "executor_failures": [],
        }
    )
    assert determine_outcome_for(state) == "completed", "默认不该有 blocked"


def test_t4b_breaker_blocks_repeated_same_target():
    """同一目标连着失败 2 次 → 工具拒绝执行。"""
    key = make_key(1, "com.x:id/a", "A")
    streak = record_failure(None, key=key, index=1, resource_id="com.x:id/a", label="A")
    streak = record_failure(streak, key=key, index=1, resource_id="com.x:id/a", label="A")
    reason = breaker_reason(streak, key)
    assert reason is not None
    assert "failed 2 times" in reason


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def test_reporter_falls_back_to_executor():
    """reporter 没单独配时回落 executor（否则 22 处 LLMConfig 构造点全炸）。"""
    from minitap.mobile_use.config import LLMConfig as Cfg

    llm = LLMWithFallback(
        provider="openai", model="gpt-5-nano", fallback=LLM(provider="openai", model="gpt-5-mini")
    )
    cfg = Cfg(
        planner=llm,
        orchestrator=llm,
        contextor=llm,
        cortex=llm,
        executor=llm,
        reporter=None,
        utils=LLMConfigUtils(outputter=llm, hopper=llm),
    )
    assert cfg.get_agent("reporter") == cfg.get_agent("executor")


def test_vision_falls_back_to_executor_and_can_be_overridden():
    """vision 没单独配 → 回落 executor；配了就用它自己的。

    两个方向都要锁：回落保证老 override（没有 vision 键）行为不变；
    覆盖保证「换 grounding 更强的模型」这条配置真的生效。
    """
    from minitap.mobile_use.config import LLMConfig as Cfg

    executor = LLMWithFallback(
        provider="openai", model="gpt-5-nano", fallback=LLM(provider="openai", model="gpt-5-mini")
    )
    vision = LLMWithFallback(
        provider="siliconflow",
        model="Qwen/Qwen3-VL-32B-Instruct",
        fallback=LLM(provider="siliconflow", model="Qwen/Qwen3-VL-8B-Instruct"),
    )
    common = dict(
        planner=executor,
        orchestrator=executor,
        contextor=executor,
        cortex=executor,
        executor=executor,
        utils=LLMConfigUtils(outputter=executor, hopper=executor),
    )

    assert Cfg(**common).get_agent("vision") == executor
    assert Cfg(**common, vision=vision).get_agent("vision") == vision


# ---------------------------------------------------------------------------
# T16~T18 —— 屏幕描述只有一种形态：统一元素表（UI 行 + OCR 行）
# ---------------------------------------------------------------------------
#
# 2026-10-28：iOS / 云设备的层级树形状不认识，但 OCR 结果仍能建表（只剩
# OCR 行）。屏幕描述因此和 Android 一样是一张统一索引的表，模型只填 `index`；
# 原来那条「无元素表 → 教 ocr_index」的 legacy OCR 块已经删除。


def _ocr(n: int = 3) -> list:
    from minitap.mobile_use.services.ocr.types import OcrText

    return [
        OcrText(
            text=f"Text {i}",
            score=0.95,
            bounds=ElementBounds(x=0, y=0, width=100, height=40),
        )
        for i in range(1, n + 1)
    ]


def _prompt_text(msgs) -> str:
    return "\n".join(str(m.content) for m in msgs)


def test_t16_ocr_only_table_is_still_one_indexed_table():
    """只有 OCR 行时（iOS / 云设备），屏幕描述仍是同一张表、同一套 `index`。"""
    from minitap.mobile_use.agents.executor.prompt import has_element_table
    from minitap.mobile_use.utils.screen_elements import build_screen_elements

    assert has_element_table(None) is False
    assert has_element_table([]) is False

    elements = build_screen_elements(tree_rows=None, ocr_results=_ocr())
    assert has_element_table(elements) is True

    msgs = build_executor_messages(
        ctx=_ctx(DevicePlatform.IOS),
        goal="点开设置",
        screen_elements=elements,
        local_messages=[],
        device_state_lines=[],
    )
    text = _prompt_text(msgs)

    # 表头和 OCR 行都在，编号从 1 开始
    assert "Screen elements, one per line" in text
    assert '[1] "Text 1"' in text
    # 不再教 ocr_index —— 一套 index 走到底
    assert "ocr_index" not in text


def test_t17_element_table_prompt_still_uses_index():
    """有元素表时：target 示例仍然是 `index`。"""
    msgs = build_executor_messages(
        ctx=_ctx(),
        goal="点开设置",
        screen_elements=_elements(),
        local_messages=[],
        device_state_lines=[],
    )
    text = _prompt_text(msgs)

    # 元素表照旧渲染（表头、行格式、编号都没动）
    assert "Screen elements, one per line" in text
    assert '[1] "Row 1"' in text

    # target 示例仍然是 index
    assert '"index": 7' in text
    assert "ocr_index" not in text


def test_t18_both_shapes_assemble_without_raising():
    """有表 / 空表都要能正常组装（不抛异常）。"""
    for elements in (None, [], _elements()):
        msgs = build_executor_messages(
            ctx=_ctx(),
            goal="g",
            screen_elements=elements,
            local_messages=[],
            device_state_lines=[],
        )
        assert msgs, f"screen_elements={elements!r} 时不该组装出空消息列表"
        assert isinstance(_prompt_text(msgs), str)


# ---------------------------------------------------------------------------
# T20 / T21 / T22：recursion_limit 换算 + 步数口径（方案 §11.5 M12 / M13）
# ---------------------------------------------------------------------------
#
# 这几个用例跑的是**真实的图**（`graph/graph.py` 的拓扑、真实的条件边、真实的
# `State`、真实的 `ExecutorNode`），只把需要 LLM / 设备的节点体换成脚本化替身。
#
# 为什么替身跑出来的数就是真机上的数：super-step 的数量只由**拓扑**决定
# （每个 super-step 调度哪些节点），与节点体里跑 LLM 还是跑 adb 无关。
# 探针 `scripts/probe_superstep_budget.py` 用 checkpoint 的 `metadata["step"]`
# 交叉验证过「super-step 数 == 节点执行次数」（这张图是纯链，一个 super-step
# 正好跑一个节点）。
#
# 实测模型（探针，误差 0；2026-10-29 移除 Summarizer / convergence / reporter 后更新）：
#     super_steps = 2 + S × (3K + 5)
#         S = 子目标数，K = 每个子目标的动作步数
#     动作步数 E = S × (K + 1)（每个子目标最后还有一轮「我做到了」的收尾）
#   ⇒ super_steps = 2 + 3E + 2S，而 S ≤ E ⇒ 最坏 2 + 5E（每个动作步自成子目标）
#
# 实际最坏系数是 5：K=0 时每个子目标要 5 个固定 super-step，和动作步数量无关。
# `SUPER_STEPS_PER_ACTION_STEP = 9` 是刻意留的余量，比实际最坏更保守。


def _plan(n: int) -> list[Subgoal]:
    return [
        Subgoal(id=f"sg{i}", description=f"subgoal {i}", status=SubgoalStatus.NOT_STARTED)
        for i in range(n)
    ]


def _stub_planner(n_subgoals: int):
    class _Planner:
        def __init__(self, ctx):
            self.ctx = ctx

        async def __call__(self, state):
            return {"subgoal_plan": _plan(n_subgoals)}

    return _Planner


def _stub_contextor():
    class _Contextor:
        def __init__(self, ctx):
            self.ctx = ctx

        async def __call__(self, state):
            # 真 contextor **每个动作步**写 1 条 thought
            # （`agents/contextor/contextor.py:191`）—— 这正是「thought 条数
            # 随步数暴涨」的来源，照抄它。
            return await state.asanitize_update(
                ctx=self.ctx,
                update={"agents_thoughts": ["grabbed the screen"]},
                agent="contextor",
            )

    return _Contextor


def _stub_cortex():
    class _Cortex:
        def __init__(self, ctx):
            self.ctx = ctx

        async def __call__(self, state):
            # 照抄真 Cortex 的语义：``executor_done`` 表示「有一份没被消费的
            # 汇报」——是 → 判完成并提名给 plan_manager；否 → 下发目标。
            #
            # 汇报构建本身（`build_executor_report`）是纯代码，这里同样用一个
            # 纯代码的替身：只要 done，就当作 completed。
            current = get_current_subgoal(state.subgoal_plan)
            update: dict = {"latest_ui_hierarchy": None, "executor_done": False}
            if state.executor_done:
                update["complete_subgoals_by_ids"] = [current.id] if current else []
                thought = "the subgoal looks done"
            else:
                update["executor_goal"] = "read the wifi name"
                update["executor_messages"] = [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
                update["executor_steps"] = 0
                update["complete_subgoals_by_ids"] = []
                thought = "issued the goal"
            update["agents_thoughts"] = [thought]
            return await state.asanitize_update(ctx=self.ctx, update=update, agent="cortex")

    return _Cortex


def _stub_tool_node():
    class _ToolNode:
        def __init__(self, tools, messages_key, trace_id):
            self.tools = tools
            self.messages_key = messages_key
            self.trace_id = trace_id

        async def __call__(self, state):
            # 真节点会真的去点屏幕 —— 这里只回一条结果，不碰设备。
            return {"executor_messages": [ToolMessage(content="ok", tool_call_id="x")]}

    return _ToolNode


def _stub_decide(steps_per_subgoal: int):
    """脚本化的 Executor 决策：前 K 轮给动作，之后说「我做到了」。

    只替换 LLM 那一层，`ExecutorNode.__call__` 里**计数**的逻辑是真的。
    """

    async def _decide(self, *, state, goal, stall_notice: str = ""):
        if state.executor_steps >= steps_per_subgoal:
            return ExecutorOutput(
                actions=[],
                agent_thought="the goal is reached",
                done=True,
                done_reason="read it off the screen",
            )
        return ExecutorOutput(
            actions=[{"action": "tap", "target": {"index": 1}}],
            agent_thought="tap the row",
        )

    return _decide


def _initial_state() -> dict:
    """初始 state —— 照抄 `sdk/agent.py:1213` 的 `_get_graph_state`。"""
    return {
        "initial_goal": "read the wifi name",
        "subgoal_plan": [],
        "latest_ui_hierarchy": None,
        "latest_ocr_results": None,
        "latest_screen_elements": None,
        "focused_app_info": None,
        "device_date": None,
        "complete_subgoals_by_ids": [],
        "agents_thoughts": [],
        "remaining_steps": 400,
        "executor_messages": [],
        "scratchpad": {},
    }


async def _run_real_graph(
    monkeypatch: MonkeyPatch,
    *,
    n_subgoals: int,
    steps_per_subgoal: int,
    recursion_limit: int,
) -> dict:
    """跑一遍**真实的图**，返回 {"super_steps", "last", "error"}。

    只把需要 LLM / 设备的节点体换成脚本化替身；拓扑、条件边、State、
    ExecutorNode 全是真的。super-step 数 = 节点执行次数（这张图是纯链，
    一个 super-step 跑一个节点；探针用 checkpoint 的 step 交叉验证过）。
    """
    import minitap.mobile_use.graph.graph as gg

    monkeypatch.setattr(gg, "PlannerNode", _stub_planner(n_subgoals))
    monkeypatch.setattr(gg, "ContextorNode", _stub_contextor())
    monkeypatch.setattr(gg, "CortexNode", _stub_cortex())
    monkeypatch.setattr(gg, "ExecutorToolNode", _stub_tool_node())
    monkeypatch.setattr(ExecutorNode, "_decide", _stub_decide(steps_per_subgoal))

    graph = await gg.get_graph(_ctx())
    super_steps = 0
    last: dict | None = None
    error: str | None = None
    try:
        async for mode, payload in graph.astream(
            input=_initial_state(),
            config={"recursion_limit": recursion_limit},
            stream_mode=["updates", "values"],
        ):
            if mode == "updates":
                super_steps += len(payload)
            elif mode == "values":
                last = payload
    except GraphRecursionError as e:
        error = str(e)
    return {"super_steps": super_steps, "last": last, "error": error}


def test_t20_recursion_limit_covers_the_action_step_budget():
    """T20：换算出来的 `recursion_limit` 足够跑完对应的动作步数。

    实测模型（`scripts/probe_superstep_budget.py`）：
        super_steps = 2 + 3E + 2S，E = 动作步数，S = 子目标数
    最坏情况是「每个动作步都自成子目标」（S = E）⇒ 2 + 5E。
    换算必须覆盖这个最坏情况 —— 给少了任务会在中途崩掉。
    """
    for max_steps in (1, 4, 10, 30, 100, 400):
        limit = compute_recursion_limit(max_steps)
        worst_case_super_steps = 2 + 5 * max_steps
        assert limit >= worst_case_super_steps, (
            f"max_steps={max_steps} → recursion_limit={limit}，"
            f"但最坏情况要 {worst_case_super_steps} 个 super-step"
        )


def test_t20b_recursion_limit_is_monotonic_and_never_below_the_floor():
    """T20：换算是单调的，而且永远留得下起跑预算。"""
    assert compute_recursion_limit(0) < compute_recursion_limit(1) < compute_recursion_limit(10)
    # 负数（`max_steps` 没有下界校验）不能算出更小的上限
    assert compute_recursion_limit(-5) == compute_recursion_limit(0)
    # 起跑预算：0 个动作步也要够跑完「建计划 + 起跑 + 收尾」
    assert compute_recursion_limit(0) >= 32


@pytest.mark.asyncio
async def test_t20c_converted_recursion_limit_finishes_the_task(monkeypatch):
    """T20：换成算出来的 `recursion_limit` 之后，任务能**跑完**。

    脚本：2 个子目标 × 每个 3 个动作步 = 6 个动作步（+ 每子目标 1 轮收尾
    = 8 个 executor 步），实测要 30 个 super-step
    （Summarizer / convergence / reporter 移除前是 45）。
    """
    result = await _run_real_graph(
        monkeypatch,
        n_subgoals=2,
        steps_per_subgoal=3,
        recursion_limit=compute_recursion_limit(6),
    )
    assert result["error"] is None, f"不该撞 recursion_limit：{result['error']}"
    assert result["last"]["total_steps"] == 8


@pytest.mark.asyncio
async def test_t20d_unconverted_recursion_limit_crashes_the_task(monkeypatch):
    """T20 反向验证：**不换算**的话，任务会中途崩。

    这就是 M12 要修的东西 —— 同一个 `max_steps` 直接当 `recursion_limit` 用，
    在新拓扑下根本跑不完（8 个动作步要 30 个 super-step）。
    """
    result = await _run_real_graph(
        monkeypatch,
        n_subgoals=2,
        steps_per_subgoal=3,
        recursion_limit=8,
    )
    assert result["error"] is not None
    assert "Recursion limit" in result["error"]
    assert result["last"] is None or result["last"]["total_steps"] < 8


@pytest.mark.asyncio
async def test_t21_total_steps_is_not_reset_when_the_subgoal_changes(monkeypatch):
    """T21：换目标后 `total_steps` **继续累加**（`executor_steps` 会被归零）。

    2 个子目标 × 每个 3 个动作步：
      * `executor_steps` 换目标时归零 → 结束时只剩最后一个子目标的 4
      * `total_steps` 跨子目标累加 → 结束时是 8
    """
    result = await _run_real_graph(
        monkeypatch,
        n_subgoals=2,
        steps_per_subgoal=3,
        recursion_limit=compute_recursion_limit(8),
    )
    assert result["error"] is None
    last = result["last"]

    # 最后一个子目标跑完 → executor_steps 只剩「本子目标内」的步数
    assert last["executor_steps"] == 4
    # 累计值跨子目标继续累加
    assert last["total_steps"] == 8
    assert last["total_steps"] > last["executor_steps"]


def test_t21b_total_steps_defaults_to_zero_and_is_a_state_field():
    """T21：`total_steps` 是 state 字段，默认 0。"""
    assert "total_steps" in State.model_fields
    assert State.model_fields["total_steps"].default == 0


def test_t22_steps_count_is_action_steps_not_thought_count():
    """T22：`steps_count` 取的是**动作步数**，不是 thought 条数。

    构造一个「thought 条数 ≠ 动作步数」的 state：12 条 thought vs 5 个动作步。
    """
    state = State(
        **{**_initial_state(), "agents_thoughts": [f"thought {i}" for i in range(12)]},
        total_steps=5,
    )
    assert len(state.agents_thoughts) == 12
    assert count_action_steps(state) == 5
    # 老口径（len(agents_thoughts)）会返回 12 —— 正是要修掉的失真
    assert count_action_steps(state) != len(state.agents_thoughts)


def test_t22b_count_action_steps_handles_no_state():
    """T22：图没跑出结果（state 是 None）时不能炸。"""
    assert count_action_steps(None) == 0


@pytest.mark.asyncio
async def test_t22c_reported_steps_count_matches_the_action_steps(monkeypatch):
    """T22：真跑一遍，上报的 `steps_count` 应当等于动作步数，而不是 thought 条数。"""
    result = await _run_real_graph(
        monkeypatch,
        n_subgoals=2,
        steps_per_subgoal=3,
        recursion_limit=compute_recursion_limit(8),
    )
    assert result["error"] is None
    state = State(**result["last"])

    assert count_action_steps(state) == 8
    # thought 条数（Contextor 每个动作步 1 条 + 监督节点每个子目标各几条）
    # ≠ 动作步数 —— 这正是老口径失真的地方
    assert len(state.agents_thoughts) != count_action_steps(state)


# ---------------------------------------------------------------------------
# T23 / T24 —— L5：工具结果截断（只对 executor 局部历史这一路）
# ---------------------------------------------------------------------------
#
# 为什么只截这一路：局部历史是 prompt 里唯一随步数线性增长的一段，
# 而 Reporter 走的是 ``state.executor_messages``（原对象），守
# 「事实不能压」的纪律（``utils/thoughts_compaction.py`` 的模块文档）——
# 截断只发生在**发给 Executor LLM 的那份副本**上。


def test_t23_long_tool_result_is_truncated_but_errors_stay_verbatim():
    """L5/T23：超长的**成功**结果被截断；**错误结果逐字保留**。

    报错那部分模型要靠它判断「为什么失败、下一步换哪」—— 压了就等于让它盲猜。
    """
    from minitap.mobile_use.agents.executor.prompt import (
        MAX_TOOL_RESULT_TOKENS,
        estimate_tokens,
        truncate_tool_result,
    )

    long_body = "The quick brown fox jumps over the lazy dog. " * 40
    long_ok = f"Successfully read note 'k'. 'k' note full content: {long_body}"
    truncated = truncate_tool_result(long_ok)

    # 截了，而且截到预算以内
    assert truncated != long_ok
    assert estimate_tokens(truncated) <= MAX_TOOL_RESULT_TOKENS
    # 首句是结论，必须留着
    assert truncated.startswith("Successfully read note 'k'.")

    # 错误结果：一个字节都不许动
    long_error_text = f"Failed to tap the row: {long_body}"
    msgs = build_executor_messages(
        ctx=_ctx(),
        goal="g",
        screen_elements=_elements(),
        local_messages=[
            ToolMessage(content=long_error_text, tool_call_id="c1", status="error"),
        ],
        device_state_lines=[],
    )
    sent = next(m for m in msgs if isinstance(m, ToolMessage))
    assert sent.content == long_error_text


def test_t23b_truncation_keeps_the_message_structure_intact():
    """L5/T23：截断只动内容 —— tool_call_id / status / 名字都不能丢。

    tool_call_id 是 OpenAI 兼容端点把 AIMessage 和 ToolMessage 配对的依据，
    动了会 400（和致命 bug 3 那一类）。

    ⚠️ 这里用的是 `tap` 而不是 `read_note` —— 后者是**载荷类工具**，
    按设计不截断（见 T23c）。
    """
    from minitap.mobile_use.agents.executor.prompt import truncate_local_history

    long_body = "x" * 4000
    original = ToolMessage(
        content=f"Successfully read note 'k'. full content: {long_body}",
        tool_call_id="call-42",
        name="tap",
        status="success",
    )
    (truncated,) = truncate_local_history([original])

    assert truncated is not original, "不能就地改 state 里的原对象"
    assert truncated.tool_call_id == "call-42"
    assert truncated.name == "tap"
    assert truncated.status == "success"
    assert len(str(truncated.content)) < len(original.content)
    # 原对象没被动过（Reporter 读的是它）
    assert original.content == f"Successfully read note 'k'. full content: {long_body}"


def test_t23c_payload_tools_are_never_truncated():
    """L5/T23c：**载荷类工具**的结果逐字保留 —— 截断等于丢信息。

    `read_note` 返回的就是模型要读的那条笔记本身；`list_notes` 返回全部
    key；`focus_and_input_text` / `focus_and_clear_text` 返回输入框的最终
    内容（模型要拿它确认「到底打进去没有」）。截掉这些 = 静默丢事实。

    ⚠️ 局部历史里的 `ToolMessage` **大多没有 `name` 字段**（工具自己建的时候
    没传），所以豁免必须能靠**前一条 AIMessage 的 tool_calls** 反查工具名 ——
    这条用例就是照那个形状写的。
    """
    from minitap.mobile_use.agents.executor.prompt import truncate_local_history

    long_body = "x" * 4000
    content = f"Successfully read note 'k'. 'k' note full content: {long_body}"

    ai = AIMessage(
        content="",
        tool_calls=[{"name": "read_note", "args": {"key": "k"}, "id": "call-7"}],
    )
    tool = ToolMessage(content=content, tool_call_id="call-7", status="success")

    (kept_ai, kept_tool) = truncate_local_history([ai, tool])

    assert kept_tool is tool, "载荷类工具不该被换成副本"
    assert str(kept_tool.content) == content, "载荷被截了 —— 事实丢了"

    # 对照组：同一个长内容，但工具不是载荷类 → 该截
    ai_other = AIMessage(
        content="",
        tool_calls=[{"name": "tap", "args": {}, "id": "call-8"}],
    )
    tool_other = ToolMessage(content=content, tool_call_id="call-8", status="success")
    (_, cut) = truncate_local_history([ai_other, tool_other])
    assert len(str(cut.content)) < len(content), "非载荷类工具应该被截"


def test_t24_short_tool_results_are_never_touched():
    """L5/T24：短结果**一个字节都不动** —— 别把正常结果也砍了。

    真机实测（`scripts/probe_tool_result_lengths.py`）：工具结果 p50 ≈ 42 字、
    max ≈ 84 字 —— 绝大多数结果本来就短，截断必须是「例外」而不是「默认」。
    """
    from minitap.mobile_use.agents.executor.prompt import truncate_tool_result

    short = "Tap on element with coordinates (1091, 1068) was successful."
    assert truncate_tool_result(short) == short

    # 走整条组装路径也一样
    msgs = build_executor_messages(
        ctx=_ctx(),
        goal="g",
        screen_elements=_elements(),
        local_messages=[ToolMessage(content=short, tool_call_id="c1")],
        device_state_lines=[],
    )
    sent = next(m for m in msgs if isinstance(m, ToolMessage))
    assert sent.content == short


# ---------------------------------------------------------------------------
# T25 —— L3：失败时走**规则换招**（不调 LLM），规则用完退回 LLM
# ---------------------------------------------------------------------------


def _cortex_state(*, report, thoughts, subgoal_id: str = "sg1"):
    """一个「有汇报、有当前子目标」的 state（给 Cortex 节点用）。

    ``report`` 不再进 state —— Cortex 在入口用 ``build_executor_report`` 构建。
    所以这里把 ``executor_done`` 置 True（「有一份没被消费的汇报」），调用方
    再把 ``build_executor_report`` 打桩成返回 ``report``。
    """
    from minitap.mobile_use.agents.planner.types import Subgoal, SubgoalStatus

    subgoal = Subgoal(
        id=subgoal_id,
        description="Enter the WLAN page",
        status=SubgoalStatus.PENDING,
    )
    return State(
        **{
            **_initial_state(),
            "subgoal_plan": [subgoal],
            "executor_done": True,
            "agents_thoughts": list(thoughts),
        }
    )


class _FakeCortexLLM:
    """只记录「被调了几次」的假 LLM。"""

    def __init__(self, calls: list, output):
        self.calls = calls
        self.output = output

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self.output


@pytest.mark.asyncio
async def test_t25_failed_subgoal_uses_the_rule_retry_without_calling_the_llm(monkeypatch):
    """L3/T25：子目标第一次失败 → **规则换招**，一次 LLM 都不调。

    省下的就是方案 L3 说的那次固定调用（3~5 秒 + 1,000+ token）。
    """
    import minitap.mobile_use.agents.cortex.cortex as cortex_mod
    from minitap.mobile_use.agents.cortex.types import CortexOutput

    calls: list = []
    monkeypatch.setattr(
        cortex_mod,
        "get_llm",
        lambda **kwargs: _FakeCortexLLM(
            calls,
            CortexOutput(
                next_subgoal="should never be used",
                completion_reason="should never be used",
            ),
        ),
    )

    report = ExecutorReport(
        outcome="failed",
        summary="子目标未完成，见失败清单（1 条失败事实）。",
        actions_taken=["tap({'target': {'index': 3}})"],
    )
    monkeypatch.setattr(cortex_mod, "build_executor_report", lambda *_: report)
    state = _cortex_state(report=report, thoughts=[])
    update = await cortex_mod.CortexNode(_ctx())(state)

    assert calls == [], "规则换招路径不该调 LLM"
    goal = update.get("executor_goal")
    assert goal, "规则换招也要下发目标"
    # 目标还是**同一个**子目标（换的是路径，不是目标）
    assert "Enter the WLAN page" in goal
    # 带上了「换一条路径」的指令和上次失败的事实
    assert "[retry]" in goal
    assert "already tried" in goal
    # 局部历史照旧清空（换招 = 换目标陈述）
    assert "executor_steps" in update


@pytest.mark.asyncio
async def test_t25b_rule_budget_exhausted_falls_back_to_the_llm(monkeypatch):
    """L3/T25b：规则换招用完 → **退回 LLM**（不许变成「永远不问 LLM」）。

    规则换招的质量不如模型，所以它有次数上限；用完之后必须由模型来决定
    「换个思路」还是「放弃」。
    """
    import minitap.mobile_use.agents.cortex.cortex as cortex_mod
    from minitap.mobile_use.agents.cortex.types import CortexOutput

    calls: list = []
    monkeypatch.setattr(
        cortex_mod,
        "get_llm",
        lambda **kwargs: _FakeCortexLLM(
            calls,
            CortexOutput(
                next_subgoal="Try the network details page instead",
                completion_reason="the first attempt read an empty name field",
            ),
        ),
    )

    report = ExecutorReport(
        outcome="failed",
        summary="子目标未完成，见失败清单（1 条失败事实）。",
    )
    # 这个子目标已经用过一次规则换招 ⇒ 额度用完了，退回 LLM
    thoughts = [
        "[cortex] Issued goal for subgoal sg1: Enter the WLAN page",
        "[cortex] Issued goal for subgoal sg1: Enter the WLAN page "
        "(rule-based retry (previous attempt failed))",
    ]
    monkeypatch.setattr(cortex_mod, "build_executor_report", lambda *_: report)
    state = _cortex_state(report=report, thoughts=thoughts)
    update = await cortex_mod.CortexNode(_ctx())(state)

    assert len(calls) == 1, "规则额度用完后必须退回 LLM"
    assert update.get("executor_goal") == "Try the network details page instead"


def test_t25c_rule_retry_budget_is_bounded_and_counted_from_the_thoughts():
    """L3/T25c：规则的次数上限是**硬**的，而且只数「规则自己用掉的那几次」。

    数 thoughts 而不是另找计数器，是因为 ``agents_thoughts`` 是 append-only 的，
    不用往 State 里加字段（State 是别人在改的地盘）。
    """
    from minitap.mobile_use.agents.cortex.cortex import (
        RULE_RETRY_BUDGET,
        count_goal_issues,
        count_rule_retries,
        should_use_rule_retry,
    )

    assert RULE_RETRY_BUDGET >= 1, "规则换招至少要给一次机会，否则 L3 白做"

    report = ExecutorReport(outcome="failed", summary="没做成")

    # 刚下发、第一次尝试就失败 ⇒ 规则还有额度
    assert should_use_rule_retry(
        report=report,
        subgoal_id="sg1",
        thoughts=["[cortex] Issued goal for subgoal sg1: Enter the WLAN page"],
    )
    # 规则换招已经用过一次 ⇒ 额度用完，退回 LLM
    assert not should_use_rule_retry(
        report=report,
        subgoal_id="sg1",
        thoughts=[
            "[cortex] Issued goal for subgoal sg1: Enter the WLAN page",
            "[cortex] Issued goal for subgoal sg1: Enter the WLAN page "
            "(rule-based retry (previous attempt failed))",
        ],
    )
    # **LLM 自己发的换招不算规则额度** —— 否则额度会被模型的下发次数吃掉，
    # 规则换招就永远轮不上
    assert (
        count_rule_retries(
            [
                "[cortex] Issued goal for subgoal sg1: Enter the WLAN page",
                "[cortex] Issued goal for subgoal sg1: Enter the WLAN page "
                "(the model chose a different route)",
            ],
            "sg1",
        )
        == 0
    )
    # **完成**了也要问 LLM（判证据需要智能）—— 规则只管失败路径
    assert not should_use_rule_retry(
        report=ExecutorReport(outcome="completed", summary="做成了"),
        subgoal_id="sg1",
        thoughts=[],
    )
    # 别的子目标的下发记录不能算数
    assert (
        count_goal_issues(
            ["[cortex] Issued goal for subgoal other: x"],
            "sg1",
        )
        == 0
    )


# ---------------------------------------------------------------------------
# T31 —— §11.6「replan 触发计数」：验证看门狗还活着（R16）
# ---------------------------------------------------------------------------
#
# 这个埋点只有一个用途：R16 要求「计划级失败不能没人管」，新拓扑里的处理链是
# ``executor → cortex（构建汇报并判断）→ plan_manager（看门狗）→ replan → planner``。
# **每一次真正走到 replan，就证明这条链是通的**；反过来「跑了很久却一次
# replan 都没有」有两种可能（一次没失败过 / 看门狗断了），光看别的指标分不开。


class _FakePlannerLLM:
    """只记录调用次数的假 planner LLM。"""

    def __init__(self, calls: list):
        self.calls = calls

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        from minitap.mobile_use.agents.planner.types import PlannerOutput, PlannerSubgoalOutput

        return PlannerOutput(subgoals=[PlannerSubgoalOutput(description="Enter the WLAN page")])


@pytest.mark.asyncio
async def test_t31_replan_is_counted_when_a_subgoal_failed(monkeypatch):
    """T31：计划里有子目标被标记失败 → 记一次 replan。"""
    import minitap.mobile_use.agents.planner.planner as planner_mod
    from minitap.mobile_use.agents.planner.types import Subgoal, SubgoalStatus

    events: list[dict] = []
    monkeypatch.setattr(
        planner_mod.telemetry,
        "capture_replan",
        lambda **kwargs: events.append(kwargs),
    )
    monkeypatch.setattr(
        planner_mod,
        "get_current_foreground_package_async",
        lambda ctx: _async_none(),
    )
    calls: list = []
    monkeypatch.setattr(planner_mod, "get_llm", lambda **kwargs: _FakePlannerLLM(calls))

    state = State(
        **{
            **_initial_state(),
            "subgoal_plan": [
                Subgoal(
                    id="sg1",
                    description="Open Settings",
                    status=SubgoalStatus.SUCCESS,
                ),
                Subgoal(
                    id="sg2",
                    description="Enter the WLAN page",
                    status=SubgoalStatus.FAILURE,
                ),
            ],
            "plan_version": 2,
        }
    )

    await planner_mod.PlannerNode(_ctx())(state)

    assert len(events) == 1, "计划级失败必须记一次 replan（否则看门狗断了没人知道）"
    event = events[0]
    assert event["task_id"] == "test"
    assert event["plan_version"] == 2
    assert event["failed_subgoals"] == 1
    assert event["completed_subgoals"] == 1
    assert event["total_subgoals"] == 2
    # 事件里**不能**有子目标原文（那是用户数据）
    assert "WLAN" not in str(event)


@pytest.mark.asyncio
async def test_t31b_no_replan_event_when_the_plan_is_healthy(monkeypatch):
    """T31b：计划里没有失败子目标 → **不记** replan。

    反向验证：不然这个计数就恒为正，看不出「看门狗有没有真的被触发过」。
    """
    import minitap.mobile_use.agents.planner.planner as planner_mod
    from minitap.mobile_use.agents.planner.types import Subgoal, SubgoalStatus

    events: list[dict] = []
    monkeypatch.setattr(
        planner_mod.telemetry,
        "capture_replan",
        lambda **kwargs: events.append(kwargs),
    )
    monkeypatch.setattr(
        planner_mod,
        "get_current_foreground_package_async",
        lambda ctx: _async_none(),
    )
    calls: list = []
    monkeypatch.setattr(planner_mod, "get_llm", lambda **kwargs: _FakePlannerLLM(calls))

    state = State(
        **{
            **_initial_state(),
            "subgoal_plan": [
                Subgoal(
                    id="sg1",
                    description="Open Settings",
                    status=SubgoalStatus.PENDING,
                ),
            ],
        }
    )

    await planner_mod.PlannerNode(_ctx())(state)

    assert events == [], "健康计划不该记 replan"


# ---------------------------------------------------------------------------
# T32 —— `retry_count` 的口径：它必须是**真实的重发次数**，不是恒为 0
# ---------------------------------------------------------------------------
#
# 真机踩出来的坑（2026-10-21 第六轮）：3 个子目标实际重发了 5 次，
# 而 `retry_count` 全程报 0 —— 因为旧实现拿「失败事实条数」近似，
# 而重发的三条路径里有两条（规则换招、撞步数上限）根本不产生失败事实。
#
# 修法：口径改成「这个子目标被下发过几次 − 1」。
# 「下发」这件事 Cortex 一定写在 `agents_thoughts` 里，绕不开。


def test_t32_retry_count_is_the_real_reissue_count():
    """T32：重发次数 = 下发次数 − 1。

    旧口径（失败事实条数）在**最常见的重发路径**上恒为 0：
    规则换招和撞步数上限都不产生失败事实。
    """
    from minitap.mobile_use.agents.reporter.reporter import _retry_count
    from minitap.mobile_use.agents.planner.types import Subgoal, SubgoalStatus

    def _state(thoughts: list[str]):
        return State(
            **{
                **_initial_state(),
                "subgoal_plan": [
                    Subgoal(
                        id="sg1",
                        description="Open Settings",
                        status=SubgoalStatus.PENDING,
                    )
                ],
                "agents_thoughts": list(thoughts),
                # 关键：**没有任何失败事实**（规则换招 / 撞上限都不产生）
                "executor_failures": [],
            }
        )

    def issue(subgoal_id: str, note: str = "") -> str:
        return f"[cortex] Issued goal for subgoal {subgoal_id}: do it {note}".rstrip()

    # 第一次尝试：下发过 1 次 ⇒ 重发 0 次
    assert _retry_count(_state([issue("sg1")])) == 0

    # 被重发过 1 次（第二次尝试）：下发过 2 次
    assert _retry_count(_state([issue("sg1"), issue("sg1", "again")])) == 1

    # 被重发过 2 次
    assert _retry_count(_state([issue("sg1"), issue("sg1", "again"), issue("sg1", "third")])) == 2

    # **规则换招**的重发也要算进去（它不产生失败事实）
    assert (
        _retry_count(
            _state([issue("sg1"), issue("sg1", "(rule-based retry (previous attempt failed))")])
        )
        == 1
    )

    # 别的子目标的下发记录不能算到本子目标头上
    assert _retry_count(_state([issue("sg1"), issue("other")])) == 0


def test_t32b_issue_log_counts_only_its_own_subgoal():
    """T32b：计数原语按子目标隔离，且对空 ID 返回 0（不能瞎数）。"""
    from minitap.mobile_use.agents.cortex.issue_log import count_goal_issues, count_rule_retries

    thoughts = [
        "[cortex] Issued goal for subgoal sg1: a",
        "[cortex] Issued goal for subgoal sg1: b (rule-based retry)",
        "[cortex] Issued goal for subgoal sg2: c",
        "[executor] Issued goal for subgoal sg1: d",  # 前缀不对，不算
    ]
    assert count_goal_issues(thoughts, "sg1") == 2
    assert count_goal_issues(thoughts, "sg2") == 1
    assert count_goal_issues(thoughts, "sg3") == 0
    assert count_rule_retries(thoughts, "sg1") == 1
    assert count_rule_retries(thoughts, "sg2") == 0
    # 空 ID ⇒ 0（不能把「[cortex] Issued goal for subgoal :」这种脏数据算进去）
    assert count_goal_issues(thoughts, "") == 0


# ---------------------------------------------------------------------------
# T33 —— 证据的**来源**：汇报必须带「屏幕上看到了什么」
# ---------------------------------------------------------------------------
#
# 🔴 这是「为什么会重发」的**根因**（真机 2026-10-21，第七轮/第六轮）。
#
# 判「这个子目标真的完成了吗」需要的是**屏幕状态**；而执行日志里只有
# **动作结果**（「已成功点击 WLAN」= 我点了，不是 我看到了）。
#
# 真机日志里 Cortex 的原话：
#     「报告的证据只有『滑动成功』这类操作结果，summary 也是代码生成的，
#       没有任何 WLAN 页面标题、开关状态或网络名称的原文，因此无法确认
#       已停留在 WLAN 页面。」
# ⇒ Cortex 判不了 ⇒ 重发 ⇒ 清空局部历史 ⇒ 执行器瞎转 ⇒ 撞步数上限 ⇒ 再重发。
#
# 而「屏幕上看到了什么」**早就在 state 里**（`latest_screen_elements`），
# 且它就是**最后一个动作执行之后**那一屏（每执行一个动作都会回 contextor
# 重新抓屏）。所以这是**零成本、零 LLM、零额外抓屏**的修复。


def test_t33_evidence_contains_the_visible_screen_text():
    """T33：证据 = 动作结果 + **屏幕上的可见文字**。

    没有屏幕文字，Cortex 就判不了「做成了没」→ 重发 → 更多调用 + 更多步数。
    """
    from minitap.mobile_use.agents.reporter.reporter import (
        compose_evidence,
        visible_screen_text,
    )

    state = State(
        **{
            **_initial_state(),
            "latest_screen_elements": [
                ScreenElement(
                    index=1,
                    kind="ui",
                    label="WLAN",
                    bounds=ElementBounds(x=0, y=0, width=100, height=50),
                    flags="C",
                ),
                ScreenElement(
                    index=2,
                    kind="ui",
                    label="CMCC-evGt",
                    bounds=ElementBounds(x=0, y=60, width=100, height=50),
                    flags="",
                ),
            ],
        }
    )

    screen = visible_screen_text(state)
    assert screen is not None
    assert "WLAN" in screen
    assert "CMCC-evGt" in screen

    evidence = compose_evidence("已成功点击元素 [1] WLAN", screen)
    assert evidence is not None
    # 两段都要：动作结果说「我做了什么」，屏幕文字说「因此看到了什么」
    assert "已成功点击元素" in evidence
    assert "WLAN" in evidence
    assert "CMCC-evGt" in evidence


def test_t33b_evidence_falls_back_to_ocr_when_the_tree_has_no_text():
    """T33b：元素表没有文字时，用 OCR 的文字（另一条感知通道）。"""
    from minitap.mobile_use.agents.reporter.reporter import visible_screen_text
    from minitap.mobile_use.services.ocr.types import OcrText

    state = State(
        **{
            **_initial_state(),
            "latest_screen_elements": [
                ScreenElement(
                    index=1,
                    kind="ui",
                    label="",
                    bounds=ElementBounds(x=0, y=0, width=10, height=10),
                    flags="",
                )
            ],
            "latest_ocr_results": [
                OcrText(
                    text="蓝牙",
                    score=0.9,
                    bounds=ElementBounds(x=0, y=0, width=50, height=20),
                ),
                OcrText(
                    text="已配对设备",
                    score=0.8,
                    bounds=ElementBounds(x=0, y=30, width=80, height=20),
                ),
            ],
        }
    )

    screen = visible_screen_text(state)
    assert screen is not None
    assert "蓝牙" in screen


def test_t33c_no_visible_text_means_no_evidence_fabrication():
    """T33c：这一屏确实没文字 ⇒ 不编，返回 None。"""
    from minitap.mobile_use.agents.reporter.reporter import visible_screen_text

    state = State(**{**_initial_state(), "latest_screen_elements": []})
    assert visible_screen_text(state) is None


# ---------------------------------------------------------------------------
# T37 —— 证据**不许按「前 N 行」截**（2026-10-24 真机 execution 853）
# ---------------------------------------------------------------------------
#
# 小米应用商城「番茄畅听」详情页：元素表 **152 行**，主按钮「打开」在**第 141 行**。
# 旧实现只取前 25 行 ⇒ 证据里没有「打开」⇒ Cortex 读不到「按钮显示打开」
# ⇒ **正确地**拒绝汇报 ⇒ 改下一个难十倍的目标 ⇒ 执行器掉进 MIUI 看点浮层陷阱
# ⇒ 原地打转 16 分钟、67.9 万 token。
#
# 实测那一屏全量只有 827 字符 ⇒ **字符预算本来够，砍掉的是行数**。


def _long_detail_page(middle_rows: int = 140) -> list:
    """复刻那个详情页的形状：顶部标题 + 一堆推荐位 + **屏底的主按钮**。"""
    elements = [
        ScreenElement(
            index=1,
            kind="ui",
            label="北京臻鼎科技有限公司",
            bounds=ElementBounds(x=339, y=115, width=465, height=59),
            flags="CF",
        ),
        ScreenElement(
            index=2,
            kind="ui",
            label="已通过安全检测",
            bounds=ElementBounds(x=339, y=186, width=282, height=58),
            flags="CF",
        ),
    ]
    for i in range(middle_rows):
        elements.append(
            ScreenElement(
                index=3 + i,
                kind="ui",
                label=f"推荐应用{i}",
                bounds=ElementBounds(x=280, y=500 + i * 10, width=500, height=60),
                flags="CF",
            )
        )
    # 屏底：主操作按钮 —— 旧实现在这里被砍掉了
    elements.append(
        ScreenElement(
            index=3 + middle_rows,
            kind="ui",
            label="打开",
            bounds=ElementBounds(x=141, y=2501, width=938, height=124),
            flags="",
        )
    )
    return elements


def test_t37_evidence_keeps_the_bottom_action_button():
    """🔴 **核心断言**：屏底的主按钮必须进证据。

    少了它 ⇒ supervisor 核实不了 ⇒ 拒绝 ⇒ 换成更难的目标 ⇒ 死循环（853）。
    """
    from minitap.mobile_use.agents.reporter.reporter import (
        compose_evidence,
        visible_screen_text,
    )

    state = State(
        **{
            **_initial_state(),
            "latest_screen_elements": _long_detail_page(),
        }
    )

    screen = visible_screen_text(state)
    assert screen is not None
    assert "打开" in screen, f"屏底的主按钮被砍掉了：{screen[-200:]}"
    assert "北京臻鼎科技有限公司" in screen, "屏顶也不该丢"

    evidence = compose_evidence("已成功点击元素 [1]", screen)
    assert evidence is not None
    assert "打开" in evidence


def test_t37b_short_screens_are_not_truncated_at_all():
    """短屏（绝大多数）⇒ **一字不落地**给出去。"""
    from minitap.mobile_use.agents.reporter.reporter import visible_screen_text

    state = State(
        **{
            **_initial_state(),
            "latest_screen_elements": [
                ScreenElement(
                    index=i + 1,
                    kind="ui",
                    label=f"第{i}行",
                    bounds=ElementBounds(x=0, y=i * 10, width=100, height=10),
                    flags="",
                )
                for i in range(20)
            ],
        }
    )

    screen = visible_screen_text(state)
    assert screen is not None
    for i in range(20):
        assert f"第{i}行" in screen, f"第{i}行不见了"


def test_t37c_over_budget_keeps_both_ends_and_says_how_many_were_dropped():
    """超预算 ⇒ **保首尾** + **明说**省略了多少条（不装成「全都看到了」）。"""
    from minitap.mobile_use.agents.reporter.reporter import (
        MAX_SCREEN_EVIDENCE_CHARS,
        visible_screen_text,
    )

    state = State(
        **{
            **_initial_state(),
            "latest_screen_elements": [
                ScreenElement(
                    index=i + 1,
                    kind="ui",
                    label=f"很长的一行文字{i}" + "x" * 60,
                    bounds=ElementBounds(x=0, y=i * 10, width=100, height=10),
                    flags="",
                )
                for i in range(80)
            ],
        }
    )

    screen = visible_screen_text(state)
    assert screen is not None
    assert len(screen) <= MAX_SCREEN_EVIDENCE_CHARS + 60, "超预算了没裁"
    assert "省略" in screen, "省略了多少条要说清楚"
    assert "很长的一行文字0" in screen, "开头要保住"
    assert "很长的一行文字79" in screen, "**结尾要保住** —— 主按钮在屏底"


# ---------------------------------------------------------------------------
# T34 / T35 —— cortex 的韧性 + 「一轮一生命周期」（每轮清空 + 上一轮摘要）
# ---------------------------------------------------------------------------
#
# T34 的来源：manager 链路上真机跑出来的（2026-10-22）。MiniMax 偶发返回
# 「finish_reason='tool_calls' 但内容是空的」⇒ `_parse` 抛 ValueError ⇒
# **整个任务当场 FAILED**（execution 797）。executor 有三层韧性扛得住，
# cortex 没有。
#
# T35 的来源：真机第六轮 x39h9m 的尝试序列
# `[(2, completed), (13, blocked), (1, completed)]` —— 第一次 2 步就成了，
# 重发后那次却瞎转 13 步。根因之一是**重发时局部历史被清空了**：
# 执行器忘了自己试过什么。


class _AlwaysFailingLLM:
    """每次调用都抛（模拟 provider 返回空 tool_calls）。"""

    def __init__(self, calls: list):
        self.calls = calls

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        raise ValueError("MiniMax response had no tool_calls and no content.")


@pytest.mark.asyncio
async def test_t34_cortex_parse_failure_does_not_kill_the_task(monkeypatch):
    """T34：Cortex 解析失败 → 重试 → 降级成**重发当前子目标**，不抛异常。

    降级**不引入新的事实断言**：不判完成（那是误报成功）、不放弃
    （一次 provider 抖动就否掉整个子目标太狠）。
    """
    import minitap.mobile_use.agents.cortex.cortex as cortex_mod
    from minitap.mobile_use.agents.reporter.reporter import ExecutorReport

    calls: list = []
    monkeypatch.setattr(cortex_mod, "get_llm", lambda **kwargs: _AlwaysFailingLLM(calls))

    report = ExecutorReport(outcome="failed", summary="没做成")
    state = _cortex_state(report=report, thoughts=[])

    # ⚠️ 关键：**不抛**。抛了就是「一次 provider 抖动杀掉整个任务」。
    update = await cortex_mod.CortexNode(_ctx())(state)

    assert update.get("executor_goal"), "降级后必须重发当前子目标"
    assert "Enter the WLAN page" in update["executor_goal"]
    # 步数归零 = 给这次重发一份新的预算
    assert update.get("executor_steps") == 0


@pytest.mark.asyncio
async def test_t35_every_issue_starts_a_fresh_round(monkeypatch):
    """T35：每一轮（一次执行尝试）都清空局部历史，不带上一轮的任何内容。

    清空的理由：局部历史每步约 +3 条消息 / +490 字符，不清就按步数线性增长
    （真机归档 880：末轮 461 条消息 / 6 万字符，占 executor prompt 的 54%）。
    """
    import minitap.mobile_use.agents.cortex.cortex as cortex_mod
    from minitap.mobile_use.agents.cortex.types import CortexOutput
    from minitap.mobile_use.agents.reporter.reporter import ExecutorReport
    from minitap.mobile_use.constants import EXECUTOR_MESSAGES_KEY
    from minitap.mobile_use.utils.failed_targets import FailedTarget

    calls: list = []

    def fake_llm(**kwargs):
        class _L:
            def with_structured_output(self, schema):
                return self

            async def ainvoke(self, messages):
                calls.append(messages)
                # 说「还没成，换个思路再来」⇒ 走 issue 分支
                return CortexOutput(
                    next_subgoal="Try the search box instead",
                    completion_reason=None,
                    subgoal_completed=False,
                )

        return _L()

    monkeypatch.setattr(cortex_mod, "get_llm", fake_llm)

    # --- 情形 1：同目标重发（**有**汇报）⇒ 清空 + 上一轮摘要 ---
    #
    # report.outcome="failed" 且规则额度没用过 ⇒ 走 **L3 规则换招**（不调 LLM）。
    # 规则换招和 LLM 换招都是「新的一轮」，同样要清空、同样要带摘要。
    report = ExecutorReport(
        outcome="failed",
        summary="点了搜索框，没进到结果页",
        failure_reason="目标不在当前屏幕上",
        failed_targets=[
            FailedTarget(
                key="k",
                index=7,
                resource_id="com.x:id/search_box",
                label="搜索",
            )
        ],
    )
    state = _cortex_state(report=report, thoughts=[])
    update = await cortex_mod.CortexNode(_ctx())(state)

    assert update.get("executor_goal"), "同目标重发必须有一个目标陈述"
    messages = update[EXECUTOR_MESSAGES_KEY]
    assert isinstance(messages[0], RemoveMessage)
    assert messages[0].id == REMOVE_ALL_MESSAGES, "新一轮必须先清空上一轮的消息"
    assert len(messages) == 1, "清空后不带上一轮的任何内容"

    # --- 情形 2：首次下发（**没**汇报）⇒ 同样只清空 ---
    state = _cortex_state(report=None, thoughts=[])
    update = await cortex_mod.CortexNode(_ctx())(state)

    assert update.get("executor_goal") == "Try the search box instead"
    messages = update[EXECUTOR_MESSAGES_KEY]
    assert len(messages) == 1 and isinstance(messages[0], RemoveMessage), "首次下发也要清空"


# ---------------------------------------------------------------------------
# T36 —— planner 的韧性：解析失败不许杀死整个任务
# ---------------------------------------------------------------------------
#
# 和 cortex 同一个病（N33）：MiniMax 偶发返回空响应 ⇒ `_parse` 抛 ⇒ 任务当场死。
# planner 更靠前：**计划挂了整个任务连第一步都走不了**。
#
# 降级成什么（不引入任何新的事实断言）：
# * replan 场景（有被标失败的子目标）⇒ **把那个失败的子目标重新下发一次**
#   （它「还没做完」，所以重新下发是事实，不是编造）。
# * 首次规划（没有旧计划）⇒ **没有可退守的东西**，只能抛 —— 这条真的无解。


class _AlwaysFailingPlannerLLM:
    def __init__(self, calls: list):
        self.calls = calls

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        raise ValueError("MiniMax response had no tool_calls and no content.")


@pytest.mark.asyncio
async def test_t36_planner_parse_failure_degrades_to_reissuing_the_failed_subgoal(monkeypatch):
    """T36：planner 解析失败 → 重试 → 降级成重新下发失败的子目标（不抛）。"""
    import minitap.mobile_use.agents.planner.planner as planner_mod
    from minitap.mobile_use.agents.planner.types import Subgoal, SubgoalStatus

    calls: list = []
    monkeypatch.setattr(
        planner_mod,
        "get_llm",
        lambda **kwargs: _AlwaysFailingPlannerLLM(calls),
    )
    monkeypatch.setattr(
        planner_mod,
        "get_current_foreground_package_async",
        lambda ctx: _async_none(),
    )

    state = State(
        **{
            **_initial_state(),
            "subgoal_plan": [
                Subgoal(id="sg1", description="Open Settings", status=SubgoalStatus.SUCCESS),
                Subgoal(
                    id="sg2",
                    description="Enter the WLAN page",
                    status=SubgoalStatus.FAILURE,
                ),
            ],
        }
    )

    # ⚠️ 关键：**不抛**（抛了 = 一次 provider 抖动让整个任务当场死）
    update = await planner_mod.PlannerNode(_ctx())(state)

    plan = update.get("subgoal_plan") or []
    assert plan, "降级后必须还有计划"
    descriptions = [s.description for s in plan]
    assert "Enter the WLAN page" in descriptions, "失败的子目标应该被重新下发"
    # 已完成的子目标照旧保留（不能因为重规划就重做一遍）
    assert "Open Settings" in descriptions


async def _async_none():
    return None
