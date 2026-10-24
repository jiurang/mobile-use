"""Agent thoughts 压缩的不变量测试。

测的是**性质**，不是实现细节。核心那条是「位置无关」——
它保证已发出的字节永不改变，也就是前缀缓存不因压缩而击穿。
这条性质一旦破了，压缩就从「省 34% token」变成「满价重算涨 4.7 倍」。

对应调研：doc/thoughts-compaction-plan.md
"""

from __future__ import annotations

import pytest

from minitap.mobile_use.utils.thoughts_compaction import (
    PLANNER_AUDIENCE,
    compress_thought,
    is_planner_relevant,
    render_thoughts_history,
    split_role,
    split_sentences,
)

# ---- 真实数据样本（取自真机归档 exec 212） ----

CORTEX_ZH = (
    "[cortex] Decisions reason: 屏幕显示快手极速版已打开，底部「首页」为选中态，"
    "顶部「发现」流被选中（S 标志），说明已处于首页推荐信息流，可完成第一个子目标。"
    "当前正在播放第1个视频：作者「@小福影视」，内容为电视剧《玫瑰之战》剪辑，"
    "与用户画像中「人民的名义电视剧」偏好相符。先单击视频区域唤出进度条，"
    "读取已播放时长。\n\nGoals completion reason: UI 证据显示底部「首页」"
    "CheckedTextView 为 KS 选中态，已确认处于首页推荐信息流。"
)

CORTEX_EN = (
    "[cortex] Decisions reason: The current screen shows Kuaishou, not Settings. "
    "No prior attempts were made. I'll launch the system Settings app via launch_app, "
    "which is an unpredictable action, so it should be the only action this turn. "
    "After observing the new screen, I'll locate and tap the WLAN entry."
)

EXECUTOR_OK = "[executor] Tap on element with coordinates (610, 1258) was successful."
EXECUTOR_FAIL = (
    "[executor] Failed to tap on element. Attempts: resource_id='com.kuaishou.nebula:id/root': "
    "no element with resource_id='com.kuaishou.nebula:id/root' on the current screen."
)
ORCHESTRATOR = "[orchestrator] ==== NEXT SUBGOAL: - [ID:klvi1r]: 处理第2个视频：符合画像则点赞 ===="


class TestSplitRole:
    def test_splits_known_roles(self):
        assert split_role(CORTEX_EN) == ("cortex", CORTEX_EN[len("[cortex] ") :])
        assert split_role(EXECUTOR_OK)[0] == "executor"
        assert split_role(ORCHESTRATOR)[0] == "orchestrator"

    def test_unknown_role_when_no_prefix(self):
        role, body = split_role("no prefix here")
        assert role == "unknown"
        assert body == "no prefix here"

    def test_body_is_stripped(self):
        _, body = split_role("[cortex]   padded body  ")
        assert body == "padded body"


class TestSplitSentences:
    def test_splits_on_chinese_period(self):
        assert len(split_sentences("第一句。第二句。第三句。")) == 3

    def test_splits_on_english_period(self):
        assert len(split_sentences("First one. Second one. Third one.")) == 3

    def test_keeps_the_punctuation(self):
        assert split_sentences("第一句。第二句。")[0] == "第一句。"

    def test_empty_input_gives_empty_list(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []

    def test_single_sentence_without_terminator(self):
        assert split_sentences("没有句号的一句话") == ["没有句号的一句话"]


class TestCompressThought:
    """事实必须逐字保留 —— 这是硬约束，不是偏好。"""

    def test_executor_facts_are_verbatim(self):
        assert compress_thought(EXECUTOR_OK) == EXECUTOR_OK
        assert compress_thought(EXECUTOR_FAIL) == EXECUTOR_FAIL

    def test_orchestrator_milestones_are_verbatim(self):
        assert compress_thought(ORCHESTRATOR) == ORCHESTRATOR

    def test_unknown_role_is_verbatim(self):
        raw = "[contextor] 一些没有角色规则的内容。更多内容。"
        assert compress_thought(raw) == raw

    def test_cortex_prose_is_compressed(self):
        out = compress_thought(CORTEX_ZH)
        assert len(out) < len(CORTEX_ZH)
        assert out.startswith("[cortex] ")
        assert " … " in out

    def test_role_prefix_appears_exactly_once(self):
        """草稿里出现过双 `[cortex]`，这条锁死它。"""
        out = compress_thought(CORTEX_ZH)
        assert out.count("[cortex]") == 1

    def test_single_sentence_is_preserved(self):
        raw = "[cortex] Decisions reason: 只有一句话，没有更多内容。"
        assert compress_thought(raw) == raw

    def test_keeps_first_and_last_sentence(self):
        out = compress_thought(CORTEX_EN)
        assert "The current screen shows Kuaishou" in out
        assert "I'll locate and tap the WLAN entry." in out
        # 中间那句被丢掉
        assert "No prior attempts were made." not in out

    def test_first_equals_last_does_not_duplicate(self):
        raw = "[cortex] Decisions reason: 同一句话。"
        out = compress_thought(raw)
        assert out == raw


class TestRenderThoughtsHistory:
    def test_empty_gives_empty_string(self):
        assert render_thoughts_history([]) == ""

    def test_disabled_is_verbatim(self):
        thoughts = [CORTEX_ZH, EXECUTOR_OK]
        assert render_thoughts_history(thoughts, enabled=False) == "\n".join(thoughts)

    def test_enabled_compresses_only_cortex(self):
        out = render_thoughts_history([CORTEX_ZH, EXECUTOR_OK])
        assert EXECUTOR_OK in out
        assert " … " in out

    def test_line_count_preserved(self):
        """逐条对应：行数不变，归档统计的行首锚定口径才不破。"""
        thoughts = [CORTEX_ZH, EXECUTOR_OK, ORCHESTRATOR, EXECUTOR_FAIL]
        out = render_thoughts_history(thoughts)
        assert len(out.split("\n")) == len(thoughts)

    def test_fact_line_counts_survive(self):
        """归档分析的权威口径：数 `[executor]` 开头行。压完必须逐条一致。"""
        thoughts = [EXECUTOR_OK, EXECUTOR_FAIL, CORTEX_ZH, ORCHESTRATOR, EXECUTOR_OK]
        out = render_thoughts_history(thoughts)
        assert len([ln for ln in out.split("\n") if ln.startswith("[executor]")]) == 3
        assert len([ln for ln in out.split("\n") if ln.startswith("[orchestrator]")]) == 1


class TestPrefixStability:
    """本模块全部收益的根据。

    f(ts[:k]) 必须恒为 f(ts[:k+1]) 的字节前缀 —— 已发出的字节永不改变。
    破了这条，压缩就从「省 34%」变成「满价重算涨 4.7 倍」。
    """

    @pytest.fixture
    def thoughts(self):
        return [
            CORTEX_ZH,
            EXECUTOR_OK,
            CORTEX_EN,
            ORCHESTRATOR,
            EXECUTOR_FAIL,
            "[cortex] Decisions reason: 第三条推理。第二句。",
            "[executor] Successfully waited for 60000 milliseconds.",
        ]

    def test_prefix_property_holds(self, thoughts):
        for k in range(1, len(thoughts)):
            prev = render_thoughts_history(thoughts[:k])
            cur = render_thoughts_history(thoughts[: k + 1])
            assert cur.startswith(prev), f"第 {k + 1} 轮击穿了前缀"

    def test_earlier_entries_never_change(self, thoughts):
        """第 k 条的渲染结果，在之后任何一轮里都不许变。"""
        snapshots = [render_thoughts_history(thoughts[:k]) for k in range(1, len(thoughts) + 1)]
        for later_k in range(1, len(thoughts)):
            for earlier_k in range(1, later_k + 1):
                earlier = snapshots[earlier_k - 1]
                later = snapshots[later_k]
                assert later.startswith(earlier)

    def test_position_independence(self, thoughts):
        """同一条 thought，无论它处在什么位置，渲染结果必须一致。"""
        single = render_thoughts_history([CORTEX_ZH])
        embedded = render_thoughts_history([EXECUTOR_OK, CORTEX_ZH, ORCHESTRATOR])
        assert single in embedded

    def test_disabled_also_prefix_stable(self, thoughts):
        for k in range(1, len(thoughts)):
            prev = render_thoughts_history(thoughts[:k], enabled=False)
            cur = render_thoughts_history(thoughts[: k + 1], enabled=False)
            assert cur.startswith(prev)


class TestPlannerAudience:
    """规划者受众（cortex / orchestrator）：砍掉 executor 的成功流水。

    依据（实测 2026-10-24，exec 861）：executor 的行占 cortex thoughts 的
    **60.9%**，而 `cortex.md` 的「What you are given」里没有 thoughts 这一项
    —— 结果走 Executor report、失败走两个失败块 ⇒ 对规划者是冗余的。

    对应方案：doc/cortex-prompt-cache-optimization-20261024.md
    """

    def test_drops_successful_executor_line(self):
        out = render_thoughts_history([EXECUTOR_OK, CORTEX_ZH], audience=PLANNER_AUDIENCE)
        assert EXECUTOR_OK not in out
        assert "[cortex]" in out

    def test_keeps_failed_executor_line(self):
        """失败行是「这条路走不通」的**唯一具体原因** —— 失败块里只有名字。"""
        out = render_thoughts_history([EXECUTOR_FAIL], audience=PLANNER_AUDIENCE)
        assert EXECUTOR_FAIL in out

    def test_keeps_orchestrator_milestone(self):
        out = render_thoughts_history([ORCHESTRATOR], audience=PLANNER_AUDIENCE)
        assert "[orchestrator]" in out

    def test_keeps_line_without_role_prefix(self):
        """没有角色前缀的行：不知道是什么，别砍。"""
        raw = "some line without a role prefix"
        out = render_thoughts_history([raw], audience=PLANNER_AUDIENCE)
        assert raw in out

    def test_full_audience_unchanged(self):
        """默认 `audience="full"` 一个字都没变 —— 归档口径靠它。"""
        thoughts = [EXECUTOR_OK, EXECUTOR_FAIL, CORTEX_ZH, ORCHESTRATOR, EXECUTOR_OK]
        out = render_thoughts_history(thoughts)
        assert len([ln for ln in out.split("\n") if ln.startswith("[executor]")]) == 3
        assert len([ln for ln in out.split("\n") if ln.startswith("[orchestrator]")]) == 1

    def test_filter_applies_even_when_compression_disabled(self):
        """受众与压缩是**两个维度**：关掉压缩不该把流水放回来。"""
        out = render_thoughts_history(
            [EXECUTOR_OK, CORTEX_ZH], enabled=False, audience=PLANNER_AUDIENCE
        )
        assert EXECUTOR_OK not in out

    def test_prefix_property_holds(self):
        """planner 版同样必须前缀稳定 —— 破了这条，整个优化白做。"""
        thoughts = [
            CORTEX_ZH,
            EXECUTOR_OK,
            CORTEX_EN,
            ORCHESTRATOR,
            EXECUTOR_FAIL,
            "[cortex] Decisions reason: 第三条推理。第二句。",
            "[executor] Successfully waited for 60000 milliseconds.",
        ]
        for k in range(1, len(thoughts)):
            prev = render_thoughts_history(thoughts[:k], audience=PLANNER_AUDIENCE)
            cur = render_thoughts_history(thoughts[: k + 1], audience=PLANNER_AUDIENCE)
            assert cur.startswith(prev), f"第 {k + 1} 轮击穿了前缀"

    def test_earlier_entries_never_change(self):
        """第 k 条的去留，在之后任何一轮里都不许变（逐条独立）。"""
        thoughts = [
            CORTEX_ZH,
            EXECUTOR_OK,
            ORCHESTRATOR,
            EXECUTOR_FAIL,
            "[executor] Successfully waited for 60000 milliseconds.",
        ]
        snapshots = [
            render_thoughts_history(thoughts[:k], audience=PLANNER_AUDIENCE)
            for k in range(1, len(thoughts) + 1)
        ]
        for later_k in range(1, len(thoughts)):
            for earlier_k in range(1, later_k + 1):
                assert snapshots[later_k].startswith(snapshots[earlier_k - 1])

    def test_position_independence(self):
        """同一条 thought，无论处在什么位置，去留必须一致。"""
        assert not is_planner_relevant(EXECUTOR_OK)
        assert is_planner_relevant(EXECUTOR_FAIL)
        assert is_planner_relevant(CORTEX_ZH)
        assert is_planner_relevant(ORCHESTRATOR)
        # 单独渲染 vs 嵌在别的条目中间 —— 结果必须一样
        single = render_thoughts_history([EXECUTOR_FAIL], audience=PLANNER_AUDIENCE)
        embedded = render_thoughts_history(
            [EXECUTOR_OK, EXECUTOR_FAIL, ORCHESTRATOR], audience=PLANNER_AUDIENCE
        )
        assert single in embedded

    def test_planner_output_is_shorter(self):
        """方案的核心收益：同样的输入，规划者看到的少得多。"""
        thoughts = [EXECUTOR_OK] * 10 + [CORTEX_ZH] * 3
        full = render_thoughts_history(thoughts)
        planner = render_thoughts_history(thoughts, audience=PLANNER_AUDIENCE)
        assert "[executor]" not in planner
        assert "[cortex]" in planner
        assert len(planner) < len(full) * 0.6
