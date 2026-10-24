"""「上一步的结果」写进执行器局部历史（2026-10-24，真机 execution 853）。

🔴 **为什么**：执行器的局部历史原来只有「调用 + 工具回执」
（「Key Home pressed successfully.」），**没有「结果」** ——
屏幕长什么样、前台 app 是谁，一概没记。执行器于是每轮只能拿当屏去和
脑子里的几十条记忆比，靠回忆判断自己有没有在原地打转。
实测：它把 Home 按了 **13 次**，每一次都以为 Home 会回到桌面。

⚠️ **只陈述事实，不下结论**：「这个动作没用」是**相对于目标**的判断，
框架做不了（「Home 把我送到看点流」和「滑动让我看到下一个视频」在框架
眼里一模一样）。⇒ 框架只搬客观事实，判定留给模型。
"""

from __future__ import annotations

from minitap.mobile_use.agents.contextor.contextor import render_action_result
from minitap.mobile_use.utils.screen_elements import (
    ElementBounds,
    ScreenElement,
    is_noise_text,
    screen_fingerprint,
    visible_texts,
)


def _el(index: int, label: str) -> ScreenElement:
    return ScreenElement(
        index=index,
        kind="ui",
        label=label,
        bounds=ElementBounds(x=0, y=index * 10, width=100, height=10),
        flags="",
    )


class TestVisibleTexts:
    def test_takes_label_first_and_dedupes_by_original_text(self):
        elements = [
            _el(1, "打开"),
            _el(2, "打开"),
            _el(3, "  打开  "),
            _el(4, "北京臻鼎科技有限公司"),
        ]
        # 去重按**原文**（strip 之后）：「打开」和「打开 」是同一条事实
        assert visible_texts(elements) == ["打开", "北京臻鼎科技有限公司"]

    def test_empty_elements_gives_empty_list(self):
        assert visible_texts([]) == []
        assert visible_texts(None) == []


class TestScreenFingerprint:
    def test_digits_are_normalized_so_the_clock_is_not_a_change(self):
        """状态栏时钟自己会变 ⇒ 指纹必须不认它。"""
        before = [_el(1, "2:14"), _el(2, "打开")]
        after = [_el(1, "3:47"), _el(2, "打开")]
        assert screen_fingerprint(before) == screen_fingerprint(after)

    def test_real_content_change_is_a_change(self):
        before = [_el(1, "安装"), _el(2, "番茄畅听")]
        after = [_el(1, "打开"), _el(2, "番茄畅听")]
        assert screen_fingerprint(before) != screen_fingerprint(after)

    def test_numbered_items_are_different_content(self):
        """🔴 **带文字的编号不许归一**。

        踩过的坑：一开始把**所有**数字都归成 ``#``，结果「视频标题1 → 视频标题2」
        也被归成同一个 ⇒ **刷短视频（正常浏览）被误判成「屏幕内容没变」**。
        ⇒ 归一只覆盖「纯数字/标点的短串」（时钟、电量）。
        """
        before = [_el(1, "视频标题1"), _el(2, "2:14")]
        after = [_el(1, "视频标题2"), _el(2, "2:31")]
        assert screen_fingerprint(before) != screen_fingerprint(after)

    def test_noise_detection_is_narrow(self):
        assert is_noise_text("2:24")
        assert is_noise_text("96 4")
        assert is_noise_text("29%4")
        assert not is_noise_text("视频标题2")
        assert not is_noise_text("471MB")
        assert not is_noise_text("北京臻鼎科技有限公司")


class TestRenderActionResult:
    def test_first_screen_has_no_previous_step(self):
        assert (
            render_action_result(
                previous_elements=None,
                previous_app=None,
                current_elements=[_el(1, "打开")],
                current_app="com.x",
            )
            is None
        )

    def test_same_screen_same_app(self):
        """什么都没变 —— 这是 853 按 Home 之后的真实情况。"""
        line = render_action_result(
            previous_elements=[_el(1, "看点视频"), _el(2, "2:14")],
            previous_app="com.miui.newhome",
            current_elements=[_el(1, "看点视频"), _el(2, "2:31")],
            current_app="com.miui.newhome",
        )
        assert line is not None
        assert "前台应用没变" in line
        assert "com.miui.newhome" in line
        assert "屏幕内容没变" in line

    def test_foreground_moved(self):
        line = render_action_result(
            previous_elements=[_el(1, "番茄畅听")],
            previous_app="com.miui.newhome",
            current_elements=[_el(1, "番茄畅听")],
            current_app="com.miui.home",
        )
        assert line is not None
        assert "前台应用变成了 com.miui.home" in line

    def test_new_text_is_reported(self):
        """刷短视频：每次滑一下都出现新标题 ⇒ 「新出现了：…」⇒ 不是「没进展」。"""
        line = render_action_result(
            previous_elements=[_el(1, "视频标题1"), _el(2, "2:14")],
            previous_app="com.video",
            current_elements=[_el(1, "视频标题2"), _el(2, "2:31")],
            current_app="com.video",
        )
        assert line is not None
        assert "前台应用没变" in line
        assert "屏幕上新出现了：视频标题2" in line
        # 时钟不算「新内容」（它是噪音）
        assert "2:31" not in line

    def test_foreground_read_failure_is_not_said_to_be_gone(self):
        """读不到 ≠ 没了 —— 措辞不能误导。"""
        line = render_action_result(
            previous_elements=[_el(1, "某内容")],
            previous_app="com.x",
            current_elements=[_el(1, "某内容")],
            current_app=None,
        )
        assert line is not None
        assert "前台应用读不到了" in line
        assert "前台应用没了" not in line

    def test_does_not_say_invalid_or_useless(self):
        """🔴 **不许下结论**：「重复」「无效」「没用」这类词一律不许出现。

        刷短视频时连滑 10 次是在正常浏览 —— 框架要是说「这个动作没用」，
        就是把正常浏览误伤掉。「有没有用」是相对目标的判断，只有模型能做。
        """
        line = render_action_result(
            previous_elements=[_el(1, "视频标题A")],
            previous_app="com.video",
            current_elements=[_el(1, "视频标题B")],
            current_app="com.video",
        )
        assert line is not None
        for forbidden in ("无效", "没用", "重复", "进展", "invalid", "useless", "repeat"):
            assert forbidden not in line, f"框架不许下这种结论：{line}"

    def test_result_line_goes_into_the_executor_history(self):
        """这行必须进 ``executor_messages`` —— 执行器下一轮才读得到。"""
        from minitap.mobile_use.constants import EXECUTOR_MESSAGES_KEY

        assert EXECUTOR_MESSAGES_KEY == "executor_messages"


class TestFailureFactInResultLine:
    """「上一步失败了，目标是谁」并进 `[上一步的结果]` 那一行（2026-10-24）。

    为什么并进这一行而不是单开一条：失败事实要**记在产生它的那一步旁边**
    （模型读历史时不用跨块对账），而这一行是 Contextor 写的。
    """

    def _line(self, failure: dict[str, str]) -> str:
        line = render_action_result(
            previous_elements=[_el(1, "视频标题A")],
            previous_app="com.video",
            current_elements=[_el(1, "视频标题B")],
            current_app="com.video",
            failure=failure,
        )
        assert line is not None
        return line

    def test_failure_fact_is_merged_into_the_line(self):
        line = self._line({"label": "点赞", "resource_id": "com.app:id/like_button"})
        assert "这一步失败了" in line
        assert '"点赞"' in line
        assert "like_button" in line
        # 和其它事实**同一行**（不是一个独立块）
        assert "前台应用没变" in line
        assert "屏幕上新出现了" in line

    def test_no_bare_index_in_the_line(self):
        """🔴 索引每轮重编号 —— 带上它会让模型去点一个完全无关的元素。"""
        line = self._line({"label": "点赞", "resource_id": "com.app:id/like_button"})
        assert "[48]" not in line
        assert "48" not in line

    def test_falls_back_to_id_tail_when_there_is_no_label(self):
        """纯图标按钮没有文字 ⇒ 退回 resource-id 尾段。"""
        line = self._line({"label": "", "resource_id": "com.app:id/download_btn"})
        assert "download_btn" in line
        assert "这一步失败了" in line

    def test_no_conclusion_wording(self):
        """🔴 **只记录结果，不下结论**。

        「不要再用这个目标」是**相对于目标**的判断 —— 框架不知道目标。
        失败不总是「元素没了」（也可能是「输入框聚焦失败」），所以连
        「已经不在屏幕上了」都不能断言。
        """
        line = self._line({"label": "点赞", "resource_id": "com.app:id/like_button"})
        for forbidden in (
            "不要再用",
            "换个",
            "不在屏幕",
            "无效",
            "没用",
            "重复",
            "进展",
        ):
            assert forbidden not in line, f"框架不许下这种结论：{line}"

    def test_no_failure_means_no_failure_clause(self):
        """动作成功 ⇒ 这一行里不该出现失败。"""
        line = render_action_result(
            previous_elements=[_el(1, "视频标题A")],
            previous_app="com.video",
            current_elements=[_el(1, "视频标题B")],
            current_app="com.video",
            failure=None,
        )
        assert line is not None
        assert "这一步失败" not in line

    def test_empty_identity_does_not_render_a_placeholder(self):
        """🔴 **占位符不许进模型的提示词**。

        模型只给索引时（label 和 resource_id 都是空的）—— 而索引每轮重编号，
        不能写进去。这时只说「失败了」，不拿 ``(unknown target)`` 顶替：
        那个词长得像个元素名，模型会去找一个叫 "unknown target" 的东西。
        """
        line = self._line({"label": "", "resource_id": ""})
        assert "这一步失败了" in line
        assert "(unknown target)" not in line


class TestLastActionFailureFact:
    """工具层递过来的那个瞬时事实：原始数据，不带 index。"""

    def test_carries_label_and_resource_id_but_no_index(self):
        from minitap.mobile_use.tools.types import Target
        from minitap.mobile_use.tools.utils import last_action_failure_fact

        fact = last_action_failure_fact(
            Target(index=48, resource_id="com.app:id/like_button", text="点赞")
        )
        assert fact["label"] == "点赞"
        assert fact["resource_id"] == "com.app:id/like_button"
        assert "48" not in str(fact)

    def test_label_is_stripped_and_may_be_empty(self):
        from minitap.mobile_use.tools.types import Target
        from minitap.mobile_use.tools.utils import last_action_failure_fact

        fact = last_action_failure_fact(Target(index=1, text="  搜索  "))
        assert fact["label"] == "搜索"
        assert fact["resource_id"] == ""

        empty = last_action_failure_fact(Target(index=1))
        assert empty["label"] == ""
        assert empty["resource_id"] == ""

    def test_fact_has_exactly_the_two_identity_fields(self):
        """事实就两个字段：身份。不带 index，也不带结论。

        形状固定 ⇒ 用 TypedDict（``ActionFailureFact``）而不是 ``dict[str, str]``，
        键名拼错时类型检查器会拦（规则：types-typeddict-over-dict-any）。
        """
        from minitap.mobile_use.tools.types import Target
        from minitap.mobile_use.tools.utils import last_action_failure_fact

        fact = last_action_failure_fact(
            Target(index=48, resource_id="com.app:id/like_button", text="点赞")
        )
        assert set(fact) == {"label", "resource_id"}
