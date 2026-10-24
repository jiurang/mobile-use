"""失败目标账本的测试。

这块解决的是真机上最常见的一类失败：**目标已经不在屏幕上了**
（横屏全屏页被视频播完带走，模型思考 17 秒才动手）。

要守住三件事：
1. 同一个目标连着失败要能数出来，并且到阈值后**顶回去**（不再执行）；
2. 换了目标就不算「连续失败」（索引每轮重编号，只看 index 会误伤）；
3. 账本能渲染成模型看得懂的一行。
"""

from __future__ import annotations

import pytest

from minitap.mobile_use.utils.failed_targets import (
    FAILURE_BREAKER_THRESHOLD,
    FailedTarget,
    breaker_reason,
    describe,
    describe_target,
    make_key,
    record_failure,
    render_failures,
)


def test_key_carries_index_and_identity():
    """键里必须有身份：索引每轮重编号，只看 index 会把「换了目标」当成「重复犯错」。"""
    key = make_key(index=48, resource_id="com.app:id/like", label="点赞")
    assert "index=48" in key
    assert "com.app:id/like" in key
    assert "点赞" in key

    # 同一个 index、不同元素 → 不同的键
    other = make_key(index=48, resource_id="com.app:id/danmaku", label="弹幕")
    assert key != other


def test_same_target_twice_counts_up_and_trips_the_breaker():
    streak = record_failure(
        [],
        key="index=48|id=com.app:id/like",
        index=48,
        resource_id="com.app:id/like",
        reason="element is no longer on the screen",
    )
    assert streak[0].count == 1
    # 第一次失败不熔断：模型重试一次是合理的
    assert breaker_reason(streak, "index=48|id=com.app:id/like") is None

    streak = record_failure(
        streak,
        key="index=48|id=com.app:id/like",
        index=48,
        resource_id="com.app:id/like",
        reason="element is no longer on the screen",
    )
    assert streak[0].count == 2
    reason = breaker_reason(streak, "index=48|id=com.app:id/like")
    assert reason is not None
    # 拒绝理由必须说清「别再报同一个目标」和「换什么」
    assert "re-read" in reason
    assert "different element" in reason


def test_a_different_target_resets_the_streak():
    """换了目标就不算连续失败 —— 账本只记最后一次失败。"""
    streak = record_failure([], key="index=48|id=a", index=48, resource_id="id=a")
    streak = record_failure(streak, key="index=48|id=b", index=48, resource_id="id=b")
    assert len(streak) == 1
    assert streak[0].resource_id == "id=b"
    assert streak[0].count == 1
    assert breaker_reason(streak, "index=48|id=b") is None


def test_breaker_only_blocks_the_exact_same_target():
    streak = record_failure([], key="index=48|id=a", index=48, resource_id="id=a")
    streak = record_failure(streak, key="index=48|id=a", index=48, resource_id="id=a")
    # 别的目标不受影响
    assert breaker_reason(streak, "index=52|id=c") is None
    # 同一个 index 但元素换了 → 也不该被拦
    assert breaker_reason(streak, "index=48|id=z") is None


def test_ledger_is_capped():
    streak: list[FailedTarget] = []
    for i in range(12):
        streak = record_failure(streak, key=f"index={i}|id=x{i}", index=i)
    assert len(streak) <= 5


def test_render_is_empty_when_nothing_failed():
    assert render_failures([]) == ""
    assert render_failures(None) == ""


def test_render_names_the_failed_target():
    streak = record_failure(
        [],
        key="index=48|id=com.app:id/like",
        index=48,
        resource_id="com.app:id/like",
        label="点赞",
    )
    text = render_failures(streak)
    assert "48" in text
    assert "点赞" in text
    assert "no longer on screen" in text


def test_describe_is_human_readable():
    assert "48" in describe(48, "com.app:id/like", "点赞")
    assert "com.app:id/like" not in describe(48, "com.app:id/like", "点赞")
    assert describe(48, "com.app:id/like", "点赞").count("com.app:id/like") == 0


def test_describe_target_has_no_index():
    """🔴 **给模型看的失败事实不许带 index**。

    索引每轮重编号 —— 同一个 48 在下一屏可能是完全不同的元素。带上去，
    模型就会去点一个无关的东西。
    """
    text = describe_target(label="点赞", resource_id="com.app:id/like_button")
    assert "48" not in text
    assert "点赞" in text
    assert "like_button" in text
    assert "com.app:id/" not in text


def test_describe_target_falls_back_to_id_tail():
    """纯图标按钮没有文字 ⇒ 至少还有 resource-id 尾段。"""
    assert "download_btn" in describe_target(label="", resource_id="com.app:id/download_btn")
    assert describe_target(label=None, resource_id=None) == "(unknown target)"


def test_describe_and_describe_target_share_one_identity_format():
    """两处描述用的是**同一份**身份格式 —— ``describe`` 只多一个 index 前缀。

    两份各写一遍的话，迟早一个改了另一个没改 ⇒ 同一个元素在两处显示成
    两个样子，模型对不上账。
    """
    identity = describe_target(label="点赞", resource_id="com.app:id/like_button")
    assert describe(48, "com.app:id/like_button", "点赞") == f"[48] {identity}"
    assert describe(None, "com.app:id/like_button", "点赞") == identity


def test_describe_target_is_keyword_only():
    """🔴 参数顺序不能靠记 —— ``describe`` 的顺序和它**正好相反**。

    ``describe(index, resource_id, label)`` vs ``describe_target(label, resource_id)``：
    位置参数传错不会报错，只会渲染出**另一个元素**的身份 ⇒ 所以强制
    keyword-only（规则：api-keyword-only-params）。
    """
    with pytest.raises(TypeError):
        describe_target("点赞", "com.app:id/like_button")  # type: ignore[misc]


def test_threshold_is_two():
    """阈值就是 2：模型重试一次是合理的，连着两次说明它在原地打转。"""
    assert FAILURE_BREAKER_THRESHOLD == 2
