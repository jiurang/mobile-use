"""坐标一致性校验的单测。

fixture 原则
------------
这里的元素**必须用真机 dump 出来的原始形状**：`bounds` 是字符串 `"[x1,y1][x2,y2]"`
而不是 dict。上一轮 `get_bounds_for_element` 就是因为测试只喂了 dict 形状，
真实的 Android 字符串形状从没被覆盖过，bug 因此在真机上 0/559 全军覆没而测试全绿。

阈值来自真机实测（scripts/probe_validation_criteria2.py，Redmi Turbo 4，4 屏 68 个可操作节点）：
    归一化 IoU   good: min=1.000 max=1.000    bad: p50=0.322 max=0.816
    阈值 0.90    ->  好坐标接受 100.0%   坏坐标拦截 100.0%

真机回归可重跑：.venv\\Scripts\\python.exe scripts\\verify_bounds_validation.py
"""

from __future__ import annotations

from minitap.mobile_use.tools.bounds_validation import (
    MIN_IOU,
    normalized_iou_for_bounds,
    coordinates_look_like_hierarchy_derived_from_elements,
)
from minitap.mobile_use.tools.types import Target
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds


def _el(bounds: str, **extra) -> dict:
    """构造真机形状的元素：bounds 是字符串。"""
    return {"bounds": bounds, **extra}


def _target(x: int, y: int, w: int, h: int) -> Target:
    return Target(bounds=ElementBounds(x=x, y=y, width=w, height=h))


# 真机形状的假 dump。**必须包含大容器** —— 真机上有 31 个覆盖 >=95% 屏幕的
# 容器节点，漏掉它们会让校验器在小样本上表现良好、在真机上 68/68 全漏。
REAL_DUMP = [
    _el("[0,0][1220,2712]", **{"class": "android.widget.FrameLayout"}),
    _el("[0,0][1220,2660]", **{"class": "android.widget.LinearLayout"}),
    _el("[0,0][1220,2660]", **{"class": "android:id/content"}),
    # 真机上存在的中间容器：面积约 31.6% 屏幕，半尺寸坐标会掉进这里
    _el("[0,1000][1220,1700]", **{"class": "com.app:id/middle_container"}),
    _el("[40,2500][1180,2600]", text="发送", **{"resource-id": "com.app:id/send"}),
    _el("[60,100][300,180]", text="搜索", **{"resource-id": "com.app:id/search"}),
]


class TestAcceptsHierarchyDerivedCoordinates:
    def test_exact_match_is_accepted(self):
        """从层级树原样抄来的坐标必须被接受 —— 这是主路径，误拒就是功能回归。"""
        target = _target(40, 2500, 1140, 100)
        ok, reason = coordinates_look_like_hierarchy_derived_from_elements(target, REAL_DUMP)
        assert ok, f"精确坐标被误拒: {reason}"

    def test_all_actionable_nodes_are_accepted(self):
        for x1, y1, x2, y2 in [(40, 2500, 1180, 2600), (60, 100, 300, 180)]:
            target = _target(x1, y1, x2 - x1, y2 - y1)
            ok, reason = coordinates_look_like_hierarchy_derived_from_elements(target, REAL_DUMP)
            assert ok, f"({x1},{y1}) 被误拒: {reason}"

    def test_slightly_larger_box_still_accepted(self):
        """模型给的框可能比真实节点大一圈。归一化正是为此存在。"""
        target = _target(24, 2490, 1172, 120)  # 外扩约 20%
        ok, reason = coordinates_look_like_hierarchy_derived_from_elements(target, REAL_DUMP)
        assert ok, f"外扩 20% 的框被误拒: {reason}"

    def test_extreme_size_drift_still_accepted(self):
        """±30% 的尺寸漂移都不该误伤 —— 框大小不是可信度的依据。"""
        for scale in (0.7, 1.3):
            w, h = int(1140 * scale), int(100 * scale)
            target = _target(40 - (w - 1140) // 2, 2500 - (h - 100) // 2, w, h)
            ok, reason = coordinates_look_like_hierarchy_derived_from_elements(target, REAL_DUMP)
            assert ok, f"尺寸漂移 {scale} 被误拒: {reason}"


class TestRejectsScreenshotEstimatedCoordinates:
    def test_half_scale_guess_is_rejected(self):
        """模型在 0.5 倍截图上目测、忘换算 —— 必须拦下。

        这是整个改动要拦的场景：坐标仍在屏幕内，不归一化 IoU 只有 0.13。
        """
        target = _target(20, 1250, 570, 50)
        ok, reason = coordinates_look_like_hierarchy_derived_from_elements(target, REAL_DUMP)
        assert not ok, "半分辨率目测坐标竟然通过了校验"
        assert reason and "IoU" in reason

    def test_guess_falling_into_big_container_is_rejected(self):
        """锁住最阴险的情形：坏坐标正好掉进那个 31.6% 屏幕的中间容器。

        朴素判据（「是否落在某节点内」「中心点是否命中」）在这里都会放行，
        因为那个容器确实存在且确实覆盖该点。归一化 IoU 才能识别出来。
        """
        # 搜��框真实位置 (60,100,240,80)，目测成一半 -> (30,50,120,40)
        target = _target(30, 50, 120, 40)
        ok, _ = coordinates_look_like_hierarchy_derived_from_elements(target, REAL_DUMP)
        assert not ok, "掉进大容器的坏坐标被放行了"

    def test_reason_is_actionable_for_the_agent(self):
        """失败原因要写给模型看，必须明确指向 resource_id / text。"""
        target = _target(20, 1250, 570, 50)
        ok, reason = coordinates_look_like_hierarchy_derived_from_elements(target, REAL_DUMP)
        assert not ok
        assert reason is not None
        assert "screenshot" in reason
        assert "resource_id" in reason

    def test_normalized_iou_separates_good_from_bad(self):
        """好坐标归一化后 IoU 恒为 1.000，坏坐标远低于门限。"""
        good = _target(40, 2500, 1140, 100)
        giou, _, _ = normalized_iou_for_bounds(good.bounds, REAL_DUMP)
        assert giou >= MIN_IOU

        bad = _target(20, 1250, 570, 50)
        biou, _, _ = normalized_iou_for_bounds(bad.bounds, REAL_DUMP)
        assert biou < MIN_IOU
        assert giou - biou > 0.1, "好坐标与坏坐标的分离度不足，阈值没有安全边际"


class TestFailOpen:
    """校验器自己出错时必须放行，否则一次 dump 失败就让 agent 全面瘫痪。"""

    def test_no_bounds_passes(self):
        ok, reason = coordinates_look_like_hierarchy_derived_from_elements(Target(), REAL_DUMP)
        assert ok and reason is None

    def test_empty_hierarchy_passes(self):
        target = _target(20, 1250, 570, 50)
        ok, _ = coordinates_look_like_hierarchy_derived_from_elements(target, [])
        assert ok, "空层级树不应阻断点击"

    def test_unparsable_bounds_in_dump_is_skipped(self):
        dirty = [{"bounds": "garbage"}, {"bounds": None}, *REAL_DUMP]
        target = _target(40, 2500, 1140, 100)
        ok, reason = coordinates_look_like_hierarchy_derived_from_elements(target, dirty)
        assert ok, f"脏数据影响了正常匹配: {reason}"


class TestIosShape:
    """iOS / 云设备产出的是 frame dict 形状，校验要同样可用。"""

    def test_frame_dict_shape_is_supported(self):
        ios_dump = [
            {
                "type": "Button",
                "label": "OK",
                "frame": {"x": 0, "y": 0, "width": 390, "height": 844},
            },
            {
                "type": "Button",
                "label": "Cancel",
                "frame": {"x": 10, "y": 700, "width": 200, "height": 60},
            },
        ]
        ok, _ = coordinates_look_like_hierarchy_derived_from_elements(
            _target(10, 700, 200, 60), ios_dump
        )
        assert ok

        ok2, _ = coordinates_look_like_hierarchy_derived_from_elements(
            _target(5, 350, 100, 30), ios_dump
        )
        assert not ok2

