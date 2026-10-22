"""稳定性判据的测试：现行「全量指纹」vs 新的「结构摘要」。

背景（2026-10-22 真机实测，Redmi Turbo 4 / 快手极速版）
------------------------------------------------------
``_confirm_tree_stable`` 用 ``tree_fingerprint`` 判「屏定下来没有」，
而那个指纹哈希的是 ``resource-id + text + class + bounds`` 的**全量**元素。

**视频页每秒都在变** —— 点赞数、倒计时、进度、推荐流内容 ⇒ 指纹**永远不相等**
⇒ 每次都跑满重试次数，最后还是「判定不稳，照用」：三倍的 dump 换同一个结果。

而这个仓库里早就写下了正确答案（``tools/live_target.py``）：

    OCR 的**内容会变**（点赞数在跳、时钟在走）
    **位置比内容稳**

⇒ 结构摘要：只比「可交互元素的身份 + 位置」，**不看内容**。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from minitap.mobile_use.agents.contextor.contextor import ContextorNode
from minitap.mobile_use.context import PerceptionConfig
from minitap.mobile_use.utils.screen_stability import (
    is_interactive,
    structural_change_count,
    structural_digest,
    tree_fingerprint,
)


def _el(
    rid: str = "",
    text: str = "",
    cls: str = "android.widget.FrameLayout",
    bounds: str = "[0,0][100,100]",
    clickable: str = "true",
    **extra: str,
) -> dict:
    """造一个 uiautomator 解析出来的元素 dict（属性值是**字符串**）。"""
    element = {
        "resource-id": rid,
        "text": text,
        "class": cls,
        "bounds": bounds,
        "clickable": clickable,
        "enabled": "true",
        "focusable": "false",
        "scrollable": "false",
        "long-clickable": "false",
        "package": "com.kuaishou.nebula",
    }
    element.update(extra)
    return element


# ---------------------------------------------------------------------------
# 视频页：内容在变，结构没变
# ---------------------------------------------------------------------------


def test_video_page_text_change_is_stable_structurally():
    """视频页的点赞数在跳 ⇒ 旧指纹永远不等，新摘要相等。"""
    before = [
        _el(rid="com.ks:id/like", text="未点赞，点赞1445", bounds="[100,2000][180,2080]"),
        _el(rid="com.ks:id/comment", text="评论2713", bounds="[200,2000][280,2080]"),
    ]
    after = [
        _el(rid="com.ks:id/like", text="未点赞，点赞2161", bounds="[100,2000][180,2080]"),
        _el(rid="com.ks:id/comment", text="评论2715", bounds="[200,2000][280,2080]"),
    ]

    # 旧判据：内容变了 ⇒ 判不稳（视频页上每次都这样）
    assert tree_fingerprint(before) != tree_fingerprint(after)
    # 新判据：身份+位置没变 ⇒ 判稳
    assert structural_digest(before) == structural_digest(after)
    assert structural_change_count(before, after) == 0


def test_video_surface_churn_is_ignored():
    """视频画面在动 —— 它不是可交互元素，不该影响判据。

    注意这里刻意让 SurfaceView 带 ``clickable="true"``：有些 app 会给整块视频
    挂可点属性，光看属性会把它算进摘要 ⇒ 视频每帧一变摘要就变 ⇒ 又回到
    「永远判不稳」。所以要按 class 再排一次。
    """
    before = [_el(rid="com.ks:id/like", text="点赞1445")]
    after = [
        _el(rid="com.ks:id/like", text="点赞1445"),
        _el(
            rid="",
            text="",
            cls="android.view.SurfaceView",
            bounds="[0,500][1220,1800]",
            clickable="true",
        ),
    ]
    assert structural_digest(before) == structural_digest(after)
    assert is_interactive(
        _el(cls="android.view.SurfaceView", clickable="true")
    ) is False


def test_non_interactive_text_churn_is_ignored():
    """非可交互元素的文案变化（时间、状态栏）也不该影响判据。"""
    before = [_el(rid="com.ks:id/like", text="", clickable="false", cls="android.widget.TextView")]
    after = [_el(rid="com.ks:id/like", text="5:28", clickable="false", cls="android.widget.TextView")]
    assert structural_digest(before) == structural_digest(after)


# ---------------------------------------------------------------------------
# 真的在动：必须判不稳
# ---------------------------------------------------------------------------


def test_one_pixel_drift_is_stable():
    """**真机实测到的视频页常态**：可交互元素每采样一次整体漂 1 像素。

    快手视频页 6 次采样（间隔 500ms），每一对都有 1px 位移：
    ``[1008,1677][1220,1914]`` → ``[1007,1676][1220,1914]``

    精确比对会把这种漂移当成「还在动」⇒ 每次都跑满重试 ⇒ 三倍 dump 换同一个结果。
    容差匹配（默认 8px）把它吃掉。
    """
    before = [
        _el(rid="com.ks:id/like", bounds="[1008,1677][1220,1914]"),
        _el(rid="com.ks:id/comment", bounds="[1007,1203][1220,1440]"),
    ]
    after = [
        _el(rid="com.ks:id/like", bounds="[1007,1676][1220,1914]"),
        _el(rid="com.ks:id/comment", bounds="[1006,1202][1220,1439]"),
    ]
    # 精确摘要会认为变了（所以别用它判稳）
    assert structural_digest(before) != structural_digest(after)
    # 容差匹配：判稳
    assert structural_change_count(before, after) == 0
    assert structural_change_count(before, after, tolerance_px=0) > 0


def test_real_moves_still_detected_under_tolerance():
    """容差不能把真实变化也吃掉。"""
    before = [_el(rid="com.ks:id/row", bounds="[0,100][100,300]")]
    after = [_el(rid="com.ks:id/row", bounds="[0,400][100,600]")]  # 滚了 300px
    assert structural_change_count(before, after) > 0

    # caption 展开（实测 102px 位移 + 多一个元素）
    before2 = [_el(rid="com.ks:id/caption", bounds="[35,2215][905,2360]")]
    after2 = [
        _el(rid="com.ks:id/caption", bounds="[35,2113][905,2258]"),
        _el(rid="com.ks:id/special", bounds="[35,2292][920,2346]"),
    ]
    assert structural_change_count(before2, after2) > 0


def test_scroll_moves_bounds_is_unstable():
    """滚动：元素集合不变、只有位置在变 ⇒ 旧判据和��判据都判不稳。"""
    before = [_el(rid="com.ks:id/row1", bounds="[0,100][100,300]")]
    after = [_el(rid="com.ks:id/row1", bounds="[0,400][100,600]")]
    assert tree_fingerprint(before) != tree_fingerprint(after)
    assert structural_digest(before) != structural_digest(after)
    assert structural_change_count(before, after) == 2  # 一个位置变 ⇒ 增删各一条


def test_dialog_appearing_is_unstable():
    """弹窗/翻页：多了一个可点元素 ⇒ 判不稳。"""
    before = [_el(rid="com.ks:id/like", bounds="[100,2000][180,2080]")]
    after = [
        _el(rid="com.ks:id/like", bounds="[100,2000][180,2080]"),
        _el(rid="com.ks:id/confirm", bounds="[500,1200][700,1300]"),
    ]
    assert structural_digest(before) != structural_digest(after)
    assert structural_change_count(before, after) == 1


def test_element_disappearing_is_unstable():
    """元素没了（视频播完被带走）⇒ 判不稳。"""
    before = [
        _el(rid="com.ks:id/like", bounds="[100,2000][180,2080]"),
        _el(rid="com.ks:id/detail", bounds="[100,2100][180,2180]"),
    ]
    after = [_el(rid="com.ks:id/like", bounds="[100,2000][180,2080]")]
    assert structural_digest(before) != structural_digest(after)
    assert structural_change_count(before, after) == 1


def test_scrollbar_fade_is_ignored():
    """MIUI 滚动条淡入淡出 ⇒ 两种判据都不该被骗。"""
    before = [_el(rid="com.ks:id/like", bounds="[100,2000][180,2080]")]
    after = [
        _el(rid="com.ks:id/like", bounds="[100,2000][180,2080]"),
        _el(rid="com.android.settings:id/main_settings_scrollBar", bounds="[1200,0][1220,2530]"),
    ]
    assert tree_fingerprint(before) == tree_fingerprint(after)
    assert structural_digest(before) == structural_digest(after)


# ---------------------------------------------------------------------------
# 可交互性的判定
# ---------------------------------------------------------------------------


def test_interactive_flags_are_strings():
    """uiautomator 的属性值是字符串，不是 bool。"""
    assert is_interactive(_el(clickable="true")) is True
    assert is_interactive(_el(clickable="false")) is False
    assert is_interactive(_el(clickable="false", **{"long-clickable": "true"})) is True
    assert is_interactive(_el(clickable="false", **{"scrollable": "true"})) is True


def test_edit_text_counts_as_interactive():
    """输入框没有 clickable，靠 class 认。"""
    assert is_interactive(_el(cls="android.widget.EditText", clickable="false")) is True
    assert is_interactive(_el(cls="android.widget.TextView", clickable="false")) is False


def test_empty_tree_is_stable_with_itself():
    assert structural_digest([]) == structural_digest([])
    assert structural_digest(None) == structural_digest([])
    assert structural_change_count([], []) == 0


# ---------------------------------------------------------------------------
# contextor 的 structural 路径
# ---------------------------------------------------------------------------


class _FakeController:
    """只实现 ``get_hierarchy_only`` —— structural 路径只用这一个能力。"""

    def __init__(self, *samples: list[dict]):
        self._samples = list(samples)
        self.calls = 0

    async def get_hierarchy_only(self) -> list[dict] | None:
        self.calls += 1
        if not self._samples:
            return None
        return self._samples.pop(0)


def _node(**perception_kwargs) -> ContextorNode:
    node = ContextorNode.__new__(ContextorNode)
    node.ctx = SimpleNamespace(perception=PerceptionConfig(**perception_kwargs))
    return node


@pytest.mark.asyncio
async def test_structural_path_keeps_stable_when_only_text_moved():
    """视频页：内容在跳，结构没动 ⇒ 判稳，且**一次 dump 都不补**。"""
    baseline = [_el(rid="com.ks:id/like", text="点赞1445", bounds="[100,2000][180,2080]")]
    current = [_el(rid="com.ks:id/like", text="点赞2161", bounds="[100,2000][180,2080]")]

    node = _node(stability_mode="structural", stability_extra_dumps=1)
    controller = _FakeController()

    assert (
        await node._verify_structural_stability(
            device_controller=controller,
            state=SimpleNamespace(),
            baseline=baseline,
            current=current,
        )
        is True
    )
    assert controller.calls == 0


@pytest.mark.asyncio
async def test_structural_path_reports_unstable_without_extra_dumps():
    """结构真的变了（弹窗/翻页）⇒ 判不稳；默认不补采，直接照用最新的。"""
    baseline = [_el(rid="com.ks:id/like", bounds="[100,2000][180,2080]")]
    current = [
        _el(rid="com.ks:id/like", bounds="[100,2000][180,2080]"),
        _el(rid="com.ks:id/confirm", bounds="[500,1200][700,1300]"),
    ]

    node = _node(stability_mode="structural", stability_extra_dumps=0)
    controller = _FakeController()

    assert (
        await node._verify_structural_stability(
            device_controller=controller,
            state=SimpleNamespace(),
            baseline=baseline,
            current=current,
        )
        is False
    )
    assert controller.calls == 0


@pytest.mark.asyncio
async def test_structural_path_extra_dump_can_confirm_stability():
    """补采一次，结构**没有再变** ⇒ 表不是过渡态 ⇒ 判稳。

    补采的语义是「表拍完之后结构有没有继续变」：没变 ⇒ 表拍到的是稳定态。
    """
    baseline = [_el(rid="com.ks:id/like", bounds="[100,2000][180,2080]")]
    current = [
        _el(rid="com.ks:id/like", bounds="[100,2000][180,2080]"),
        _el(rid="com.ks:id/confirm", bounds="[500,1200][700,1300]"),
    ]
    # 补采到的结构与建表那次一致 ⇒ 表拍的不是过渡态
    unchanged = [
        _el(rid="com.ks:id/like", bounds="[100,2000][180,2080]"),
        _el(rid="com.ks:id/confirm", bounds="[500,1200][700,1300]"),
    ]

    node = _node(stability_mode="structural", stability_extra_dumps=1, settle_delay_ms=0)
    controller = _FakeController(unchanged)

    assert (
        await node._verify_structural_stability(
            device_controller=controller,
            state=SimpleNamespace(),
            baseline=baseline,
            current=current,
        )
        is True
    )
    assert controller.calls == 1


@pytest.mark.asyncio
async def test_structural_path_extra_dump_still_changing_stays_unstable():
    """补采之后结构**还在变** ⇒ 判不稳（但照样照用最新的，交给执行期重定位）。"""
    baseline = [_el(rid="com.ks:id/like", bounds="[100,2000][180,2080]")]
    current = [
        _el(rid="com.ks:id/like", bounds="[100,2000][180,2080]"),
        _el(rid="com.ks:id/confirm", bounds="[500,1200][700,1300]"),
    ]
    still_moving = [_el(rid="com.ks:id/like", bounds="[100,2000][180,2080]")]

    node = _node(stability_mode="structural", stability_extra_dumps=1, settle_delay_ms=0)
    controller = _FakeController(still_moving)

    assert (
        await node._verify_structural_stability(
            device_controller=controller,
            state=SimpleNamespace(),
            baseline=baseline,
            current=current,
        )
        is False
    )
    assert controller.calls == 1


def test_structural_mode_is_opt_in():
    """默认还是现行判据 —— 改动必须可一键退回。"""
    assert PerceptionConfig().stability_mode == "fingerprint"
    assert PerceptionConfig(stability_mode="structural").stability_mode == "structural"


def test_stability_mode_env_override(monkeypatch):
    """环境变量能一键切到新判据 —— A/B 不用改代码。"""
    monkeypatch.setenv("MOBILE_USE_STABILITY_MODE", "structural")
    assert PerceptionConfig().stability_mode == "structural"

    monkeypatch.setenv("MOBILE_USE_STABILITY_MODE", "nonsense")
    assert PerceptionConfig().stability_mode == "fingerprint"

    monkeypatch.delenv("MOBILE_USE_STABILITY_MODE", raising=False)
    assert PerceptionConfig().stability_mode == "fingerprint"
