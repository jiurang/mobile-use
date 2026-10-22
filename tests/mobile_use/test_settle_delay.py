"""动作后沉降等待（感知前的下限）的测试。

真机实测（2026-10-20，Redmi Turbo 4）：上滑换页后，交互控件是**延迟挂载**的
——新页面的 like 类 resource-id 在动作后 2.0s / 2.5s 才出现在层级树里。
抓早了会拿到一棵「控件还没挂上」的树，看起来像「这个按钮不存在」。

它是**下限**不是固定延迟：决策延迟中位 19.2s，绝大多数轮次根本不会触发。
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from minitap.mobile_use.agents.contextor.contextor import ContextorNode
from minitap.mobile_use.context import PerceptionConfig


def _ctx(settle_delay_ms: int) -> SimpleNamespace:
    return SimpleNamespace(perception=PerceptionConfig(settle_delay_ms=settle_delay_ms))


def _state(last_action_at: float | None) -> SimpleNamespace:
    return SimpleNamespace(last_action_at=last_action_at)


@pytest.mark.asyncio
async def test_no_action_means_no_wait():
    node = ContextorNode.__new__(ContextorNode)
    node.ctx = _ctx(3000)
    assert await node._wait_for_settle(_state(None)) == 0.0


@pytest.mark.asyncio
async def test_zero_delay_disables_the_wait():
    node = ContextorNode.__new__(ContextorNode)
    node.ctx = _ctx(0)
    assert await node._wait_for_settle(_state(time.time())) == 0.0


@pytest.mark.asyncio
async def test_action_long_ago_means_no_wait():
    """决策慢的时候（实测中位 19.2s）这个下限根本不会触发 —— 不能变成固定延迟。"""
    node = ContextorNode.__new__(ContextorNode)
    node.ctx = _ctx(3000)
    assert await node._wait_for_settle(_state(time.time() - 30.0)) == 0.0


@pytest.mark.asyncio
async def test_recent_action_waits_only_the_remainder():
    node = ContextorNode.__new__(ContextorNode)
    node.ctx = _ctx(3000)
    # 动作发生在 1 秒前 → 还差 2 秒
    waited = await node._wait_for_settle(_state(time.time() - 1.0))
    assert 1.5 < waited <= 2.1


@pytest.mark.asyncio
async def test_settle_delay_zero_is_allowed_by_the_config():
    """配 0 = 关掉这个下限，是合法配置。"""
    assert PerceptionConfig(settle_delay_ms=0).settle_delay_ms == 0
    # 🔴 2026-10-22：默认从 3000 降到 200。
    #
    # 3000 那个数的依据是「控件延迟挂载，抓早了拿到半成品」，但真机实测
    # （2026-10-22）拆穿了它：**`dump_hierarchy` 自带 waitForIdle**，
    # 屏幕还在动的时候它自己会等（实测滚动场景里它等了 609ms）。
    # 6/6 次试验里不加外部下限时最终内容在 274~2681ms 就位 ⇒
    # 那个 3 秒下限**从来没让树更完整**，只是让每次抓屏晚 3 秒。
    assert PerceptionConfig().settle_delay_ms == 200


# ---------------------------------------------------------------------------
# 树稳定确认（2026-10-22 新增）
# ---------------------------------------------------------------------------


class _FakeController:
    """按脚本吐出若干棵「树」。"""

    def __init__(self, trees: list[list[dict]]):
        self.trees = trees
        self.calls = 0

    async def get_hierarchy_only(self) -> list[dict] | None:
        tree = self.trees[min(self.calls, len(self.trees) - 1)]
        self.calls += 1
        return tree


def _tree(*markers: str) -> list[dict]:
    return [
        {"text": m, "class": "android.widget.TextView", "bounds": "[0,0][10,10]"}
        for m in markers
    ]


def _ctx_with(retries: int) -> SimpleNamespace:
    return SimpleNamespace(
        perception=SimpleNamespace(settle_delay_ms=0, settle_max_retries=retries)
    )


def _state_with_action() -> SimpleNamespace:
    return SimpleNamespace(last_action_at=time.time())


@pytest.mark.asyncio
async def test_stability_check_is_skipped_without_a_screen_changing_action():
    """没动过屏 ⇒ 屏早就是稳的，**不做任何额外 dump**。"""
    node = ContextorNode.__new__(ContextorNode)
    node.ctx = _ctx_with(2)
    controller = _FakeController([_tree("a")])

    stable = await node._confirm_tree_stable(controller, SimpleNamespace(last_action_at=None))

    assert stable is False
    assert controller.calls == 0


@pytest.mark.asyncio
async def test_stability_check_is_skipped_when_retries_are_zero():
    """``settle_max_retries=0`` = 完全不做稳定确认。"""
    node = ContextorNode.__new__(ContextorNode)
    node.ctx = _ctx_with(0)
    controller = _FakeController([_tree("a")])

    assert await node._confirm_tree_stable(controller, _state_with_action()) is False
    assert controller.calls == 0


@pytest.mark.asyncio
async def test_stable_screen_is_detected_after_two_identical_samples():
    """两次采样一样 ⇒ 屏定下来了。"""
    node = ContextorNode.__new__(ContextorNode)
    node.ctx = _ctx_with(2)
    controller = _FakeController([_tree("a", "b")])

    assert await node._confirm_tree_stable(controller, _state_with_action()) is True
    assert controller.calls == 2


@pytest.mark.asyncio
async def test_changing_screen_gives_up_after_the_retry_budget():
    """一直变 ⇒ 用尽预算后返回 False（**不阻塞流程**）。"""
    node = ContextorNode.__new__(ContextorNode)
    node.ctx = _ctx_with(2)
    controller = _FakeController([_tree("a"), _tree("b"), _tree("c")])

    assert await node._confirm_tree_stable(controller, _state_with_action()) is False
    # 1 次初始采样 + 2 次重试
    assert controller.calls == 3


@pytest.mark.asyncio
async def test_unavailable_hierarchy_does_not_break_the_round():
    """拿不到树就当「没法确认」—— 保留现状，**不把流程卡住**。"""
    node = ContextorNode.__new__(ContextorNode)
    node.ctx = _ctx_with(2)

    class _Broken:
        async def get_hierarchy_only(self):
            return None

    assert await node._confirm_tree_stable(_Broken(), _state_with_action()) is False
