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
    assert PerceptionConfig().settle_delay_ms == 3000
