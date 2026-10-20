"""「按身份在当屏重新定位」的测试。

这块取代了「坐标 + IoU 门限」作为点击的主判据，所以要守住三件事：
1. 按 id 能找到那个元素，拿到的是它**当屏**的位置；
2. 找不到就明确失败（不能退化成「点别的地方」）；
3. 同 id 有多个时，用「离模型当时看到的位置最近的那个」消歧。
"""

from __future__ import annotations

import pytest

from minitap.mobile_use.tools import live_target
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds


def _el(resource_id: str, x: int, y: int, w: int = 100, h: int = 50) -> dict:
    """一条 uiautomator 形状的元素（bounds 是 "[x1,y1][x2,y2]" 字符串）。"""
    return {
        "text": "",
        "content-desc": "",
        "resource-id": resource_id,
        "class": "android.widget.TextView",
        "bounds": f"[{x},{y}][{x + w},{y + h}]",
        "clickable": "true",
        "enabled": "true",
        "visible-to-user": "true",
    }


@pytest.fixture
def fake_dump(monkeypatch):
    """把实时 dump 换成一组固定元素。"""

    def _install(elements: list[dict]):
        class StubController:
            def __init__(self, ctx):
                pass

            async def get_ui_elements(self):
                return elements

        monkeypatch.setattr(live_target, "UnifiedMobileController", StubController)

    return _install


async def _resolve(resource_id, index=0, hint=None, elements=None):
    return await live_target.resolve_live_bounds_by_id(
        ctx=object(),
        resource_id=resource_id,
        index=index,
        hint=hint,
    )


@pytest.mark.asyncio
async def test_finds_the_element_and_returns_its_live_position(fake_dump):
    fake_dump([_el("com.app:id/btn", 100, 200)])

    bounds, error = await _resolve("com.app:id/btn", elements=None)

    assert error == ""
    assert bounds is not None
    assert (bounds.x, bounds.y, bounds.width, bounds.height) == (100, 200, 100, 50)


@pytest.mark.asyncio
async def test_missing_element_fails_loudly(fake_dump):
    """当屏上没有它 → 明确失败，绝不能退化成「点别的地方」。

    失败文案还必须把因果说清楚：**东西没了**（屏幕变了），不是「索引指错人了」。
    真机实测里模型就是把这句话误读成后者，然后在同一个目标上反复打转。
    """
    fake_dump([_el("com.app:id/other", 0, 0)])

    bounds, error = await _resolve("com.app:id/btn")

    assert bounds is None
    assert "com.app:id/btn" in error
    assert "no longer on the screen" in error
    # 要说清「不是索引指错了」和「该怎么办」
    assert "NOT a wrong-index problem" in error
    assert "re-read" in error


@pytest.mark.asyncio
async def test_same_id_uses_the_nearest_to_the_hint(fake_dump):
    """同 id 有多个时，用模型当时看到的位置消歧。"""
    fake_dump(
        [
            _el("com.app:id/textView", 0, 0),
            _el("com.app:id/textView", 500, 900),
        ]
    )
    hint = ElementBounds(x=505, y=905, width=100, height=50)

    bounds, error = await _resolve("com.app:id/textView", hint=hint)

    assert error == ""
    assert bounds is not None
    assert bounds.x == 500


@pytest.mark.asyncio
async def test_same_id_without_hint_falls_back_to_index(fake_dump):
    """没有位置提示时退回「同 id 里的第 N 个」。"""
    fake_dump(
        [
            _el("com.app:id/textView", 0, 0),
            _el("com.app:id/textView", 500, 900),
        ]
    )

    bounds, _ = await _resolve("com.app:id/textView", index=1)

    assert bounds is not None
    assert bounds.x == 500


@pytest.mark.asyncio
async def test_index_beyond_range_does_not_crash(fake_dump):
    fake_dump([_el("com.app:id/btn", 10, 20)])

    bounds, error = await _resolve("com.app:id/btn", index=9)

    assert error == ""
    assert bounds is not None


@pytest.mark.asyncio
async def test_accepts_the_camelcase_resource_id_key(fake_dump):
    """另一条来源用 resourceId，两种键都要认。"""
    fake_dump(
        [
            {
                "resourceId": "com.app:id/btn",
                "bounds": {"x": 7, "y": 8, "width": 20, "height": 10},
            }
        ]
    )

    bounds, error = await _resolve("com.app:id/btn")

    assert error == ""
    assert bounds is not None
    assert (bounds.x, bounds.y) == (7, 8)


@pytest.mark.asyncio
async def test_dump_failure_is_reported_not_raised(monkeypatch):
    class BoomController:
        def __init__(self, ctx):
            pass

        async def get_ui_elements(self):
            raise RuntimeError("uiautomator died")

    monkeypatch.setattr(live_target, "UnifiedMobileController", BoomController)

    bounds, error = await _resolve("com.app:id/btn")

    assert bounds is None
    assert "live dump failed" in error
