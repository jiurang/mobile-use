"""`resolve_target` 的测试：索引怎么还原成元素，以及越界时怎么报错。

这是「统一索引寻址」真正生效的地方 —— cortex 决策里的 `target.index`
就是靠这个函数变成可点的 bounds 的。
"""

from __future__ import annotations

from types import SimpleNamespace

from minitap.mobile_use.services.ocr.types import OcrText
from minitap.mobile_use.tools.types import Target
from minitap.mobile_use.tools.utils import has_valid_selectors, resolve_target
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds


def _element(index: int, kind: str = "ui", **kwargs):
    payload = {
        "index": index,
        "kind": kind,
        "label": kwargs.get("label", ""),
        "text": kwargs.get("text", ""),
        "resource_id": kwargs.get("resource_id", ""),
        "class_name": kwargs.get("class_name", ""),
        "bounds": ElementBounds(
            x=kwargs.get("x", 10),
            y=kwargs.get("y", 20),
            width=kwargs.get("width", 100),
            height=kwargs.get("height", 50),
        ),
        "flags": kwargs.get("flags", ""),
        "confidence": kwargs.get("confidence"),
    }
    from minitap.mobile_use.utils.screen_elements import ScreenElement

    return ScreenElement(**payload)


def _state(elements=None, ocr_results=None):
    return SimpleNamespace(
        latest_screen_elements=elements,
        latest_ocr_results=ocr_results,
    )


# --------------------------------------------------------------------------
# 索引还原
# --------------------------------------------------------------------------


def test_index_resolves_to_bounds_resource_id_and_text():
    elements = [
        _element(1, label="热门", resource_id="com.app:id/textView"),
        _element(2, kind="ocr", text="9:33", x=79, y=16),
    ]
    state = _state(elements=elements)

    target, error = resolve_target(state, Target(index=2))
    assert error is None
    assert target.bounds is not None
    assert target.bounds.x == 79
    assert target.text == "9:33"

    target, error = resolve_target(state, Target(index=1))
    assert error is None
    assert target.resource_id == "com.app:id/textView"
    assert target.text == "热门"


def test_index_wins_over_the_coordinates_the_model_copied():
    """索引指向的是模型当时看到的那一行 —— 它比模型手抄的坐标可靠。"""
    elements = [_element(1, label="热门", x=160, y=128)]
    state = _state(elements=elements)

    target = Target(index=1, bounds=ElementBounds(x=0, y=0, width=1, height=1))
    resolved, error = resolve_target(state, target)
    assert error is None
    assert resolved.bounds is not None
    assert (resolved.bounds.x, resolved.bounds.y) == (160, 128)


def test_index_only_target_is_a_valid_selector():
    """只给 index 也是合法的定位 —— 工具不该说「没有有效选择器」。"""
    elements = [_element(1)]
    state = _state(elements=elements)

    target, error = resolve_target(state, Target(index=1))
    assert error is None
    assert has_valid_selectors(target)


# --------------------------------------------------------------------------
# 越界不静默
# --------------------------------------------------------------------------


def test_out_of_range_index_returns_an_explicit_error():
    elements = [_element(1)]
    state = _state(elements=elements)

    target, error = resolve_target(state, Target(index=9))
    assert error is not None
    assert "index=9" in error
    assert "1 rows" in error
    # 越界时不要动 target 自己的字段，让工具按老路径继续试
    assert target.index == 9


def test_missing_table_reports_it_instead_of_silently_failing():
    target, error = resolve_target(_state(elements=None), Target(index=1))
    assert error is not None
    assert "no table" in error


# --------------------------------------------------------------------------
# ocr_index：两种世界
# --------------------------------------------------------------------------


def test_ocr_index_is_an_alias_of_index_when_a_table_exists():
    """有元素表时，ocr_index 和 index 是同一个编号空间。"""
    elements = [_element(1, label="热门"), _element(2, kind="ocr", text="9:33")]
    state = _state(elements=elements)

    target, error = resolve_target(state, Target(ocr_index=2))
    assert error is None
    assert target.text == "9:33"
    assert target.bounds is not None
    assert target.bounds.x == 10


def test_ocr_index_falls_back_to_raw_ocr_results_without_a_table():
    """没有元素表（iOS / 云设备）时，老行为不变：索引 latest_ocr_results。"""
    ocr = [
        OcrText(text="9:33", score=0.9, bounds=ElementBounds(x=79, y=16, width=99, height=94)),
        OcrText(text="热搜", score=0.8, bounds=ElementBounds(x=10, y=20, width=50, height=20)),
    ]
    state = _state(elements=None, ocr_results=ocr)

    target, error = resolve_target(state, Target(ocr_index=2))
    assert error is None
    assert target.bounds is not None
    assert (target.bounds.x, target.bounds.y) == (10, 20)


def test_out_of_range_ocr_index_without_a_table_reports_it():
    """老世界的报错文案要跟改造前逐字一致 —— 模型靠它知道索引错在哪。"""
    state = _state(
        elements=None,
        ocr_results=[
            OcrText(text="9:33", score=0.9, bounds=ElementBounds(x=1, y=1, width=1, height=1))
        ],
    )
    target, error = resolve_target(state, Target(ocr_index=7))
    assert error is not None
    assert "out of range" in error
    assert "1..1" in error


def test_missing_ocr_results_without_a_table_says_so():
    state = _state(elements=None, ocr_results=None)
    target, error = resolve_target(state, Target(ocr_index=2))
    assert error is not None
    assert "no OCR results available" in error
