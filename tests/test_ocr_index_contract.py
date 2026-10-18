"""OCR 索引的契约：渲染出来的 `[i]` 必须能原样喂给 `Target.ocr_index`。

这条契约的两端
--------------
`render_ocr_results` 印出的 `[i]` 和 `resolve_ocr_index` 吃的 `index` 是同一个
编号规则的两端。任何一边改了编号方式，`{"ocr_index": i}` 就会静默地点到别的
元素上 —— 没有报错，tap 照样返回 success。

曾经踩过的坑：渲染前按「层级树里已经出现过」剔过重复项，剩下的重新从 1 编号。
于是模型手里的 `[3]` 和原始第 3 条对不上，而它无从得知。抖音搜索页实测
25 条里有 19 条会被剔掉 —— 一个只为了省 token 的优化，代价是让工具参数失真。

所以：**不去重、不重排。识别到什么就渲染什么，编号就是列表位置。**
下面 `TestNoDedupNoRenumber` 和 `TestIndexRoundTrip` 守这一条。
"""

from __future__ import annotations

from minitap.mobile_use.services.ocr import OcrText
from minitap.mobile_use.tools.types import Target
from minitap.mobile_use.utils.ocr_view import render_ocr_results, resolve_ocr_index
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds
from minitap.mobile_use.utils.ui_hierarchy_view import (
    _normalize_label,
    render_hierarchy_and_labels,
)


def node(**overrides) -> dict:
    base = {
        "text": "",
        "content-desc": "",
        "accessibilityText": "",
        "resource-id": "",
        "class": "android.widget.LinearLayout",
        "package": "com.app",
        "clickable": "false",
        "long-clickable": "false",
        "focusable": "false",
        "enabled": "true",
        "checkable": "false",
        "selected": "false",
        "scrollable": "false",
        "visible-to-user": "true",
        "bounds": "[0,0][1000,1000]",
    }
    base.update(overrides)
    return base


def ocr(text: str, x: int = 10, y: int = 20, score: float = 0.95) -> OcrText:
    return OcrText(
        text=text, score=score, bounds=ElementBounds(x=x, y=y, width=100, height=40)
    )


def body(block: str) -> list[str]:
    return [line for line in block.splitlines()[1:] if line.strip()]


class TestNoDedupNoRenumber:
    """不去重、不重排。"""

    def test_every_line_is_rendered(self):
        """即使某条文字在层级树里出现过，也必须留在 OCR 块里。"""
        results = [ocr("全屏观看"), ocr("6.8万"), ocr("树里没有的")]
        block = render_ocr_results(results)
        assert "全屏观看" in block
        assert "6.8万" in block
        assert "树里没有的" in block

    def test_render_takes_no_dedup_argument(self):
        """去重参数已彻底移除 —— 留着就会有人再把它用起来。

        这条守的是「签名级」的不可逆：曾经有 `already_rendered`。
        """
        import inspect

        params = inspect.signature(render_ocr_results).parameters
        assert "already_rendered" not in params
        assert list(params) == ["results"]

    def test_indices_are_contiguous_with_no_gaps(self):
        """编号必须 1..N 连续，不能跳号。"""
        results = [ocr(f"t{i}") for i in range(1, 8)]
        lines = body(render_ocr_results(results))
        assert len(lines) == 7
        for i, line in enumerate(lines, 1):
            assert line.startswith(f"[{i}] "), line

    def test_duplicates_are_not_collapsed(self):
        """同一条文字识别出两次就是两行，各自可点。"""
        results = [ocr("收藏"), ocr("收藏"), ocr("分享")]
        lines = body(render_ocr_results(results))
        assert len(lines) == 3
        assert lines[0].startswith('[1] "收藏"')
        assert lines[1].startswith('[2] "收藏"')
        assert lines[2].startswith('[3] "分享"')

    def test_case_and_whitespace_variants_are_all_kept(self):
        """'Search' / ' search ' / 'search' 是三条，不是零条。"""
        results = [ocr("Search"), ocr(" search "), ocr("search")]
        lines = body(render_ocr_results(results))
        assert len(lines) == 3

    def test_empty_input_renders_nothing(self):
        assert render_ocr_results(None) == ""
        assert render_ocr_results([]) == ""


class TestIndexRoundTrip:
    """`[i]` -> `ocr_index=i` -> 还原出同一条。这是整个特性的命门。"""

    def test_each_index_resolves_to_the_same_item(self):
        results = [ocr(f"t{i}", x=i * 10, y=i * 20) for i in range(1, 6)]
        block = render_ocr_results(results)
        for i, item in enumerate(results, 1):
            assert f"[{i}]" in block
            assert resolve_ocr_index(results, i) is item

    def test_index_is_one_based(self):
        """`[1]` 是第一条 —— 模型抄 `[1]` 就该点到第一条。"""
        results = [ocr("第一条"), ocr("第二条")]
        assert resolve_ocr_index(results, 1).text == "第一条"

    def test_zero_is_rejected_not_treated_as_first(self):
        """0-based 的 `[1]` -> 第一条；反过来 0 必须被拒。

        把 0 当成第一条就是「点错一格」，比报错更糟。
        """
        results = [ocr("第一条"), ocr("第二条")]
        assert resolve_ocr_index(results, 0) is None

    def test_negative_and_overflow_are_rejected(self):
        results = [ocr("a"), ocr("b")]
        assert resolve_ocr_index(results, -1) is None
        assert resolve_ocr_index(results, 3) is None

    def test_none_index_is_rejected(self):
        assert resolve_ocr_index([ocr("a")], None) is None

    def test_no_results_rejected(self):
        """没有 OCR 数据时返回 None，调用方据此报错，不许退化成点别处。"""
        assert resolve_ocr_index(None, 1) is None
        assert resolve_ocr_index([], 1) is None

    def test_resolved_bounds_are_usable_as_target_bounds(self):
        """还原出来的 bounds 能直接进 Target —— 类型就是 ElementBounds。"""
        item = ocr("全屏观看", x=700, y=2000)
        hit = resolve_ocr_index([item], 1)
        target = Target(ocr_index=1).model_copy(update={"bounds": hit.bounds})
        assert target.bounds.x == 700 and target.bounds.y == 2000

    def test_target_accepts_ocr_index_field(self):
        assert Target(ocr_index=3).ocr_index == 3
        assert "ocr_index" in Target.model_fields

    def test_text_index_is_a_different_numbering(self):
        """`ocr_index` 1-based，`text_index` 0-based —— 两者不可互换。

        模型把 `[3]` 误填进 `text_index` 会静默点错，必须在类型上就分开。
        """
        t = Target(text="搜索", text_index=3)
        assert t.text_index == 3
        assert t.ocr_index is None


class TestUiHierarchyLabelsStillAvailable:
    """去重没了，但 `render_hierarchy_and_labels` 的第二返回值仍是公开契约。"""

    def test_returned_labels_are_the_ones_actually_rendered(self):
        text, labels = render_hierarchy_and_labels(
            [
                node(text="提交订单", clickable="true", bounds="[0,0][600,120]"),
                node(text="Cancel", bounds="[0,200][600,260]"),
            ]
        )
        assert "提交订单" in text and "Cancel" in text
        assert labels == {"提交订单", "cancel"}

    def test_empty_input(self):
        assert render_hierarchy_and_labels(None) == ("", set())
        assert render_hierarchy_and_labels([]) == ("", set())

    def test_normalize_strips_whitespace_and_case(self):
        assert _normalize_label("  Full Screen ") == "fullscreen"
        assert _normalize_label("全屏观看") == "全屏观看"

    def test_ocr_keeps_text_whose_node_never_rendered(self):
        """屏幕外节点被丢掉，但 OCR 仍然把它呈上来 —— 现在更不会有去重把它误删。"""
        elements = [
            node(text="看不见的按钮", **{"visible-to-user": "false"}),
            node(text="看得见的", clickable="true", bounds="[0,500][600,560]"),
        ]
        tree, labels = render_hierarchy_and_labels(elements)
        assert "看不见的按钮" not in tree

        block = render_ocr_results([ocr("看不见的按钮"), ocr("看得见的")])
        assert "看不见的按钮" in block
        # 去重已移除：树里有的那条现在也留在 OCR 块里
        assert "看得见的" in block
