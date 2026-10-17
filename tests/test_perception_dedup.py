"""UI 层级树 与 OCR 两块感知文本之间的组合契约。

背景（2026-10-17 真机实测，抖音搜索页）：
    层级树渲染 116 行 / 3,372 tok，其中 `id=` 一项就占 45%
    OCR 25 条 / 642 tok —— 但其中 **19 条是层级树里已经有的文字**，
    只有 6 条是树里没有的，而且那 6 条全是误识别噪声

所以 `render_hierarchy_and_labels()` 会把「真正进了 prompt 的 label」一起返回，
`render_ocr_results(already_rendered=...)` 据此把重复的那部分剔掉。

这个组合最危险的地方
--------------------
树和 OCR 是**前后两道筛子**，共用同一份渲染结果才安全。如果 OCR 拿「原始 dump
的全集」去判重，就会出现这条文字**两边都没有**：

    某条文字在原始树里存在 → OCR 认为「树里已经有了」→ 剔掉
    但它所在的节点被树过滤掉了 → 它其实没进 prompt
    => 模型完全看不见这条文字

下面 `test_ocr_keeps_text_whose_node_never_rendered` 就是守这一条。
"""

from __future__ import annotations

from minitap.mobile_use.services.ocr import OcrText
from minitap.mobile_use.utils.ocr_view import render_ocr_results
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


class TestRenderedLabels:
    def test_returned_labels_are_the_ones_actually_rendered(self):
        """第二返回值必须只包含真正出现在文本块里的 label。"""
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

    def test_fallback_shape_returns_no_labels(self):
        """非 Android 形状退回 JSON dump 时没有可用的 label 集合。

        这里必须返回空集合而不是全集：dump 里什么字段都有，
        但那不是「模型看到的文字」。
        """
        ios = [{"type": "Button", "label": "OK", "frame": {"x": 1, "y": 2}}]
        text, labels = render_hierarchy_and_labels(ios)
        assert text.startswith("Here is the UI hierarchy:")
        assert labels == set()

    def test_normalize_strips_whitespace_and_case(self):
        assert _normalize_label("  Full Screen ") == "fullscreen"
        assert _normalize_label("全屏观看") == "全屏观看"


class TestOcrDedup:
    def test_drops_text_already_in_the_rendered_tree(self):
        tree, labels = render_hierarchy_and_labels(
            [
                node(text="全屏观看", clickable="true", bounds="[0,0][600,120]"),
                node(text="6.8万", bounds="[0,200][600,260]"),
            ]
        )
        block = render_ocr_results(
            [ocr("全屏观看"), ocr("6.8万"), ocr("树里没有的")], already_rendered=labels
        )
        assert "全屏观看" not in block
        assert "6.8万" not in block
        assert "树里没有的" in block

    def test_tolerates_case_and_whitespace_mismatch(self):
        """OCR 引擎和 uiautomator 对同一段文字的切分经常不一样。"""
        tree, labels = render_hierarchy_and_labels([node(text="Search")])
        block = render_ocr_results([ocr(" search ")], already_rendered=labels)
        assert block == ""

    def test_does_not_drop_on_substring_match(self):
        """子串包含不算重复。

        "512" 和 "6.8万" 不是同一段话；把短的丢掉会真的弄丢可点的目标。
        """
        tree, labels = render_hierarchy_and_labels([node(text="6.8万")])
        block = render_ocr_results([ocr("512")], already_rendered=labels)
        assert "512" in block

    def test_no_dedup_when_labels_not_supplied(self):
        """不传 already_rendered 时行为和以前完全一样。"""
        block = render_ocr_results([ocr("全屏观看")])
        assert "全屏观看" in block

    def test_indices_are_renumbered_after_dedup(self):
        tree, labels = render_hierarchy_and_labels(
            [
                node(text="A", bounds="[0,0][600,120]"),
                node(text="B", bounds="[0,200][600,260]"),
            ]
        )
        block = render_ocr_results([ocr("A"), ocr("X"), ocr("B"), ocr("Y")], already_rendered=labels)
        lines = [line for line in block.splitlines()[1:] if line.strip()]
        assert len(lines) == 2
        assert lines[0].startswith('[1] "X"')
        assert lines[1].startswith('[2] "Y"')

    def test_everything_deduped_yields_empty_block(self):
        """全被去重掉时返回空串，调用方据此不发那条 HumanMessage。"""
        tree, labels = render_hierarchy_and_labels([node(text="A")])
        assert render_ocr_results([ocr("A")], already_rendered=labels) == ""


class TestNoInformationHole:
    def test_ocr_keeps_text_whose_node_never_rendered(self):
        """**这条是整个组合的核心防线。**

        屏幕外节点被 _normalize 丢掉，但它在原始 dump 里是有文字的。
        如果 OCR 拿原始全集判重，这条文字就会被两边同时抹掉。
        """
        elements = [
            node(text="看不见的按钮", **{"visible-to-user": "false"}),
            node(text="看得见的", clickable="true", bounds="[0,500][600,560]"),
        ]
        tree, labels = render_hierarchy_and_labels(elements)

        # 树里确实没有它
        assert "看不见的按钮" not in tree
        assert "看不见的按钮" not in labels

        # 所以 OCR 必须把它留下来，模型还能知道屏幕上写着这句话
        block = render_ocr_results(
            [ocr("看不见的按钮"), ocr("看得见的")], already_rendered=labels
        )
        assert "看不见的按钮" in block
        assert "看得见的" not in block
