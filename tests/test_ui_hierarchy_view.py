"""UI 层级树裁剪渲染器的测试。

真正要守住的是 `minitap.mobile_use.utils.ui_hierarchy_view` 里写明的不变量：
可操作节点一个都不能丢、resource-id 必须原样输出、形状不认识时要退回原来的
JSON dump 而不是让模型什么都看不见。

元素 bounds 采用 uiautomator 的格式："[x1,y1][x2,y2]"。
"""

import json
import re

from minitap.mobile_use.utils.ui_hierarchy_view import (
    _prune,
    _render,
    _truthy,
    render_ui_hierarchy,
)

GEOM_RE = re.compile(r"\((-?\d+),(-?\d+),(\d+)x(\d+)\)")


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


def rows_of(rendered: str) -> list[str]:
    """取元素行，不含 2 行的格式说明头。"""
    return rendered.splitlines()[2:]


def labels_of(rendered: str) -> list[str]:
    pattern = r'^\[\d+\] ("(?:[^"\\]|\\.)*")'
    return [json.loads(m.group(1)) for m in re.finditer(pattern, rendered, re.M)]


class TestFiltering:
    def test_drops_nodes_with_no_signal(self):
        # 没有文本/资源 id/动作标记的裸布局容器，携带不了任何可操作信息。
        out = render_ui_hierarchy([node()])
        assert not [line for line in rows_of(out) if line.startswith("[")]

    def test_keeps_labelled_node(self):
        out = render_ui_hierarchy([node(text="Submit", clickable="true")])
        assert '"Submit"' in rows_of(out)[0]
        assert "id=" not in rows_of(out)[0]

    def test_keeps_full_resource_id_verbatim(self):
        # find_element() 用的是精确等值匹配，所以资源 id 绝不能缩短。
        out = render_ui_hierarchy([node(**{"resource-id": "com.app:id/send"})])
        assert "id=com.app:id/send" in rows_of(out)[0]

    def test_drops_invisible_nodes(self):
        out = render_ui_hierarchy([node(text="Secret", **{"visible-to-user": "false"})])
        assert "Secret" not in out
        # 形状认识但裁剪为空时，绝不能再把原始 JSON 重新吐出来
        assert not out.startswith("Here is the UI hierarchy:")
        assert "(none)" in out

    def test_disabled_clickable_is_not_flagged(self):
        # 给禁用元素标 C，等于诱导模型去点一个注定失败的元素。
        out = render_ui_hierarchy([node(text="Off", clickable="true", enabled="false")])
        assert not out.splitlines()[-1].endswith("C")

    def test_empty_input_renders_nothing(self):
        assert render_ui_hierarchy([]) == ""
        assert render_ui_hierarchy(None) == ""


class TestActionableRecall:
    """agent 能点、能长按、能聚焦的节点，一个都不允许被丢弃。"""

    def test_every_actionable_node_survives(self):
        elements = [
            node(text="Top", clickable="true", bounds="[0,0][200,80]"),
            node(**{"resource-id": "com.app:id/mid"}, bounds="[0,100][400,180]"),
            node(text="Bottom", long_clickable="true", bounds="[0,200][200,280]"),
            node(text="Field", **{"class": "android.widget.EditText"}, bounds="[0,300][600,380]"),
            node(text="List", scrollable="true", bounds="[0,400][600,900]"),
        ]
        out = render_ui_hierarchy(elements)
        assert len(rows_of(out)) == 5
        for expected in ('"Top"', "com.app:id/mid", '"Bottom"', '"Field"', '"List"'):
            assert expected in out

    def test_layout_container_wrapping_many_children_is_dropped(self):
        # 裹着 3 个以上真实元素的无文本容器，就是典型的噪音行。
        elements = [
            node(bounds="[0,0][1000,1000]"),  # 外层容器
            node(text="A", bounds="[10,10][100,60]"),
            node(text="B", bounds="[10,70][100,120]"),
            node(text="C", bounds="[10,130][100,180]"),
        ]
        rows = rows_of(render_ui_hierarchy(elements))
        assert len(rows) == 3
        assert all("id=" not in r for r in rows)

    def test_labelled_container_is_kept(self):
        # android-use 保留带标签的容器，因为标签本身就是信息。
        elements = [
            node(text="Section", bounds="[0,0][1000,300]"),
            node(text="A", bounds="[10,10][100,60]"),
            node(text="B", bounds="[10,70][100,120]"),
            node(text="C", bounds="[10,130][100,180]"),
        ]
        assert '"Section"' in render_ui_hierarchy(elements)


class TestDedup:
    def test_identical_boxes_collapse_to_the_informative_one(self):
        # 父子层叠在同一坐标上：保留带标签的那个。
        elements = [
            node(bounds="[0,0][500,500]"),
            node(text="Real label", bounds="[0,0][500,500]"),
        ]
        rows = rows_of(render_ui_hierarchy(elements))
        assert len(rows) == 1
        assert '"Real label"' in rows[0]

    def test_distinct_boxes_are_both_kept(self):
        elements = [
            node(text="Left", bounds="[0,0][100,100]"),
            node(text="Right", bounds="[500,0][600,100]"),
        ]
        assert len(rows_of(render_ui_hierarchy(elements))) == 2

    def test_dedup_prefers_label_over_bare_resource_id(self):
        # 回归用例：之前用升序排序，结果留下空节点、丢掉了标签。
        elements = [
            node(**{"resource-id": "com.app:id/time_line_1"}, bounds="[950,980][1122,1036]"),
            node(text="9:26", **{"resource-id": "com.app:id/time"}, bounds="[950,980][1122,1036]"),
        ]
        rows = rows_of(render_ui_hierarchy(elements))
        assert len(rows) == 1
        assert '"9:26"' in rows[0]


class TestRendering:
    def test_reading_order_top_then_left(self):
        elements = [
            node(text="Lower", bounds="[0,500][100,600]"),
            node(text="Upper", bounds="[0,100][100,200]"),
            node(text="UpperRight", bounds="[500,100][600,200]"),
        ]
        assert labels_of(render_ui_hierarchy(elements)) == ["Upper", "UpperRight", "Lower"]

    def test_bounds_rendered_as_x_y_w_by_h(self):
        out = render_ui_hierarchy([node(text="Go", bounds="[91,2446][1129,2583]")])
        assert "(91,2446,1038x137)" in out

    def test_flags_reflect_attributes(self):
        out = render_ui_hierarchy(
            [
                node(text="Toggle", checkable="true", selected="true", clickable="true"),
            ]
        )
        assert out.splitlines()[-1].endswith("CKS")

    def test_labels_are_json_escaped(self):
        out = render_ui_hierarchy([node(text='Say "hi"\nnow')])
        assert labels_of(out) == ['Say "hi"\nnow']

    def test_non_ascii_labels_survive(self):
        out = render_ui_hierarchy([node(text="提交订单")])
        assert "提交订单" in out
        assert "\\u" not in out

    def test_header_documents_the_format(self):
        header = render_ui_hierarchy([node(text="X")]).splitlines()[:2]
        assert "id=<resource-id>" in header[0]
        assert "(x,y,WxH)" in header[0]
        assert "C=clickable" in header[1]


class TestFallback:
    def test_unrecognized_shape_falls_back_to_legacy_dump(self):
        # iOS / 云设备客户端产出这种形状，绝不能因此让模型什么都看不见。
        ios = [
            {
                "type": "Button",
                "label": "OK",
                "value": "",
                "frame": {"x": 1, "y": 2, "width": 3, "height": 4},
                "enabled": True,
                "visible": True,
            }
        ]
        out = render_ui_hierarchy(ios)
        assert out.startswith("Here is the UI hierarchy:")
        assert json.loads(out.split("\n", 1)[1]) == ios


class TestHelpers:
    def test_malformed_input_is_not_fatal(self):
        assert _prune([]) == []
        assert _render([]) == ""
        # 解析不出 bounds 就拿不到几何信息，属预期行为
        assert "id=" in render_ui_hierarchy([node(text="x", bounds="garbage")])

    def test_truthy_accepts_both_spellings(self):
        assert _truthy("true") and _truthy(True)
        assert not _truthy("false") and not _truthy(False) and not _truthy(None)
