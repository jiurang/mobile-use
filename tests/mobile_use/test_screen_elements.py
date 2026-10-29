"""屏幕元素表（统一索引）的测试。

守住的契约
----------
1. 索引 1-based、连续、不跳号，UI 行和 OCR 行**共用一个空间**（不撞号）。
2. 渲染出的 `[i]` 和元素表一一对应（能反解）。
3. **编号在所有过滤之后才分配** —— OCR 去重之后编号不前移、不错位。
   这条是 `ocr_view.py` 记着的踩坑：按层级树去重 OCR 之后编号整体前移，
   模型手里的 `[3]` 和原始第 3 条对不上，而它无从得知。
4. 越界不静默：解析不到就是 None，由调用方明确报错。
5. `render_hierarchy_and_labels` 的输出**逐字节不变**（改造前是既有行为）。
"""

from __future__ import annotations

from types import SimpleNamespace

from minitap.mobile_use.services.ocr.types import OcrText
from minitap.mobile_use.tools.types import Target
from minitap.mobile_use.tools.utils import resolve_target
from minitap.mobile_use.utils.screen_elements import (
    build_screen_elements,
    render_screen_elements,
    resolve_screen_index,
)
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds
from minitap.mobile_use.utils.ui_hierarchy_view import (
    normalize_hierarchy,
    render_hierarchy_and_labels,
)


def _node(
    *,
    text: str = "",
    resource_id: str = "",
    cls: str = "android.widget.TextView",
    bounds: str = "[0,0][100,50]",
    clickable: str = "false",
    focusable: str = "false",
    content_desc: str = "",
) -> dict:
    return {
        "text": text,
        "content-desc": content_desc,
        "resource-id": resource_id,
        "class": cls,
        "bounds": bounds,
        "clickable": clickable,
        "long-clickable": "false",
        "focusable": focusable,
        "checkable": "false",
        "selected": "false",
        "scrollable": "false",
        "enabled": "true",
        "visible-to-user": "true",
    }


def _ocr(text: str, x: int, y: int, score: float = 0.9) -> OcrText:
    return OcrText(
        text=text,
        score=score,
        bounds=ElementBounds(x=x, y=y, width=50, height=20),
    )


# --------------------------------------------------------------------------
# 1. 索引契约
# --------------------------------------------------------------------------


def test_index_is_one_based_continuous_and_unique_across_ui_and_ocr():
    tree = normalize_hierarchy(
        [
            _node(text="热门", resource_id="com.app:id/textView", bounds="[0,0][100,50]"),
            _node(text="同城", resource_id="com.app:id/follow", bounds="[0,60][100,110]"),
        ]
    )
    elements = build_screen_elements(tree, [_ocr("热搜", 0, 120)])

    assert [e.index for e in elements] == [1, 2, 3]
    assert len({e.index for e in elements}) == len(elements)


def test_ui_and_ocr_share_one_index_space():
    """OCR 行和 UI 行交错排，但编号是一套 —— 不会撞号。"""
    tree = normalize_hierarchy(
        [_node(text="热门", resource_id="com.app:id/t", bounds="[0,0][100,50]")]
    )
    elements = build_screen_elements(tree, [_ocr("热搜", 0, 120)])

    kinds = [e.kind for e in elements]
    assert kinds.count("ui") == 1
    assert kinds.count("ocr") == 1
    # 编号连续，不因为混排而出现空洞
    assert sorted(e.index for e in elements) == list(range(1, len(elements) + 1))


def test_elements_are_sorted_by_reading_order():
    tree = normalize_hierarchy(
        [
            _node(text="下", resource_id="com.app:id/b", bounds="[0,100][100,150]"),
            _node(text="上", resource_id="com.app:id/a", bounds="[0,0][100,50]"),
        ]
    )
    elements = build_screen_elements(tree, [_ocr("中间", 0, 50)])
    assert [e.label for e in elements] == ["上", "中间", "下"]


# --------------------------------------------------------------------------
# 2. 渲染与反解
# --------------------------------------------------------------------------


def test_rendered_index_resolves_back_to_the_same_element():
    tree = normalize_hierarchy(
        [
            _node(text="热门", resource_id="com.app:id/textView", bounds="[160,128][256,256]"),
            _node(text="同城", resource_id="com.app:id/follow", bounds="[160,300][256,400]"),
        ]
    )
    elements = build_screen_elements(tree, [_ocr("9:33", 79, 16)])
    text = render_screen_elements(elements)

    for element in elements:
        assert f"[{element.index}]" in text
        assert resolve_screen_index(elements, element.index) is element


def test_rendered_row_keeps_full_resource_id_for_the_code_side():
    """提示里给的是尾段，完整 id 留在元素上（工具做精确匹配要用全名）。"""
    tree = normalize_hierarchy(
        [_node(text="热门", resource_id="com.app:id/textView", bounds="[0,0][100,50]")]
    )
    elements = build_screen_elements(tree, [])
    assert elements[0].resource_id == "com.app:id/textView"

    # 同样的 id 出现两次时，提示里才会给出尾段（用来区分）
    tree2 = normalize_hierarchy(
        [
            _node(text="热门", resource_id="com.app:id/textView", bounds="[0,0][100,50]"),
            _node(text="热门", resource_id="com.app:id/textView", bounds="[0,60][100,110]"),
        ]
    )
    elements2 = build_screen_elements(tree2, [])
    assert all(e.display_id == "textView" for e in elements2)
    assert "id=textView" in render_screen_elements(elements2)
    assert "com.app:id" not in render_screen_elements(elements2)


def test_ocr_row_carries_confidence_and_no_id():
    tree = normalize_hierarchy([])
    elements = build_screen_elements(tree, [_ocr("9:33", 79, 16, score=0.42)])
    rendered = render_screen_elements(elements)
    row = rendered.splitlines()[-1]
    assert row == '[1] "9:33" (79,16,50x20) 0.42'
    # OCR 行没有 id —— header 里那个 "id=" 不算
    assert "id=" not in row


# --------------------------------------------------------------------------
# 3. 去重不能破坏编号（这是踩过的坑）
# --------------------------------------------------------------------------


def test_ocr_dedup_does_not_shift_the_numbering():
    """树里已有的文字不再来一份，但**编号不因此错位**。

    曾经 OCR 块按树去重之后编号整体前移，模型手里的 `[3]` 指向了别的行。
    现在编号在所有过滤之后才分配，所以渲染出的每一行都能原地反解。
    """
    tree = normalize_hierarchy(
        [
            _node(text="热门", resource_id="com.app:id/a", bounds="[0,0][100,50]"),
            _node(text="同城", resource_id="com.app:id/b", bounds="[0,60][100,110]"),
        ]
    )
    # "热门" 与树里重复 → 被去掉；"热搜" 是新的 → 保留
    elements = build_screen_elements(tree, [_ocr("热门", 0, 0), _ocr("热搜", 0, 120)])

    assert [e.label for e in elements] == ["热门", "同城", "热搜"]
    assert [e.index for e in elements] == [1, 2, 3]

    # 渲染出来的每一行的编号都能原地反解回它自己
    for element in elements:
        assert resolve_screen_index(elements, element.index) is element


def test_dedup_ignores_case_and_whitespace():
    tree = normalize_hierarchy(
        [_node(text="Search", resource_id="com.app:id/a", bounds="[0,0][100,50]")]
    )
    elements = build_screen_elements(tree, [_ocr(" search ", 0, 0)])
    assert len(elements) == 1
    assert elements[0].kind == "ui"


# --------------------------------------------------------------------------
# 4. 越界不静默
# --------------------------------------------------------------------------


def test_resolve_returns_none_for_out_of_range_index():
    tree = normalize_hierarchy(
        [_node(text="热门", resource_id="com.app:id/a", bounds="[0,0][100,50]")]
    )
    elements = build_screen_elements(tree, [])

    assert resolve_screen_index(elements, 0) is None
    assert resolve_screen_index(elements, 2) is None
    assert resolve_screen_index(elements, -1) is None
    assert resolve_screen_index(elements, None) is None
    assert resolve_screen_index([], 1) is None


# --------------------------------------------------------------------------
# 5. 回归：旧渲染逐字节不变
# --------------------------------------------------------------------------


def test_legacy_render_is_byte_identical():
    """`render_hierarchy_and_labels` 是既有行为，改造不能改它一个字节。"""
    elements = [
        _node(text="热门", resource_id="com.app:id/textView", bounds="[160,129][253,253]"),
        _node(
            text="",
            resource_id="com.app:id/kcube",
            cls="android.widget.HorizontalScrollView",
            bounds="[158,129][1060,253]",
            focusable="true",
        ),
        _node(text="同城", resource_id="com.app:id/follow", bounds="[309,129][407,253]"),
    ]
    text, labels = render_hierarchy_and_labels(elements)

    expected_lines = [
        'UI elements, one per line: [i] "label" | id=<resource-id> | Class | (x,y,WxH) | flags',
        "flags: C=clickable L=long-press F=focusable K=checkbox S=selected R=scrollable",
        "[1] id=com.app:id/kcube HorizontalScrollView (158,129,902x124) F",
        '[2] "热门" id=com.app:id/textView (160,129,93x124)',
        '[3] "同城" id=com.app:id/follow (309,129,98x124)',
    ]
    assert text.splitlines() == expected_lines
    assert labels == {"热门", "同城"}


def test_normalize_hierarchy_returns_none_for_unknown_shape():
    """形状不认识（iOS / 云设备）时必须明说，不能猜。"""
    assert normalize_hierarchy([{"label": "x", "frame": {"x": 1, "y": 2}}]) is None
    assert normalize_hierarchy([]) == []
    assert normalize_hierarchy(None) == []


# --------------------------------------------------------------------------
# 6. 系统状态栏：默认剔掉，但前台是 systemui 时必须留
# --------------------------------------------------------------------------


def _status_bar_node(text: str = "9:33"):
    return _node(
        text=text,
        resource_id="com.android.systemui:id/clock",
        cls="android.widget.TextView",
        bounds="[79,16][178,110]",
    )


def test_systemui_rows_are_dropped_for_a_normal_foreground_app():
    tree = normalize_hierarchy([_status_bar_node(), _node(text="应用", bounds="[0,200][100,300]")])
    elements = build_screen_elements(tree, None, foreground_package="com.some.app")

    assert all(not e.resource_id.startswith("com.android.systemui:") for e in elements)
    assert [e.label for e in elements] == ["应用"]
    assert [e.index for e in elements] == [1]


def test_systemui_rows_are_kept_when_systemui_is_foreground():
    """前台就是 systemui（通知栏、状态栏设置页）时，状态栏正是要点的目标。"""
    tree = normalize_hierarchy([_status_bar_node(), _node(text="应用", bounds="[0,200][100,300]")])
    elements = build_screen_elements(tree, None, foreground_package="com.android.systemui")

    assert any(e.resource_id == "com.android.systemui:id/clock" for e in elements)


def test_systemui_rows_are_kept_when_foreground_is_unknown():
    """不知道前台是谁时保守地留 —— 判错成「不需要」会让任务点不到。"""
    tree = normalize_hierarchy([_status_bar_node()])
    elements = build_screen_elements(tree, None, foreground_package=None)

    assert any(e.resource_id == "com.android.systemui:id/clock" for e in elements)


def test_dropping_systemui_lets_the_ocr_row_of_that_text_back_in():
    """状态栏树行被剔了，状态栏的文字会以 OCR 行的形式回来 —— 不能凭空少一行。"""
    tree = normalize_hierarchy(
        [
            _status_bar_node(text="9:33"),
            _node(text="应用", bounds="[0,200][100,300]"),
        ]
    )
    ocr = [_ocr("9:33", 79, 16)]

    with_bar = build_screen_elements(tree, ocr, foreground_package="com.some.app")
    assert [e.label for e in with_bar] == ["9:33", "应用"]
    # 那条 9:33 现在是 OCR 行（kind='ocr'），不是树行
    assert with_bar[0].kind == "ocr"

    # 不剔状态栏时，OCR 那条被树行去重掉
    kept = build_screen_elements(tree, ocr, foreground_package="com.android.systemui")
    assert [e.label for e in kept] == ["9:33", "应用"]
    assert kept[0].kind == "ui"


# --------------------------------------------------------------------------
# 7. id 尾段：只在必要行上给
# --------------------------------------------------------------------------


def test_id_tail_is_shown_only_when_needed():
    tree = normalize_hierarchy(
        [
            # 有唯一文字 → 不需要 id
            _node(text="热门", resource_id="com.app:id/tab", bounds="[0,0][100,50]"),
            # 没有文字 → id 是它唯一的描述
            _node(text="", resource_id="com.app:id/icon", bounds="[0,60][100,110]"),
            # 标签本屏重复 → id 用来区分
            _node(text="更多", resource_id="com.app:id/more_a", bounds="[0,120][100,170]"),
            _node(text="更多", resource_id="com.app:id/more_b", bounds="[0,180][100,230]"),
        ]
    )
    elements = build_screen_elements(tree, None)

    # 「热门」唯一 → 不给
    assert elements[0].display_id == ""
    # 「更多」重复 → 两行都给，用来区分
    assert len([e for e in elements if e.label == "更多" and e.display_id]) == 2
    # 没有文字的那行 → 给
    assert any(e.display_id == "icon" for e in elements)
    # 完整 id 永远留在代码侧
    assert all(e.resource_id for e in elements if e.resource_id)


def test_display_id_is_the_tail_not_the_full_package():
    tree = normalize_hierarchy(
        [_node(text="", resource_id="com.kuaishou.nebula:id/search_input", bounds="[0,0][100,50]")]
    )
    elements = build_screen_elements(tree, None)
    assert elements[0].display_id == "search_input"
    assert elements[0].resource_id == "com.kuaishou.nebula:id/search_input"
    # 提示里不该出现完整包名
    assert "com.kuaishou.nebula" not in render_screen_elements(elements)


def test_rendered_id_is_absent_for_unique_labelled_rows():
    tree = normalize_hierarchy(
        [_node(text="热门", resource_id="com.app:id/tab", bounds="[0,0][100,50]")]
    )
    rendered = render_screen_elements(build_screen_elements(tree, None))
    assert "id=" not in rendered.splitlines()[-1]


# --------------------------------------------------------------------------
# 8. 「纯装饰行」裁不裁？（降本：元素表是 executor prompt 里最大的不可缓存块）
#
# 候选判据：「不可交互（flags 里没有 C/L/F/K/S）+ 无文字（label 和 text 都空）」
# 的行看着像纯浪费 —— 实测 113 行的屏里 42% 是这种。
#
# **实测结论：不裁。** 真机逐个点过（Redmi Turbo 4，12 个 App 页面，
# 126 行候选逐个 adb 点）：**17 行（13.5%）点了之后屏幕真的变了**。
# 命中的都是模型会想点的东西 —— 小米天气的 7 天预报图标、城市切换块、
# 淘宝的搜索框、小红书的卡片。原因很直白：那些 App 用 Canvas / 自定义 View
# 画可点区域，dump 里既没有 clickable 也没有文字，
# 「没有交互标记」根本推不出「点它没反应」。
#
# 三个「退一步」的收窄判据实测下来**各裁 0 行**（82 行候选里最小面积
# 845px²，没有一个「小到不可能是点击目标」；82 行全都带 id；
# 层级树拍平后没有一行是别人的几何父容器）—— 不存在「只裁一部分」的落点。
#
# 所以下面这几条不是「裁剪的实现」，是**把「不裁」这个决定锁住**，
# 免得以后有人凭直觉再加一次裁剪。细节见
# `utils/screen_elements.py` 的模块文档不变量 #8。
# --------------------------------------------------------------------------


def _decor_node(resource_id: str = "com.app:id/decor_container", bounds: str = "[0,60][100,110]"):
    """一行「纯装饰」：有 id、有坐标，但没文字、也没任何交互标记。"""
    return _node(text="", resource_id=resource_id, bounds=bounds)


def test_decorative_row_is_not_dropped():
    """T26：不可交互 + 无文字的行**不裁**。

    真机实测：这类行里有 13.5% 是真能点的（Canvas / 自定义 View 画的可点区域，
    dump 里既没有 clickable 也没有文字）。裁掉等于让模型点不到它们。

    这条测试的作用是**锁住「不裁」这个决定**。
    """
    tree = normalize_hierarchy(
        [
            _node(text="热门", resource_id="com.app:id/tab", bounds="[0,0][100,50]"),
            _decor_node(),
        ]
    )
    elements = build_screen_elements(tree, None)

    assert len(elements) == 2
    assert any(e.resource_id == "com.app:id/decor_container" for e in elements)
    # 它还在提示里 —— 而且 id 尾段是模型认出「这是个可点的东西」的唯一线索
    assert "decor_container" in render_screen_elements(elements)
    assert [
        e.display_id for e in elements if e.resource_id.endswith("decor_container")
    ] == ["decor_container"]


def test_interactive_row_without_text_is_kept():
    """没有文字但**可交互**的行必须留着 —— 图标按钮就是这种。

    它是模型唯一能点的目标，裁掉等于让图标按钮点不到。
    """
    tree = normalize_hierarchy(
        [
            _node(
                text="",
                resource_id="com.app:id/icon_btn",
                bounds="[0,60][100,110]",
                clickable="true",
            )
        ]
    )
    elements = build_screen_elements(tree, None)

    assert len(elements) == 1
    assert elements[0].resource_id == "com.app:id/icon_btn"
    assert "C" in elements[0].flags


def test_scrollable_row_without_text_is_kept():
    """R（scrollable）算可交互。

    `agents/executor/prompts/messages/screen-elements.md` 里给模型的说明写着
    「`R` scrollable (**use its bounds for swipes**)」—— 模型靠 R 行的
    bounds 决定往哪划。R 行往往没有文字（那是个纯滚动容器），
    当装饰裁掉就等于把「这片能滚」和它的边界一起删了。
    """
    # RecyclerView 在 uiautomator 里是 scrollable=true
    raw = [
        {
            "text": "",
            "content-desc": "",
            "resource-id": "com.app:id/list",
            "class": "androidx.recyclerview.widget.RecyclerView",
            "bounds": "[0,60][100,110]",
            "clickable": "false",
            "long-clickable": "false",
            "focusable": "false",
            "checkable": "false",
            "selected": "false",
            "scrollable": "true",
            "enabled": "true",
            "visible-to-user": "true",
        }
    ]
    elements = build_screen_elements(normalize_hierarchy(raw), None)

    assert len(elements) == 1
    assert "R" in elements[0].flags


def test_non_interactive_row_with_text_is_kept():
    """T27：不可交互但**有文字**的行要留着 —— 它是模型的上下文。

    模型靠这些行读懂屏幕（「热门」「同城」是标签还是按钮）。
    裁掉它们等于让模型看不见屏幕上的字。
    """
    tree = normalize_hierarchy(
        [
            _node(text="热门", resource_id="com.app:id/a", bounds="[0,0][100,50]"),
            _node(text="同城", resource_id="com.app:id/b", bounds="[0,60][100,110]"),
        ]
    )
    elements = build_screen_elements(tree, None)

    assert [e.label for e in elements] == ["热门", "同城"]
    # 它们确实不可交互（没有 C/L/F/K/S/R）
    assert all(not (set(e.flags) & set("CLFKSR")) for e in elements)


def test_ocr_row_is_never_dropped():
    """T28：OCR 行永远不裁。

    OCR 引擎不给交互标记（flags 恒空），所以「flags 空 = 不可交互」
    这条推论**对 OCR 行是错的**。它有文字，本来就留得住 —— 但这里
    显式守住，免得以后的改动一脚踩进来。
    """
    tree = normalize_hierarchy(
        [
            _decor_node(),
            _node(text="热门", resource_id="com.app:id/a", bounds="[0,120][100,170]"),
        ]
    )
    elements = build_screen_elements(tree, [_ocr("热搜", 0, 200)])

    ocr_rows = [e for e in elements if e.kind == "ocr"]
    assert len(ocr_rows) == 1
    assert ocr_rows[0].text == "热搜"
    # OCR 行没有交互标记 —— 正因为如此，它绝不能被当成装饰
    assert ocr_rows[0].flags == ""


def test_index_is_consistent_with_the_table():
    """T29：编号自洽 —— 表里的 `[i]` 和表里的第 i 行对得上。

    模型报 `[i]`、工具 `resolve_target` 按同一个下标找元素，
    两边看的必须是**同一张表**（`state.latest_screen_elements`）。
    """
    tree = normalize_hierarchy(
        [
            _decor_node(bounds="[0,0][100,50]"),
            _node(text="热门", resource_id="com.app:id/a", bounds="[0,60][100,110]"),
            _decor_node(resource_id="com.app:id/decor2", bounds="[0,120][100,170]"),
            _node(text="同城", resource_id="com.app:id/b", bounds="[0,180][100,230]"),
        ]
    )
    elements = build_screen_elements(tree, [_ocr("热搜", 0, 260)])
    rendered = render_screen_elements(elements)

    # 编号连续、1-based、不跳号
    assert [e.index for e in elements] == list(range(1, len(elements) + 1))

    for element in elements:
        # 渲染出来的 [i] 就是这一行的编号
        assert f"[{element.index}]" in rendered
        # 工具侧按同一个编号解析回**同一个**元素
        assert resolve_screen_index(elements, element.index) is element
        # 解析到别的行 = 编号错位
        if element.index > 1:
            assert resolve_screen_index(elements, element.index - 1) is not element


def test_resolve_target_uses_the_same_numbering_as_the_rendered_table():
    """T29 续：模型报 `[i]` → 工具 `resolve_target` 解析到同一个元素。"""
    tree = normalize_hierarchy(
        [
            _decor_node(bounds="[0,0][100,50]"),
            _node(text="热门", resource_id="com.app:id/a", bounds="[0,60][100,110]"),
        ]
    )
    elements = build_screen_elements(tree, None)
    state = SimpleNamespace(latest_screen_elements=elements, latest_ocr_results=[])

    # 装饰行也在表里（不裁）→ 它按阅读顺序排在第 1 行，「热门」是第 2 行
    assert [e.label for e in elements] == ["", "热门"]

    target, error = resolve_target(state, Target(index=2))
    assert error is None
    assert target.text == "热门"
    assert target.bounds == elements[1].bounds

    # 越界仍然不静默
    _, error = resolve_target(state, Target(index=3))
    assert error is not None
    assert "index=3" in error


def test_ocr_index_branch_is_unaffected():
    """T30：`resolve_target` 的 `ocr_index` 分支不受元素表裁剪的影响。

    两条路径：
    1. 有元素表 → `ocr_index` 是 `index` 的别名（同一个编号空间）
    2. 没有元素表（iOS / 云设备）→ `ocr_index` 索引 `latest_ocr_results`，
       这是改造前的行为，元素表怎么裁都碰不到它
    """
    # ---- 路径 1：有表 ----
    tree = normalize_hierarchy(
        [
            _decor_node(bounds="[0,0][100,50]"),
            _node(text="热门", resource_id="com.app:id/a", bounds="[0,60][100,110]"),
        ]
    )
    elements = build_screen_elements(tree, [_ocr("热搜", 0, 120)])
    state = SimpleNamespace(latest_screen_elements=elements, latest_ocr_results=[])

    # 装饰行、UI 行、OCR 行共用一套编号：1 / 2 / 3
    assert [e.label for e in elements] == ["", "热门", "热搜"]

    target, error = resolve_target(state, Target(ocr_index=3))
    assert error is None
    assert target.text == "热搜"

    # ---- 路径 2：没有表 ----
    ocr = [
        OcrText(text="9:33", score=0.9, bounds=ElementBounds(x=79, y=16, width=99, height=94)),
        OcrText(text="热搜", score=0.8, bounds=ElementBounds(x=10, y=20, width=50, height=20)),
    ]
    legacy_state = SimpleNamespace(latest_screen_elements=None, latest_ocr_results=ocr)

    target, error = resolve_target(legacy_state, Target(ocr_index=2))
    assert error is None
    assert target.text == "热搜"
    assert target.bounds == ocr[1].bounds
