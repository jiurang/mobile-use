"""屏幕元素表：把 UI 层级树和 OCR 文本合成**一张**带统一索引的表。

为什么要有这张表
----------------
1. **一个索引空间**。改造前 UI 块和 OCR 块各带一套 `[i]`，模型得先判断
   「这个编号属于哪张表」，两边还可能撞号。合成一张表之后编号全局唯一，
   模型只报编号，代码负责反解。
2. **索引必须能反解**。`Target.index` 要还原成「模型当时看到的那屏上的
   那个元素」。所以**渲染给模型的那张表**和**执行时解析用的那张表**必须是
   同一份对象 —— 不能执行时重新 dump 再重新编号（屏可能已经变了）。
   于是这张表要存进 state，一路活到 executor。
3. **编号在所有过滤之后才分配**。`utils/ocr_view.py` 记着一次踩坑：按层级树
   去重 OCR 之后编号整体前移，模型手里的 `[3]` 和原始第 3 条对不上，而它
   无从得知。这里所有过滤（剔除、去重、排序）都在编号之前完成，编号一次性
   分配，契约天然自洽。

不变量（改这块之前先读）
------------------------
1. **索引 1-based、连续、不跳号**，UI 行和 OCR 行共用一个空间，全局唯一。
2. **越界不静默**。解析不到就返回 None，由调用方明确报错 —— 绝不能退化成
   「点别的地方」。这条和 `ocr_view.resolve_ocr_index` 是同一条纪律。
3. **每一行都能点**：行里一定带 bounds，点击走坐标。
4. **完整 resource_id 只留在代码侧**。提示里只给尾段（`id=search_input`），
   完整值存在 `ScreenElement.resource_id` 里，供工具做精确匹配 ——
   这样既省 token，又不用改 `find_element` 的等值匹配契约。
5. **树和 OCR 去重时保留树的那一行**。树里的文字是系统给的，更权威；
   OCR 那条是同义重复，留着只会让模型犹豫点哪个。
6. **系统状态栏默认剔掉**，但**前台就是 systemui 时必须留**。实测 256 屏里
   状态栏占 26% 的行 / 28% 的 token，而其中**带操作标记的是 0 条** ——
   全是时钟、电量、信号这类死容器，状态栏的文字又已经被 OCR 覆盖。
   但「下拉通知栏」「看电量」这类任务前台就是 systemui，那时必须留。
7. **id 尾段只在必要行上给**：没有文字的行（id 是它唯一的描述），
   或者本屏标签重复的行（id 是区分它们的唯一线索）。
   实测这两类只占全部 id 行的 70%，省下来的却是全部 id 字段的绝大部分。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from minitap.mobile_use.services.ocr.types import OcrText
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds

logger = get_logger(__name__)

ElementKind = Literal["ui", "ocr"]

# 系统状态栏的包名前缀。它下面的节点全是时钟/电量/信号这类死容器。
_SYSTEMUI_PACKAGE = "com.android.systemui:"

# 合并表的表头。必须和 cortex 提示词里「Element Targeting」那一段说的一致，
# 否则模型会按旧的格式去读。
_HEADER = (
    'Screen elements, one per line: [i] "text" | id=<id tail> | Class | (x,y,WxH) | flags\n'
    "flags: C=clickable L=long-press F=focusable K=checkbox S=selected R=scrollable\n"
    "One numbering for all rows. Rows without id= are OCR text. id= is the tail of the\n"
    "resource-id (package omitted), shown only when the text is missing or repeats."
)

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_label(text: str) -> str:
    """归一化：去空白 + 小写。

    OCR 引擎和 uiautomator 对同一段可见文字的切分、空格、大小写经常不一致
    （"Search" vs " search "）。判断「OCR 这条是不是树里已经有了」必须先
    归一化，否则同一句话会两边都出现，白白多花 token。
    """
    return _WHITESPACE_RE.sub("", text).strip().lower()


def _id_tail(resource_id: str) -> str:
    """`com.app:id/search_input` -> `search_input`。

    只给尾段是安全的：id 在提示里**已经不是工具参数**了（工具认的是索引），
    它只是给模型看的「这大概是个什么控件」的线索。完整的 id 留在代码侧，
    工具做精确匹配时用的是全名。
    """
    tail = resource_id.split(":", 1)[-1] if ":" in resource_id else resource_id
    return tail.rsplit("/", 1)[-1] if "/" in tail else tail


def _should_drop_systemui(foreground_package: str | None) -> bool:
    """要不要把系统状态栏的行剔掉。

    前台就是 systemui 时**必须留** —— 那说明屏幕上的就是通知栏/状态栏设置页，
    任务要点的正是它们。前台不知道（None）时也保守地留。
    """
    if not foreground_package:
        return False
    return not foreground_package.startswith("com.android.systemui")


class ScreenElement(BaseModel):
    """屏幕上的一个可寻址对象。

    UI 层级树节点和 OCR 文本行共用这个形状，靠 `kind` 区分。
    `resource_id` 存**完整**值（`com.app:id/search_input`），只有代码侧看得到；
    提示里渲染的是 `display_id`（尾段，可能为空）。
    """

    index: int = Field(description="1-based index, unique across UI rows and OCR rows")
    kind: ElementKind
    label: str = Field(default="", description="Visible text of a UI node")
    text: str = Field(default="", description="OCR text (kind='ocr')")
    resource_id: str = Field(default="", description="Full resource-id (kind='ui')")
    display_id: str = Field(
        default="",
        description="Resource-id tail shown in the prompt; empty when not needed",
    )
    class_name: str = Field(default="", description="Android class short name (kind='ui')")
    bounds: ElementBounds
    flags: str = Field(default="", description="C/L/F/K/S/R flags (kind='ui')")
    confidence: float | None = Field(default=None, description="OCR score (kind='ocr')")


def _element_from_tree_row(row: dict) -> ScreenElement | None:
    box = row.get("bounds")
    if not box:
        return None
    return ScreenElement(
        index=0,
        kind="ui",
        label=row.get("label") or "",
        resource_id=row.get("resource_id") or "",
        class_name=row.get("class_name") or "",
        bounds=ElementBounds(
            x=int(box["x"]),
            y=int(box["y"]),
            width=int(box["width"]),
            height=int(box["height"]),
        ),
        flags=row.get("flags") or "",
    )


def build_screen_elements(
    tree_rows: list[dict] | None,
    ocr_results: list[OcrText] | None = None,
    foreground_package: str | None = None,
) -> list[ScreenElement]:
    """把裁剪后的层级树和 OCR 结果合成一张带统一索引的表。

    顺序：先按屏幕位置（y 再 x）混排成「用户看到的样子」，**最后**才编号。
    编号是最后一步，是这张表能自洽的唯一原因。

    Args:
        tree_rows: `utils.ui_hierarchy_view.normalize_hierarchy()` 的输出。
        ocr_results: OCR 引擎的原始输出（**不排序、不改编号**）。
        foreground_package: 当前前台应用的包名。用来决定系统状态栏要不要留 ——
            前台就是 systemui 时（通知栏、状态栏设置页）必须保留。
    """
    elements: list[ScreenElement] = []

    drop_systemui = _should_drop_systemui(foreground_package)
    kept_rows = [
        row
        for row in (tree_rows or [])
        if not (drop_systemui and (row.get("resource_id") or "").startswith(_SYSTEMUI_PACKAGE))
    ]
    dropped = len(tree_rows or []) - len(kept_rows)

    for row in kept_rows:
        element = _element_from_tree_row(row)
        if element is not None:
            elements.append(element)

    # OCR 去重：文字已经在树里出现过的就不再来一份。
    # 保留树的那一行 —— 系统给的文字更权威，还带 id 和 flags。
    tree_labels = {_normalize_label(e.label) for e in elements if e.label}
    tree_labels.discard("")
    for ocr in ocr_results or []:
        text = (ocr.text or "").strip()
        if not text:
            continue
        if _normalize_label(text) in tree_labels:
            continue
        elements.append(
            ScreenElement(
                index=0,
                kind="ocr",
                label=text,
                text=text,
                bounds=ocr.bounds,
                confidence=ocr.score,
            )
        )

    # 阅读顺序：先上后下、先左后右。Python 的排序是稳定的，
    # 同位置的元素保持「树在前、OCR 在后」的插入顺序。
    elements.sort(key=lambda e: (e.bounds.y, e.bounds.x))

    # id 尾段只在必要行上给：
    #   - 没有文字的行：id 是它唯一的描述
    #   - 本屏标签重复的行：id 是区分它们的唯一线索
    #   实测这两类只占全部 id 行的 70%，却省掉 id 字段的大部分开销。
    label_counts = Counter(e.label for e in elements if e.label)
    for element in elements:
        if not element.resource_id:
            continue
        needed = not element.label or label_counts[element.label] > 1
        if needed:
            element.display_id = _id_tail(element.resource_id)

    # 编号放在最后一步 —— 所有过滤和排序都完成了。
    for position, element in enumerate(elements, 1):
        element.index = position

    logger.debug(
        f"Screen elements: {len(tree_rows or [])} tree rows "
        f"({dropped} systemui dropped) + {len(ocr_results or [])} ocr rows "
        f"-> {len(elements)} indexed rows"
    )
    return elements


def render_screen_elements(elements: list[ScreenElement]) -> str:
    """把元素表渲染成给模型看的那一块。

    行格式和旧的 UI 块保持一致（`[i] "text" id=... Class (x,y,WxH) flags`），
    只是 OCR 行并进了同一张表、共享同一套编号。OCR 行没有 id 和 flags，
    末尾带 OCR 置信度。
    """
    if not elements:
        return ""
    lines = []
    for element in elements:
        parts: list[str] = []
        if element.label:
            parts.append(json.dumps(element.label, ensure_ascii=False))
        if element.display_id:
            parts.append(f"id={element.display_id}")
        if element.class_name:
            parts.append(element.class_name)
        parts.append(
            f"({element.bounds.x},{element.bounds.y},"
            f"{element.bounds.width}x{element.bounds.height})"
        )
        if element.flags:
            parts.append(element.flags)
        if element.kind == "ocr" and element.confidence is not None:
            parts.append(f"{element.confidence:.2f}")
        lines.append(f"[{element.index}] " + " ".join(parts))
    return f"{_HEADER}\n" + "\n".join(lines)


def resolve_screen_index(
    elements: list[ScreenElement] | None, index: int | None
) -> ScreenElement | None:
    """把 `Target.index` 还原成元素。**1-based**，与渲染出的 `[i]` 一致。

    越界或没有表时返回 None —— 调用方必须把「这个索引不存在」告诉模型，
    不能退化成「点别的地方」。这条和 `ocr_view.resolve_ocr_index` 同源。
    """
    if index is None:
        return None
    if not elements:
        logger.warning("Cannot resolve index: no screen element table on this screen")
        return None
    if index < 1 or index > len(elements):
        logger.warning(
            f"index={index} out of range; the screen element table had {len(elements)} row(s) "
            f"(valid 1..{len(elements)})"
        )
        return None
    return elements[index - 1]
