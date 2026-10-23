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
8. **纯装饰行不裁**：「不可交互 + 无文字」的行**留着**。看着像纯浪费，
   但真机逐个点过之后发现它们里有一截是真能点的，裁不得。
   实测（Redmi Turbo 4，12 个 App 页面，126 行「装饰候选」逐个 adb 点）：
   **17 行（13.5%）点了之后屏幕真的变了**。命中的都是模型会想点的东西：
   小米天气的 7 天预报图标（`daily_forecast_item_icon`，点了进详情页，
   屏幕变化 98.5%）、城市切换块（`main_city_indicator_container`）、
   淘宝的搜索框（`search_view`）、小红书的卡片。
   原因很直白：这些 App 用 Canvas / 自定义 View 画可点区域，
   uiautomator 的 dump 里**既没有 clickable、也没有文字**，
   「没有交互标记」根本推不出「点它没反应」。
   而它们**全部有 resource_id**（82/82），id 尾段正是模型认出
   「这是个可点的东西」的唯一线索（`id/send`、`id/search_input`）。
   三个「退一步」的收窄判据实测下来**各裁 0 行**（82 行候选里
   最小面积 845px²，没有一个「小到不可能是点击目标」；
   82 行全都带 id；层级树拍平后没有一行是别人的几何父容器），
   也就是说不存在「只裁一部分」的落点。
   ⇒ 结论：**别裁**。要省这块 token，得先让 `find_element` 支持
   尾段匹配（工具契约变更），让模型拿 id 也能定位到东西；
   顺序反过来就是静默地弄坏所有 id 定位。
   复现脚本：`scripts/probe_decorative_clickable.py`（逐行实点）、
   `scripts/probe_decorative_crop_impact.py`（token 账）、
   `scripts/probe_decorative_analyze.py`（逐判据复算）。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Literal

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


def foreground_package_from_tree(elements: list[dict] | None) -> str | None:
    """从层级树里推当前前台应用的包名。

    为什么不再单独跑一次 adb
    ----------------------
    原来这一项是 ``dumpsys window | grep mCurrentFocus`` —— 一次**串行**的
    adb 往返（真机实测中位 127ms），而且排在抓屏**之后**，全额计入关键路径。

    但**答案就在刚抓回来的树里**：树里每个元素都带 ``package``。

    判据：**元素最多的那个包**（排除系统 UI）。
    ⚠️ **不能用「第一个带 package 的节点」** —— 真机实测那是错的：
    MIUI 的树第一个节点是**桌面**（``com.miui.home``），哪怕前台明明是
    设置（``dumpsys`` 说是 ``com.android.settings``）。真机逐屏对账
    （设置首页 / WLAN 页 / 通知栏 / 桌面，4/4 一致）见
    ``scripts/verify_perception_changes.py``。

    排除系统 UI 的理由：状态栏那 40~50 个 systemui 节点不属于任何应用，
    而且它们是所有应用**共有**的 —— 留着它们，前台就永远推不出来。
    全树都是系统 UI 时（通知栏展开）退回总众数。

    精度要求
    --------
    这个包名只有一个用途：``_should_drop_systemui`` 判断要不要保留系统
    状态栏的行。所以「是不是 systemui」这一层判对就够了。
    """
    if not elements:
        return None

    counts: dict[str, int] = {}
    for element in elements:
        package = element.get("package")
        if package:
            counts[package] = counts.get(package, 0) + 1
    if not counts:
        return None

    systemui = {"com.android.systemui", "miui.systemui.plugin"}
    app_counts = {k: v for k, v in counts.items() if k not in systemui}
    if app_counts:
        return max(app_counts, key=lambda k: app_counts[k])
    # 全是系统 UI（通知栏 / 状态栏设置页）⇒ 用总众数
    return max(counts, key=lambda k: counts[k])


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
    tree_rows: list[dict[str, Any]] | None,
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


def render_screen_rows(elements: list[ScreenElement]) -> str:
    """只渲染数据行（不带表头）。空表返回空串。

    **索引契约的唯一来源**：这里印出的 `[i]` 和 `resolve_screen_index` 吃的
    下标是同一条契约的两端。表头属于提示词，住在 Executor 的模板里
    （`executor.md` / `executor-zh.md` 的 `screen_elements_message`），
    所以表头可以按语言翻译，编号规则不行。
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
    return "\n".join(lines)


def render_screen_elements(elements: list[ScreenElement]) -> str:
    """表头 + 数据行（英文默认版）。

    Executor 的提示词走 `render_screen_rows` + 模板表头（支持中英切换）；
    这个函数保留给测试 / 脚本 / 其它需要独立英文块的调用方。
    """
    rows = render_screen_rows(elements)
    if not rows:
        return ""
    return f"{_HEADER}\n{rows}"


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


# ---------------------------------------------------------------------------
# 「这一屏上看得见什么」的两个原语（2026-10-24）
# ---------------------------------------------------------------------------
#
# 为什么需要：执行器的**局部历史**里只有「调用」和「工具回执」
# （「Key Home pressed successfully.」），**没有「结果」** —— 屏幕长什么样、
# 前台 app 是谁，一概没记。执行器于是每轮只能拿当屏去和自己脑子里的几十条
# 记忆比，靠「我刚才按了 Home，画面变了吗」这种回忆来判断自己有没有在原地
# 打转。实测（execution 853）：它把 Home 按了 13 次，每一次都以为 Home 会
# 回到桌面。
#
# ⇒ 把「结果」搬进历史。这里的两个原语就是搬它要用的那两样东西。


def visible_texts(elements: list[ScreenElement] | None) -> list[str]:
    """这一屏**看得见且不重复**的文字，按屏序（从上到下）。

    ⚠️ 去重按**原文**：「打开」和「打开 」是两条不同的视觉事实，合并掉就是丢事实。
    """
    out: list[str] = []
    for element in elements or []:
        text = (element.label or element.text or "").strip()
        if text and text not in out:
            out.append(text)
    return out


#: **纯数字/标点的短文本** —— 状态栏的时钟（``2:24``）、电量（``96 4``）这类
#: **自己就会变**的东西。不归一的话，「内容变了」永远成立，任何「没进展」的
#: 判据都不会触发。
#:
#: ⚠️ **只认纯数字标点，绝不碰带文字的**（2026-10-24 实测踩过）：
#: 一开始把**所有**数字都归一成 ``#``，结果「视频标题1 → 视频标题2」也被归成
#: 同一个 ⇒ **刷短视频（正常浏览）被误判成「屏幕内容没变」**。
#: ⇒ 归一的范围必须窄到只覆盖真正的噪音。
_NOISE_ONLY = re.compile(r"^[\d\s:：%/.\-+]{1,8}$")


def is_noise_text(text: str) -> bool:
    """这段文字是不是「自己会变」的噪音（时钟/电量/百分比）。"""
    return bool(_NOISE_ONLY.match(text.strip()))


def screen_fingerprint(elements: list[ScreenElement] | None) -> frozenset[str]:
    """这一屏的**内容指纹** —— 用来判断「内容和上一屏比变了没」。

    只有**纯数字/标点的短文本**被归一（见 :func:`is_noise_text`）；
    带文字的一律按**原文**算（「视频标题1」和「视频标题2」是不同的内容）。
    """
    return frozenset(
        "#noise#" if is_noise_text(text) else text for text in visible_texts(elements)
    )
