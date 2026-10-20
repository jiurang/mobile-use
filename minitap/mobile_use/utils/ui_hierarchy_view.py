"""把 Android UI 层级树裁剪成紧凑格式再喂给 LLM。

为什么需要这个
--------------
原始层级树是一份扁平列表，每个无障碍节点一个约 20 个字段的 dict。
整棵树序列化后每轮要烧 14k~24k tokens，而其中绝大部分是 agent 根本没法操作的
布局填充：实测一个 126 节点的屏，只有 5 个节点可点击，44 个带 text/desc/resource-id。

这里只渲染 agent 可能定位或读取的节点，按阅读顺序排列，并丢掉工具契约
从来不会消费的字段。

适用范围
--------
仅 Android。识别的元素形状由
`clients.ui_automator_client._parse_hierarchy_xml_to_elements` 产出
（`text` / `content-desc` / `resource-id` / `class` / `bounds` / `clickable`…，
其中 bounds 是字符串 `"[x1,y1][x2,y2]"`）。

iOS 与云设备客户端产出的是另一种形状（`label` / `value` / `type` / `frame`），
本渲染器不识别。这里不去猜：`render_ui_hierarchy` 检测到形状不认识时会退回
原来的 JSON dump，在补齐支持前保持这些平台现有行为不变。

必须守住的不变量（真机实测得出，不是推断）
------------------------------------------
1. **只裁展示层**。图状态里的完整层级树不能动：工具要拿它按
   `resource_id` / `text` / `bounds` 查元素。绝不要裁传给工具的那份副本。
2. **`resource-id` 输出全名**。`AndroidController.find_element` 用的是
   `element.get("resource-id") == resource_id` **精确等值**匹配，
   缩短成 `id=submit` 会让所有基于 id 的定位全部失配。实测输出全名相比
   只给尾段每轮多约 200 tokens —— 不值得为此改坏匹配逻辑。
3. **可操作节点一个都不能丢**。`clickable` / `long-clickable` /
   `focusable` 的节点不会被结构性过滤器丢弃；在桌面 / 设置 / 多任务 /
   通知栏四屏上实测召回率为 100%。
4. **绝不给模型一棵空树**。形状不认识才退回 JSON dump；形状认识但裁剪后
   为空时输出 `(none)` —— 那种情况下一旦退回就等于把裁剪悄悄撤销了。

实测效果（Redmi 真机，4 屏，每屏 77~135 节点）：
14,042~24,334 tokens → 1,173~2,047 tokens，即降低 91%~94%。

还剩的最大一块 fat：resource-id（2026-10-17 实测）
----------------------------------------------
抖音搜索页（1220x2712，290 原始节点 → 渲染 116 行 / 3,372 tok）逐字段拆解：
    id=  1,531 tok (45.4%)   出现在 113/116 行
    box  1,293 tok (38.3%)   出现在 116/116 行
    索引  464 tok (13.8%)
    label 318 tok ( 9.4%)   只在 42/116 行
    flags  84 tok ( 2.5%)

`id=` 一项就占了整棵树 45%、Cortex 整个 prefill 的 18.7%。而它里面
**约 77% 是每行重复的包名前缀**（`com.ss.android.ugc.aweme.lite:` ≈ 10 token × 113 行）。

**为什么不能直接把前缀提到 header、每行只留 `id=search_input`**：
模型会把看到的字符串原样抄进 `tap` 的 `resource_id`，而
`AndroidController.find_element` 是 `element.get("resource-id") == resource_id`
精确等值匹配 —— 抄短了所有基于 id 的定位全部失配。这是下面
不变量 #2 要守住的原因，实测也确实守住了。

**同样不能靠「丢掉只有 id 的节点」来省**（试过，行不通）：
抖音那一屏有 57/116 行是「只有 id 和坐标、没有文字、也没有 C/L/F」的节点，
按「可操作 or 有文字」过滤能省 43%，但 `tests/test_ui_hierarchy_view.py`
里 `test_keeps_full_resource_id_verbatim` 和 `test_every_actionable_node_survives`
明确要求这种节点必须保留 —— 因为 `input tap` 打任何坐标都有效，
id 名字（`id/send`、`id/search_input`）本身就是「这是个可点的东西」的线索。
要让这条路走得通，必须**同时**改 `find_element` 让它容忍尾段匹配，
属于工具契约变更，不能顺手做。

要动这一块，唯一正确的顺序是：先改 `find_element` 支持尾段匹配 + 补测试，
再改这里的渲染。顺序反过来就是静默地弄坏所有 id 定位。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

# 纯布局容器：自身从不携带信息，而且每行都重复一遍类名纯属噪音。
_LAYOUT_CLASSES = frozenset(
    {
        "FrameLayout",
        "LinearLayout",
        "RelativeLayout",
        "ConstraintLayout",
        "ScrollView",
        "ViewGroup",
        "View",
        "SurfaceView",
        "TextureView",
        "CoordinatorLayout",
        "GridLayout",
        "TableLayout",
    }
)

# 纯展示型叶子节点：标签已经显示出来了，再带上类名没有增量信息。
_DISPLAY_CLASSES = frozenset({"TextView", "ImageView", "ViewAnimator"})

# 即使 dump 没有标 clickable，这些类本身就意味着"可以操作"。
_INTERACTIVE_CLASSES = frozenset(
    {
        "EditText",
        "CheckBox",
        "RadioButton",
        "Switch",
        "SeekBar",
        "Spinner",
        "ToggleButton",
        "ImageButton",
        "AutoCompleteTextView",
        "Button",
    }
)

_BOUNDS_RE = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")

_HEADER = (
    'UI elements, one per line: [i] "label" | id=<resource-id> | Class | (x,y,WxH) | flags\n'
    "flags: C=clickable L=long-press F=focusable K=checkbox S=selected R=scrollable"
)


def _truthy(value: Any) -> bool:
    """uiautomator 把布尔值 dump 成字符串 "true" / "false"。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _text_of(element: dict) -> str:
    for key in ("text", "content-desc", "accessibilityText"):
        value = element.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _id_of(element: dict) -> str:
    value = element.get("resource-id")
    return value.strip() if isinstance(value, str) else ""


def _class_of(element: dict) -> str:
    value = element.get("class")
    return value.rsplit(".", 1)[-1] if isinstance(value, str) else ""


def _box_of(element: dict) -> tuple[float, float, float, float] | None:
    """把 uiautomator 的 "[x1,y1][x2,y2]" 解析成 (x1, y1, x2, y2)。"""
    raw = element.get("bounds")
    if not isinstance(raw, str):
        return None
    match = _BOUNDS_RE.match(raw.strip())
    return tuple(map(float, match.groups())) if match else None  # type: ignore[return-value]


def _normalize_label(text: str) -> str:
    """label 归一化：去空白 + 小写。

    OCR 引擎和 uiautomator 对同一段可见文字的切分、空格、大小写经常不一致
    （"Search" vs " search "）。要判断「OCR 这条是不是树里已经有了」，
    必须先归一化，否则同一句话会两边都出现，白白多花 token。
    """
    return re.sub(r"\s+", "", text).strip().lower()


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class _Node:
    label: str
    rid: str
    cls: str
    box: tuple[float, float, float, float] | None
    flags: str
    interactive: bool

    @property
    def _priority(self) -> tuple:
        """去重时的存活优先级：先看信息量，再看框越小越好。"""
        area = (
            (self.box[2] - self.box[0]) * (self.box[3] - self.box[1]) if self.box else float("inf")
        )
        return (self.box is None, not self.label, not self.rid, not self.interactive, area)


def _normalize(element: dict) -> _Node | None:
    if not isinstance(element, dict):
        return None
    # 屏幕外的节点既观察不到，也点不到。
    if "visible-to-user" in element and not _truthy(element.get("visible-to-user")):
        return None

    label = _text_of(element)
    rid = _id_of(element)
    cls = _class_of(element)
    box = _box_of(element)
    enabled = _truthy(element.get("enabled", True))

    clickable = _truthy(element.get("clickable"))
    long_clickable = _truthy(element.get("long-clickable"))
    focusable = _truthy(element.get("focusable"))
    interactive = enabled and (
        clickable or long_clickable or focusable or cls in _INTERACTIVE_CLASSES
    )

    shown_class = "" if cls in (_LAYOUT_CLASSES | _DISPLAY_CLASSES) else cls
    if not (interactive or label or rid or shown_class):
        return None

    # 标记描述的是 agent 实际能做什么，所以也必须受 enabled 约束：
    # 给禁用按钮标上 C，等于诱导模型去点一个注定失败的元素。
    flags = "".join(
        [
            "C" if clickable and enabled else "",
            "L" if long_clickable and enabled else "",
            "F" if focusable and enabled else "",
            "K" if enabled and _truthy(element.get("checkable")) else "",
            "S" if _truthy(element.get("selected")) else "",
            "R" if _truthy(element.get("scrollable")) else "",
        ]
    )

    return _Node(
        label=label, rid=rid, cls=shown_class, box=box, flags=flags, interactive=interactive
    )


def _prune(nodes: list[_Node]) -> list[_Node]:
    # 1. 合并几乎重合的框，同组里保留信息量最大的那个。
    kept: list[_Node] = []
    for node in sorted(nodes, key=lambda n: n._priority):
        if node.box and any(k.box and _iou(node.box, k.box) >= 0.95 for k in kept):
            continue
        kept.append(node)

    # 2. 丢掉只做包裹、没有文本的外层容器 —— 这是不可操作行号的最大来源。
    pruned: list[_Node] = []
    for node in kept:
        if not node.label and node.box:
            enclosed = sum(
                1
                for other in kept
                if other is not node
                and other.box
                and _iou(node.box, other.box) > 0.9
                and _iou(other.box, node.box) < 0.5
            )
            if enclosed > 2:
                continue
        pruned.append(node)

    # 3. 按阅读顺序排列，让行号和扫屏顺序一致。
    pruned.sort(key=lambda n: (n.box[1] if n.box else 0, n.box[0] if n.box else 0))
    return pruned


def _render(rows: list[dict]) -> str:
    lines = []
    for index, row in enumerate(rows, 1):
        parts: list[str] = []
        label = row.get("label") or ""
        if label:
            parts.append(json.dumps(label, ensure_ascii=False))
        rid = row.get("resource_id") or ""
        if rid:
            parts.append(f"id={rid}")
        cls = row.get("class_name") or ""
        if cls:
            parts.append(cls)
        box = row.get("bounds")
        if box:
            parts.append(f"({box['x']},{box['y']},{box['width']}x{box['height']})")
        flags = row.get("flags") or ""
        if flags:
            parts.append(flags)
        lines.append(f"[{index}] " + " ".join(parts))
    return "\n".join(lines)


def _looks_android(elements: list[dict]) -> bool:
    """判断输入是否带有 uiautomator 的属性特征。

    用来区分两件事："形状认识了，只是没东西值得显示" 和 "形状完全不认识" ——
    只有后者才允许退回原始 dump，否则这次兜底会把裁剪悄悄撤销掉。
    """
    signature = {
        "bounds",
        "resource-id",
        "class",
        "content-desc",
        "accessibilityText",
        "clickable",
        "long-clickable",
        "focusable",
        "visible-to-user",
    }
    return any(signature & element.keys() for element in elements if isinstance(element, dict))


def _node_to_dict(node: _Node) -> dict:
    """`_Node` → 可序列化的 dict（bounds 用 x,y,width,height）。

    坐标取整方式和 `_render` 原来的写法一致（`int()` 截断），所以
    `normalize_hierarchy()` 渲染出来的块和改造前**逐字节相同**。
    """
    box = None
    if node.box:
        x1, y1, x2, y2 = node.box
        box = {
            "x": int(x1),
            "y": int(y1),
            "width": int(x2 - x1),
            "height": int(y2 - y1),
        }
    return {
        "label": node.label,
        "resource_id": node.rid,
        "class_name": node.cls,
        "bounds": box,
        "flags": node.flags,
    }


def normalize_hierarchy(elements: list[dict] | None) -> list[dict] | None:
    """归一化 + 裁剪层级树，返回可复用的节点列表。

    这是 `render_hierarchy_and_labels` 的内核，单独暴露出来是为了让
    `utils/screen_elements.py` 能**基于同一份裁剪结果**建统一索引表 ——
    渲染给模型的那张表和执行时反解用的那张表必须出自同一次裁剪，
    否则行号会对不上。

    返回 `None` 表示「形状不认识」（iOS / 云设备），调用方要退回原来的
    原始 JSON dump —— 这一步不能猜，猜错就是静默剥夺别的平台的感知能力。

    每个节点形如::

        {"label": str, "resource_id": str, "class_name": str,
         "bounds": {"x","y","width","height"} | None, "flags": str}
    """
    if not elements:
        return []
    if not _looks_android(elements):
        return None
    nodes = [node for node in (_normalize(e) for e in elements) if node is not None]
    return [_node_to_dict(node) for node in _prune(nodes)]


def render_hierarchy_and_labels(elements: list[dict] | None) -> tuple[str, set[str]]:
    """渲染层级树，**同时**返回真正进入 prompt 的那些 label。

    为什么要把 label 一起返回：`utils/ocr_view.py` 会用它做去重，把
    「层级树里已经有的文字」从 OCR 块里剔掉。两个裁剪必须看到**同一棵**
    渲染结果，否则会出事——

        某条文字在树里有，但所在节点被判成不可操作而过滤掉了，
        于是它既不在树的渲染里、又被当成「树里已有」从 OCR 里剔掉，
        模型就彻底看不到这条文字了：树里没有，OCR 里也没有。

    所以去重的依据只能是「实际渲染出去的 label」，不是原始 dump 的全集。

    返回值第二项是归一化（小写去空白）后的 label 集合，直接给 OCR 去重用。
    """
    if not elements:
        return "", set()

    rows = normalize_hierarchy(elements)
    if rows is None:
        logger.warning(
            "UI hierarchy has no Android accessibility attributes; "
            "falling back to the raw JSON dump"
        )
        return "Here is the UI hierarchy:\n" + json.dumps(
            elements, indent=2, ensure_ascii=False
        ), set()

    labels = {_normalize_label(row["label"]) for row in rows if row["label"]}
    labels.discard("")

    logger.debug(
        f"UI hierarchy: {len(elements)} raw nodes -> {len(rows)} rendered "
        f"(of which {len(labels)} distinct labels)"
    )
    text = f"{_HEADER}\n{_render(rows)}" if rows else _HEADER + "\n(none)"
    return text, labels


def render_ui_hierarchy(elements: list[dict] | None) -> str:
    """把 Android 的 `elements` 渲染成 Cortex 读取的紧凑裁剪块。

    只有在元素形状不是本渲染器认识的 Android 形状时，才退回原来的 JSON dump，
    这样那些平台保持现有行为，而不是被悄悄剥夺感知能力。
    """
    return render_hierarchy_and_labels(elements)[0]
