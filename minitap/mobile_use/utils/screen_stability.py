"""屏幕**稳定性**判定 —— 「这一屏是不是已经定下来了」。

为什么需要它
------------

沉降等待（``PerceptionConfig.settle_delay_ms``）原来是一个**纯时间**的下限：
「距上次动屏不足 N 毫秒就补等到 N 毫秒」。

2026-10-22 真机实测（Redmi Turbo 4 / Android 16）把这个数证伪了：

* ``dump_hierarchy`` **自带 waitForIdle** —— 屏幕还在动的时候它会自己等。
  实测滚动场景里 dump#0 发起于动作后 496ms、**耗时 609ms**，
  那 609ms 就是它在里面等屏幕静下来。
* ��以 3 秒下限**从来没有让树更完整**，它只是让每次抓屏晚 3 秒才发起。
  返回键场景里树一次就稳（141 个元素，5 次 dump 全同），3 秒纯属白等。

但「静」不等于「全」
------------------

**idle 等待只保证「不再动了」，不保证「该挂的都挂上了」。**

页面导航场景实测（点进 WLAN）::

    dump#0  235ms   49 个元素   ← 过渡态
    dump#1   70ms   91 个元素
    dump#2  134ms   91 个元素
    dump#3  360ms  228 个元素   ← 新页面完整内容
    dump#4   87ms  228 个元素

新页面的控件是**懒加载**出来的：dump#1 和 dump#2 的内容已经一样（91 个元素），
按「两次相同 ⇒ 稳定」判会判成稳定 —— 但 228 才是完整的。

⇒ 所以「稳定」这个判据**有已知误判面**：它会漏掉「静默期内才挂上的控件」。
本模块提供的是**最便宜可得的稳定性信号**，它把「还在动」这种最常见的情况
（滚动、动画、过渡态）挡掉，但挡不住懒加载。

**验收必须看「步数」和「失败目标数」，不是「点错率」**
（表和屏是同一次抓的，索引契约自洽；真正的代价是「表里缺控件 ⇒ 模型以为
按钮不存在 ⇒ 绕路多走几步」）。

指纹怎么算
----------

``resource-id`` + ``text`` + ``class`` + ``bounds`` 四元组的**排序集合**：

* **要 bounds** —— 滚动时元素集合可能不变、只有位置在变，那也是「没稳定」。
* **要排序** —— 树的遍历顺序在结构相同时是确定的，但排序能挡住
  「同一屏、顺序微变」这种无意义的差异（少一次无谓的重抓）。
* **不要只数元素个数** —— 实测滚动场景 292→267→240→239 里有两次数相同
  但内容不同（bounds 变了），只数个数会漏判。
"""

from __future__ import annotations

import hashlib

# 单元分隔符。用不可见字符，避免和用户可见文本里的分隔符撞上。
_SEP = "\x1f"
_REC = "\x1e"


def tree_fingerprint(elements: list[dict] | None) -> str:
    """算一棵树的内容指纹。

    同样的内容 ⇒ 同样的指纹。内容变了（元素增减、文字变、位置变）⇒ 指纹变。

    **滚动条必须排除**（2026-10-22 真机实测）
    ----------------------------------------
    MIUI 的滚动条（``com.android.settings:id/main_settings_scrollBar``）
    会**淡入淡出**：即使屏幕完全静止，它也会在两次采样之间出现/消失。
    实测（设置首页静止不动，142 个元素）::

        第一次采样: 142 个（含 scrollBar）
        第二次采样: 141 个（scrollBar 淡出了）
        ⇒ 指纹不同 ⇒ 误判「屏还在动」

    ⇒ 滚动条是**纯视觉装饰**，和「屏定下来没有」无关，必须排除。
    判据用 resource-id 里含 ``scrollbar``（不区分大小写）。

    Args:
        elements: ``_parse_hierarchy_xml_to_elements`` 的输出（或同形状的 dict 列表）。

    Returns:
        十六进制 sha1。空树返回空串的哈希（仍是个稳定值）。
    """
    if not elements:
        return hashlib.sha1(b"").hexdigest()

    parts = [
        _SEP.join(
            (
                str(e.get("resource-id") or ""),
                str(e.get("text") or ""),
                str(e.get("class") or ""),
                str(e.get("bounds") or ""),
            )
        )
        for e in elements
        if not _is_scrollbar(e)
    ]
    parts.sort()
    return hashlib.sha1(_REC.join(parts).encode("utf-8", errors="replace")).hexdigest()


def _is_scrollbar(element: dict) -> bool:
    """这个元素是不是滚动条。

    滚动条会淡入淡出 ⇒ 它出现/消失**不代表**屏幕在动。
    """
    resource_id = str(element.get("resource-id") or "").lower()
    return "scrollbar" in resource_id


# ---------------------------------------------------------------------------
# 结构摘要（video-safe 的稳定性判据）
# ---------------------------------------------------------------------------
#
# 为什么要另立一套判据
# ------------------
# 现行 ``tree_fingerprint`` 哈希 ``resource-id + text + class + bounds`` 的
# **全量**元素。在**视频页**上这个判据是**语义错误**的：视频在播 ⇒ 点赞数、
# 倒计时、进度、推荐流内容每秒都在变 ⇒ 指纹**永远不相等** ⇒
# ``_confirm_tree_stable`` 每次都跑满重试次数（默认 3 次 dump），
# 而最后还是「判定不稳，照用」—— 花了三倍的钱，拿到同一个结果。
#
# 原则在**这个仓库里已经有**（``tools/live_target.py``）：
#
#     OCR 的**内容会变**（点赞数在跳、时钟在走）
#     **位置比内容稳**
#
# ⇒ 稳定性判据应该比「**身份 + 位置**」，不该比「内容」。
#
# 摘要取什么
# ----------
# * **只取可交互元素**（可点 / 长按 / 可滚 / 可聚焦 / 输入框）—— 决策只关心
#   这些；视频画面、装饰、进度条一律不进摘要。
# * **不取 text** —— 计数器和文案在跳，不该让判据一直为假。
# * **取 bounds** —— 元素真的动了（转场/滚动）就是「还没稳」。
# * **排除 scrollbar** —— 同 ``tree_fingerprint``，它会淡入淡出。

#: 算结构摘要时认这些交互属性。uiautomator 的 XML 属性是**字符串**
#: （``"true"`` / ``"false"``），不是 bool。
_INTERACTIVE_ATTRS = ("clickable", "long-clickable", "scrollable", "focusable")

#: 输入框没有 ``clickable``，靠 class 认。
_INTERACTIVE_CLASS_SUFFIX = ("EditText",)

#: **渲染面**永远不是操作目标 —— 视频画面在动，但它是 SurfaceView/TextureView。
#:
#: 有些 app 会给这些面挂 ``clickable="true"``（整块视频可点）⇒ 光看属性会把它
#: 算进摘要 ⇒ 视频每帧一变摘要就变 ⇒ 又回到「永远判不稳」。
#: 所以按 class 再排一次。
_NON_INTERACTIVE_CLASS_SUFFIX = (
    "SurfaceView",
    "TextureView",
    "GLSurfaceView",
    "VideoView",
)


def _truthy(value: object) -> bool:
    """把 uiautomator 的字符串布尔（``"true"``）当成 True。"""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1")


def _resource_id_of(element: dict) -> str:
    """取 resource-id（``resource-id`` / ``resourceId`` 两种键都认）。"""
    for key in ("resource-id", "resourceId"):
        value = element.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _bounds_of(element: dict) -> str:
    """把 bounds 归一成字符串。原始 XML 里是 ``"[0,130][1220,2530]"``，
    另一条来源可能是 ``{"x":..,"y":..,"width":..,"height":..}``。"""
    bounds = element.get("bounds")
    if isinstance(bounds, dict):
        return "%s,%s,%s,%s" % (
            bounds.get("x"),
            bounds.get("y"),
            bounds.get("width"),
            bounds.get("height"),
        )
    return str(bounds or "")


def is_interactive(element: dict) -> bool:
    """这个元素是「模型可能去操作」的吗？"""
    cls = str(element.get("class") or "")
    if any(cls.endswith(suffix) for suffix in _NON_INTERACTIVE_CLASS_SUFFIX):
        # 渲染面：视频在播就是它，但它不是操作目标。
        return False
    if any(_truthy(element.get(attr)) for attr in _INTERACTIVE_ATTRS):
        return True
    return any(cls.endswith(suffix) for suffix in _INTERACTIVE_CLASS_SUFFIX)


def structural_parts(elements: list[dict] | None) -> frozenset[str]:
    """可交互元素的「身份 + 位置」集合（**不含 text**）。"""
    if not elements:
        return frozenset()
    parts = set()
    for element in elements:
        if not is_interactive(element):
            continue
        resource_id = _resource_id_of(element)
        if "scrollbar" in resource_id.lower():
            continue
        parts.add(
            _SEP.join(
                (
                    resource_id,
                    str(element.get("class") or ""),
                    _bounds_of(element),
                )
            )
        )
    return frozenset(parts)


def _box_tuple(element: dict) -> tuple[int, int, int, int] | None:
    """把 bounds 解析成 ``(x, y, w, h)``。原始 XML 里是 ``"[0,130][1220,2530]"``。"""
    raw = _bounds_of(element)
    if not raw:
        return None
    nums: list[int] = []
    for part in raw.replace("][", ",").strip("[]").split(","):
        try:
            nums.append(int(part))
        except ValueError:
            return None
    return tuple(nums) if len(nums) == 4 else None  # type: ignore[return-value]


def structural_items(elements: list[dict] | None) -> list[tuple[str, str, tuple[int, int, int, int]]]:
    """可交互元素的「身份 + 位置」列表（**不含 text**），位置是数字四元组。"""
    items: list[tuple[str, str, tuple[int, int, int, int]]] = []
    if not elements:
        return items
    for element in elements:
        if not is_interactive(element):
            continue
        resource_id = _resource_id_of(element)
        if "scrollbar" in resource_id.lower():
            continue
        box = _box_tuple(element)
        if box is None:
            continue
        items.append((resource_id, str(element.get("class") or ""), box))
    return items


def structural_digest(elements: list[dict] | None) -> str:
    """结构摘要（video-safe）。

    同样的**可交互结构** ⇒ 同样的摘要。视频在播、计数在跳 ⇒ 摘要不变。
    元素增删/位移 ⇒ 摘要变。

    ⚠️ 这是**精确**比对 —— 视频页上元素每采样一次会整体漂 **1 像素**
    （真机实测：``[1008,1677][1220,1914]`` → ``[1007,1676][1220,1914]``），
    所以**别用它判稳**，用 :func:`structural_change_count`（带容差）。
    """
    parts = structural_parts(elements)
    if not parts:
        return hashlib.sha1(b"").hexdigest()
    return hashlib.sha1(_REC.join(sorted(parts)).encode("utf-8", errors="replace")).hexdigest()


#: 位置比较的容差（像素）。
#:
#: **为什么需要它**：快手视频页上，可交互元素每采样一次整体漂 1 像素
#: （真机实测 6 次采样，每对都有 1px 位移）。精确比对会把这种漂移当成「还在动」。
#:
#: **为什么不用网格量化**：值落在网格边界附近时，1 像素漂移会跨到相邻格子
#: ⇒ 容差越大反而「差异」越多（实测：量化到 32px 时差异比 16px 还多）。
#: 直接比「差值是否超过容差」没有这个问题。
#:
#: **为什么 8px 够**：实测漂移是 1~2px；真实变化是 100px+（caption 展开 102px、
#: 滚动 300px、浮动元素进场）。8px 落在两者之间。
POSITION_TOLERANCE_PX = 8


def structural_change_count(
    previous: list[dict] | None,
    current: list[dict] | None,
    tolerance_px: int = POSITION_TOLERANCE_PX,
) -> int:
    """两次采样之间，**可交互结构**差了多少（带位置容差的配对）。

    配对条件：``resource-id`` + ``class`` 相同，且 bounds 每一维的差都 ≤ 容差。
    返回**配不上对**的元素总数（双向）—— 0 表示结构稳定。

    真机实测（快手视频页，6 次采样，间隔 500ms）::

        容差  0px  判稳 1/5   差异 [1, 39, 0, 7, 20]
        容差  2px  判稳 2/5   差异 [1,  1, 0, 7,  0]
        容差  8px  判稳 2/5   差异 [1,  1, 0, 7,  0]

    ⇒ 1 像素漂移被容差吃掉；真实变化（浮动元素进场、caption 展开）照样判出来。
    ⇒ 人为滚动 300px 的样本，容差 8 仍判出 70 处差异 —— **不漏判滚动**。
    """
    a_items = structural_items(previous)
    b_items = structural_items(current)

    unmatched = list(b_items)
    diff = 0
    for rid, cls, box in a_items:
        hit = None
        for other in unmatched:
            if other[0] != rid or other[1] != cls:
                continue
            if all(abs(x - y) <= tolerance_px for x, y in zip(box, other[2])):
                hit = other
                break
        if hit is None:
            diff += 1
        else:
            unmatched.remove(hit)
    return diff + len(unmatched)


def is_stable(previous: str | None, current: str | None) -> bool:
    """两次采样是否算「同一屏」。

    没有上一次采样（第一次）⇒ **不算稳定** —— 我们还不知道它稳没稳，
    保守地让调用方再采一次。
    """
    if previous is None or current is None:
        return False
    return previous == current
