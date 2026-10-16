"""校验模型给出的坐标是否真的来自 UI 层级树，而不是从截图目测的。

问题
----
`tools/mobile/tap.py` 用 bounds 坐标点击，且把它排在 fallback 链第一位
（"Try with COORDINATES FIRST"）。原来的 `validate_coordinates_bounds`
只检查坐标是否落在屏幕矩形内 —— 这个条件太弱：

    模型从缩放后的截图上目测坐标（缩小后必然偏小）
      → 坐标依然在屏幕矩形内 → 校验通过
      → input tap 成功返回 → success=True
      → 报告 "Tap ... was successful."，实际点到了别处

此时 resource_id / text 两级兜底**根本不会执行**，错误无从纠正。
`cortex.md:83` 设计的纠错信号是 `"Out of bounds"`，这种情况永远触发不到。

判据：归一化 IoU
----------------
真机实测（scripts/probe_validation_criteria2.py，Redmi Turbo 4，1220x2712，
4 屏共 68 个可操作节点）：

    原始 IoU        good: min=1.000 max=1.000    bad: p50=0.129 max=0.483
    归一化 IoU      good: min=1.000 max=1.000    bad: p50=0.322 max=0.816

    阈值 0.90  ->  好坐标接受 100.0%    坏坐标拦截 100.0%

「归一化」指先把模型给的框按最佳匹配节点做尺度对齐（以中心为锚点），
再比 IoU。这一步消除了框尺寸漂移的影响，让 IoU 只反映位置差异。

为什么必须归一化
----------------
不归一化时，框外扩 10% 就让 IoU 掉到 0.826（scripts/probe_iou_tolerance.py），
而坏坐标的 IoU 上界恰好也是 0.826 —— 门限定在哪一侧都会误伤另一类。
归一化后好坐标的 IoU 恒等于 1.000（对齐后必然完全重合），
坏坐标上界降到 0.816，两者之间留出干净的间隔。

为什么不用更简单的判据（都有真机数据支撑，不是拍脑袋）
------------------------------------------------------
1. 「点是否落在某个节点内」            拦截 0%：dump 里有 31 个 >=95% 屏幕的
                                       容器节点，任何点都落在某个节点内。
2. 「中心点是否落在最小包含节点内」   同样 0%，容器问题。
3. 「中心点命中 IoU 最佳匹配节点」     同样 0%：半尺寸的框会匹配到大容器，
                                       中心点自然落在里面（68/68 全漏）。
4. 「面积够小才算真控件」              可操作节点与容器面积严重重叠：
                                       真控件最大 75.92%（整个设置列表），
                                       最小的大容器只有 5.35%
                                       （scripts/probe_container_area.py），
                                       任何单一门限都会误伤其中一类。
"""

from __future__ import annotations

from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.controllers.unified_controller import UnifiedMobileController
from minitap.mobile_use.tools.types import Target
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds, get_bounds_for_element

logger = get_logger(__name__)

# 真机实测的干净分离点：好坐标归一化 IoU 恒为 1.000，坏坐标上界 0.816。
MIN_IOU = 0.9


def _iou(a: ElementBounds, b: ElementBounds) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.width, a.y + a.height
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.width, b.y + b.height
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    if inter <= 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def _recenter_to(bounds: ElementBounds, width: int, height: int) -> ElementBounds:
    """以 bounds 的中心为锚点，把尺寸改成 width x height。"""
    cx = bounds.x + bounds.width / 2
    cy = bounds.y + bounds.height / 2
    return ElementBounds(
        x=int(round(cx - width / 2)),
        y=int(round(cy - height / 2)),
        width=width,
        height=height,
    )


def normalized_iou_for_bounds(
    bounds: ElementBounds, elements: list[dict]
) -> tuple[float, ElementBounds | None, dict | None]:
    """返回 (归一化 IoU, 最佳匹配节点的 bounds, 该节点原始 dict)。

    先按 IoU 找出「模型可能想指向谁」，再把 bounds 缩放到该节点的尺寸（中心不变）
    重新求 IoU。后者只反映位置差异，不受框大小影响。
    """
    best_raw = 0.0
    best_bounds: ElementBounds | None = None
    best_element: dict | None = None
    for element in elements:
        other = get_bounds_for_element(element)
        if other is None:
            continue
        value = _iou(bounds, other)
        if value > best_raw:
            best_raw = value
            best_bounds = other
            best_element = element

    if best_bounds is None:
        return 0.0, None, None

    aligned = _recenter_to(bounds, best_bounds.width, best_bounds.height)
    return _iou(aligned, best_bounds), best_bounds, best_element


def best_iou_for_bounds(bounds: ElementBounds, elements: list[dict]) -> tuple[float, dict | None]:
    """返回未归一化的最佳 IoU 及该节点。保留给诊断用途。"""
    best, element = 0.0, None
    for e in elements:
        other = get_bounds_for_element(e)
        if other is None:
            continue
        value = _iou(bounds, other)
        if value > best:
            best, element = value, e
    return best, element


def _label_of(element: dict | None) -> str:
    if element is None:
        return ""
    return (
        (element.get("text") or "").strip()
        or (element.get("content-desc") or "").strip()
        or element.get("resource-id")
        or (element.get("class") or "").rsplit(".", 1)[-1]
        or ""
    )


def coordinates_look_like_hierarchy_derived_from_elements(
    target: Target,
    elements: list[dict],
    min_iou: float = MIN_IOU,
) -> tuple[bool, str | None]:
    """判断 target.bounds 是否像是从层级树抄来的（而不是从截图目测的）。

    Returns:
        (是否可信, 失败原因)。可信时原因为 None。
    """
    if not target.bounds:
        return True, None  # 没有坐标可校验，交由原有逻辑处理

    if not elements:
        # 拿不到层级树时不能拒绝点击 —— 否则 dump 失败会变成全面不可用。
        return True, None

    iou, matched_bounds, matched_element = normalized_iou_for_bounds(target.bounds, elements)

    if matched_bounds is None:
        return False, (
            f"coordinates ({target.bounds.get_center().x}, {target.bounds.get_center().y}) "
            "do not overlap any element in the UI hierarchy. The bounds were likely "
            "estimated from the screenshot; use resource_id or text instead."
        )

    if iou >= min_iou:
        return True, None

    center = target.bounds.get_center()
    label = _label_of(matched_element)
    return False, (
        f"coordinates ({center.x}, {center.y}) do not line up with any UI element "
        f"(closest: {label[:60] if label else 'unnamed'} at "
        f"({matched_bounds.x},{matched_bounds.y},{matched_bounds.width}x{matched_bounds.height}), "
        f"normalized IoU {iou:.2f} < {min_iou:.2f}). The bounds were likely estimated from "
        "the screenshot instead of copied from the element list; use resource_id or text instead."
    )


async def coordinates_look_like_hierarchy_derived(
    ctx: MobileUseContext,
    target: Target,
    min_iou: float = MIN_IOU,
) -> tuple[bool, str | None]:
    """异步包装：实时 dump 层级树后做校验。

    为什么必须实时 dump
    ------------------
    不能用 `state.latest_ui_hierarchy`：`cortex.py:144` 在 cortex 返回时就把该字段
    置为 None，executor 工具执行时 state 里已经没有层级树了。
    `UnifiedMobileController.get_ui_elements()` 是现成的实时取数路径
    （`tap_element` 本来就用它）。

    失败策略
    --------
    dump 失败或返回空时**放行**，不阻断点击 —— 否则一次 dump 失败会让整个 agent 不可用。
    宁可放过一次坏坐标，也不要因为校验器本身出错而全面瘫痪。
    """
    if not target.bounds:
        return True, None

    try:
        elements = await UnifiedMobileController(ctx).get_ui_elements()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not dump UI hierarchy for bounds validation: {e}")
        return True, None

    if not elements:
        logger.warning("Empty UI hierarchy; skipping bounds validation")
        return True, None

    return coordinates_look_like_hierarchy_derived_from_elements(
        target, elements, min_iou=min_iou
    )
