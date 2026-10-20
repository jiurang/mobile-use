"""按「身份」在当屏重新定位目标 —— 取代「坐标 + IoU 门限」作为点击的主判据。

为什么要有这一层
----------------
IoU 门限当初要防的是「模型从**缩放后的截图上目测**坐标」：

    目测坐标 → 坐标还在屏幕矩形内 → 校验通过 → tap 成功返回
      → 报告 "Tap was successful"，实际点到了别的地方

那个前提今天已经不成立了：

* 截图通道在所有实际运行里都是 **DISABLED**（基线 exec 319/324/330
  以及新跑的 335~338 全部关着），模型没有图可目测；
* 模型现在只报**索引**，坐标由代码从元素表取，一个字都不用抄。
  实测旧代码手抄的坐标有 **16.8% 直接越界**，10 个 resource-id 配过
  互相矛盾的框 —— 这些错误来源在新代码里一个都没有。

而 IoU 门限自己还留着两个问题：

1. **门限是个任意的数**。0.90 是在「好坐标=原样抄 / 坏坐标=0.5 倍目测」
   这个**二值模型**上测出来的（scripts/probe_validation_criteria2.py），
   而「坏坐标」这一类现在几乎不存在。真机上出现过的误伤是
   **IoU 0.89 —— 同一个元素、偏了 14 像素**。
2. **被拒要 dump 两次**（门限判一次，resource_id 兜底再找一次）。

按身份重定位把这三点一起解决
----------------------------
树行（实测占模型点击的 **96%**）在代码侧留着**完整 resource_id**
（提示里只给尾段，完整值在 `ScreenElement.resource_id`）。
于是在当屏 dump 里按 id 找：

* 找到 → 拿到它的**当前**位置 → 直接点。**没有门限，没有误伤，dump 只做一次。**
  屏变了（元素移动、动画）也无所谓 —— 找的是同一个元素的新位置。
* 找不到 → 明确失败，让模型重读屏幕。

同 id 有多个节点时（实测 `id=textView` 一屏能出现 3 次），
用「离**模型当时看到的位置**最近的那个」消歧。注意这是**提示**不是**门限**：
挑错了也只是挑错一个同 id 的兄弟节点，不会点到无关的东西。

OCR 行（没有 resource-id，实测只占 4%）
--------------------------------------
OCR 文本**不在层级树里**（它就是"树里没有的可见文字"），所以没有 id 可查。
这一路径只能靠几何判据 —— 也就是现有的 IoU 位置校验。留着它。

但不能改成"按文本重新匹配"：OCR 的**内容会变**（点赞数在跳、时钟在走），
实测 `未点赞，点赞1445` → `未点赞，点赞2161` 几分钟内就变了。
**位置比内容稳**，所以这一条必须用位置判据。
"""

from __future__ import annotations

from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.controllers.unified_controller import UnifiedMobileController
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds, get_bounds_for_element

logger = get_logger(__name__)


def _resource_id_of(element: dict) -> str:
    """取元素的 resource-id。

    两种键都要认：uiautomator 的 dump 是 `resource-id`，
    而另一条来源（`find_element_by_resource_id` 的非 rich 路径）是 `resourceId`。
    """
    for key in ("resource-id", "resourceId"):
        value = element.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


async def resolve_live_bounds_by_id(
    ctx: MobileUseContext,
    resource_id: str,
    index: int = 0,
    hint: ElementBounds | None = None,
) -> tuple[ElementBounds | None, str]:
    """按 resource-id 在**当屏**上重新定位这个元素。

    Args:
        ctx: 设备上下文。
        resource_id: 完整 resource-id（代码侧留的，不依赖模型抄）。
        index: 同 id 元素中的序号。**只在候选唯一时用**；不唯一时按位置消歧。
        hint: 模型当时看到的那一行的位置（用来在同 id 的兄弟节点里挑对的那个）。

    Returns:
        (当屏上的 bounds, 失败原因)。bounds=None 表示当屏上找不到它。
    """
    try:
        elements = await UnifiedMobileController(ctx).get_ui_elements()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Live dump failed while resolving {resource_id!r}: {exc}")
        return None, f"live dump failed: {exc}"

    candidates: list[ElementBounds] = []
    for element in elements or []:
        if not isinstance(element, dict):
            continue
        if _resource_id_of(element) != resource_id:
            continue
        bounds = get_bounds_for_element(element)
        if bounds is not None:
            candidates.append(bounds)

    if not candidates:
        # 说清楚「是什么问题」和「模型该干什么」。
        #
        # 真机实测（2026-10-20，exec 340）里，模型看到原来那句
        # "no element with resource_id=... on the current screen" 时的真实反应是
        # 误诊成「索引指错了人」（原话：「该索引实际是弹幕文本元素，不是点赞
        # 按钮」）。真相是**目标已经不在屏上了** —— 横屏全屏页被视频播完带走，
        # 模型思考 17 秒才动手。
        #
        # 所以这句话必须把因果讲明白：东西没了，不是指错了；怎么办是重读屏幕、
        # 换一个**现在还在**的元素。
        return None, (
            f"the element with resource_id={resource_id!r} is no longer on the screen "
            f"(the screen changed after it was read — a page may have been left, or a "
            f"video may have finished). It is NOT a wrong-index problem: re-read the "
            f"current screen and pick an element that is on it now."
        )

    if len(candidates) == 1 or hint is None:
        pick = candidates[index] if index < len(candidates) else candidates[0]
    else:
        # 同 id 有多个：用「离模型当时看到的位置最近的那个」消歧。
        # 这是提示不是门限 —— 挑错也只是挑错一个同 id 的兄弟节点。
        hx, hy = hint.get_center().x, hint.get_center().y
        pick = min(
            candidates,
            key=lambda b: (b.get_center().x - hx) ** 2 + (b.get_center().y - hy) ** 2,
        )
    return pick, ""
