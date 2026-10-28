from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.controllers.types import (
    CoordinatesSelectorRequest,
    PercentagesSelectorRequest,
)
from minitap.mobile_use.controllers.unified_controller import UnifiedMobileController
from minitap.mobile_use.graph.state import State
from minitap.mobile_use.tools.types import Target
from minitap.mobile_use.utils.failed_targets import (
    ActionFailureFact,
    FailedTarget,
    append_failure,
    breaker_reason,
    describe,
    make_key,
    record_failure,
)
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.ocr_view import resolve_ocr_index
from minitap.mobile_use.utils.screen_elements import ScreenElement, resolve_screen_index
from minitap.mobile_use.utils.ui_hierarchy import (
    ElementBounds,
    Point,
    find_element_by_resource_id,
    get_bounds_for_element,
    get_element_text,
    is_element_focused,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 失败账本：让「目标已经不在屏上了」这件事进到模型下一轮的上下文里，
# 并在同一个目标连着失败时直接熔断（见 utils/failed_targets.py 的说明）。
# ---------------------------------------------------------------------------


def target_key(target: Target) -> str:
    """给一个 target 算账本键：索引 + 元素身份。

    ⚠️ **必须把 ``ocr_index`` 也算进去**（2026-10-21 补）。

    没有统一元素表时（iOS / 云设备 / `perception.ui_hierarchy=False`），
    模型只能填 ``Target.ocr_index`` —— 工具侧的 ``resolve_target`` 就是按它
    索引 ``latest_ocr_results`` 的。

    而如果这里只读 ``target.index``，那条路径下 ``index`` 恒为 None，
    键会退化成 ``text=...`` 或 ``"unknown"``：
    * 熔断账本把「同一个 ocr_index 连着失败」判成「换了目标」→ 连续失败计数失效；
    * ``render_failure_facts`` 渲染成 ``[unknown] "点赞"`` → 模型看不到是第几行。

    ⇒ 取 ``index or ocr_index``：两条路径下键都稳定。
    """
    return make_key(
        index=target.index if target.index is not None else target.ocr_index,
        resource_id=target.resource_id,
        # 视觉目标没有文字 / id，用描述当身份 —— 否则键退化成 "unknown"，
        # 「同一个 vision 目标连着失败」就判不出来（见 target_key 的文档）。
        label=target.text or target.vision or None,
    )


def target_label(target: Target) -> str:
    """给人看的目标描述。"""
    return describe(
        target.index if target.index is not None else target.ocr_index,
        target.resource_id,
        target.text or target.vision,
    )


def check_failure_breaker(state: State, target: Target) -> str | None:
    """这个目标是不是已经连着失败到阈值了？是就返回拒绝理由（None = 放行）。

    熔断的理由不是「怕重复干活」，而是**重复点一个已经不在屏上的元素**：
    它必然失败，而且每次都要白花一次 1.2s 的 live dump。模型如果一直在同一个
    目标上打转，这层能把它顶回去，让它去重读屏幕、换一个现在还在的元素。
    """
    return breaker_reason(state.failed_targets, target_key(target))


def record_target_failure(
    state: State,
    target: Target,
    reason: str,
) -> list[FailedTarget]:
    """记一次失败，返回新的账本（熔断用：换键即替换、成功即清）。"""
    return record_failure(
        state.failed_targets,
        key=target_key(target),
        index=target.index,
        resource_id=target.resource_id,
        label=target.text,
        reason=reason,
    )


def record_target_failure_fact(
    state: State,
    target: Target,
    reason: str,
) -> list[dict]:
    """记一次失败到**跨子目标的事实账本**（只增不减）。

    和 `record_target_failure` 的区别：

    * 熔断账本（`state.failed_targets`）是「**当下**这个目标连着失败了几次」
      的即时保护 —— 换键即替换、成功即清。
    * 这个事实账本（`state.executor_failures`）是「**整个任务里**哪些目标
      试过失败了」的历史记录 —— 只增不减，跨子目标可见。

    为什么必须分开：熔断账本在一次成功点击后就被清空了，
    而汇报要交给 Cortex 的「失败事实」恰恰发生在子目标**成功收尾**的时候 ——
    从熔断账本里取会取到空。
    """
    return append_failure(
        state.executor_failures,
        key=target_key(target),
        index=target.index,
        resource_id=target.resource_id,
        label=target.text,
        reason=reason,
    )


def last_action_failure_fact(target: Target) -> ActionFailureFact:
    """「上一步失败的目标」—— 给 Contextor 并进 `[上一步的结果]` 那一行。

    **只给身份，不给结论、也不重复原因**：

    * **不给结论**：「不要再用这个目标」是相对于目标的判断，框架不知道目标
      （见 `agents/contextor/contextor.py` 的 `render_action_result`）。
    * **不重复原因**：失败原因就在上一行的工具回执里，而错误结果**逐字保留**
      （不截断）—— 再抄一遍就是每轮白烧 token。

    ⇒ 这里补的是工具回执**没有**的东西：**元素的可读名字**
    （回执里只有 resource-id）。

    ⚠️ **不带 index**：索引每轮重编号，带上会让模型点错元素
    （见 `utils/failed_targets.py` 的 `describe_target`）。

    ⇒ 返回**原始数据**（不预先拼字符串）：拼字符串是渲染层的事
    （`describe_target` 统一负责「文字 + id 尾段」的格式）。

    ⚠️ 两个字段**都可能为空**（模型只给了索引时就是这样）：空身份**不是**
    「没有失败」，而是「失败了但认不出是哪个目标」—— 由渲染层决定怎么措辞，
    这里不替它下判断。
    """
    return {
        "label": (target.text or "").strip(),
        "resource_id": target.resource_id or "",
    }


def find_element_by_text(
    ui_hierarchy: list[dict], text: str, index: int | None = None
) -> dict | None:
    """
    Find a UI element by its text content (adapted to both flat and rich hierarchy)

    This function performs a recursive, case-insensitive search.
    Checks text, label, and value fields for iOS compatibility.

    Args:
        ui_hierarchy: List of UI element dictionaries.
        text: The text content to search for.
        index: Optional index to select nth matching element.

    Returns:
        The complete UI element dictionary if found, None otherwise.
    """
    matches: list[dict] = []

    def search_recursive(elements: list[dict]) -> None:
        for element in elements:
            if isinstance(element, dict):
                src = element.get("attributes", element)
                # Check text (Android), label (iOS), and value (iOS)
                element_text = src.get("text", "")
                element_label = src.get("label", "")
                element_value = src.get("value", "")

                # Guard against non-string values
                if not isinstance(element_text, str):
                    element_text = ""
                if not isinstance(element_label, str):
                    element_label = ""
                if not isinstance(element_value, str):
                    element_value = ""

                text_lower = text.lower()
                if (
                    (element_text and text_lower == element_text.lower())
                    or (element_label and text_lower == element_label.lower())
                    or (element_value and text_lower == element_value.lower())
                ):
                    matches.append(element)

                if children := element.get("children", []):
                    search_recursive(children)

    search_recursive(ui_hierarchy)

    if not matches:
        return None

    if index is None:
        idx = 0
    elif index < 0:
        return None
    else:
        idx = index
    if idx < len(matches):
        return matches[idx]
    return None


async def tap_bottom_right_of_element(bounds: ElementBounds, ctx: MobileUseContext):
    bottom_right: Point = bounds.get_relative_point(x_percent=0.99, y_percent=0.99)
    await tap(
        ctx=ctx,
        selector_request=SelectorRequestWithCoordinates(
            coordinates=CoordinatesSelectorRequest(
                x=bottom_right.x,
                y=bottom_right.y,
            ),
        ),
    )


async def move_cursor_to_end_if_bounds(
    ctx: MobileUseContext,
    state: State,
    target: Target,
    elt: dict | None = None,
) -> dict | None:
    """
    Best-effort move of the text cursor near the end of the input by tapping the
    bottom-right area of the focused element (if bounds are available).

    兜底链：resource_id → bounds → text。
    每一级查不到都要**继续往下走**，不能直接返回 —— 尤其 resource_id 这一级：
    executor 跑的时候 `state.latest_ui_hierarchy` 已经被 cortex 清空了
    （工具侧走 controller 现取），所以按 id 查必然查不到。直接 return 的话，
    后面那条真正管用的坐标分支永远走不到，「把光标移到末尾」就是死的。
    """
    if target.resource_id:
        if not elt:
            elt = find_element_by_resource_id(
                ui_hierarchy=state.latest_ui_hierarchy or [],
                resource_id=target.resource_id,
                index=target.resource_id_index,
            )
        if elt:
            bounds = get_bounds_for_element(elt)
            if bounds:
                logger.debug("Tapping near the end of the input to move the cursor")
                await tap_bottom_right_of_element(bounds=bounds, ctx=ctx)
                logger.debug(f"Tapped end of input {target.resource_id}")
                return elt

    if target.bounds:
        await tap_bottom_right_of_element(target.bounds, ctx=ctx)
        logger.debug("Tapped end of input by coordinates")
        return elt

    if target.text:
        text_elt = find_element_by_text(
            state.latest_ui_hierarchy or [], target.text, index=target.text_index
        )
        if text_elt:
            bounds = get_bounds_for_element(text_elt)
            if bounds:
                await tap_bottom_right_of_element(bounds=bounds, ctx=ctx)
                logger.debug(f"Tapped end of input that had text'{target.text}'")
                return text_elt
        return None

    return None


async def focus_element_if_needed(
    ctx: MobileUseContext, target: Target
) -> Literal["resource_id", "coordinates", "text", "already_focused"] | None:
    """
    Ensures the element is focused, with a sanity check to prevent trusting misleading IDs.

    If no target locator is provided (empty target), assumes the field is already focused
    and returns "already_focused" to allow typing without re-focusing.
    """
    # If no locator is provided, assume already focused (e.g., keyboard is visible)
    if not target.resource_id and not target.bounds and not target.text:
        logger.debug("No target locator provided, assuming element is already focused")
        return "already_focused"

    controller = UnifiedMobileController(ctx)
    rich_hierarchy = await controller.get_ui_elements()
    elt_from_id = None
    if target.resource_id:
        elt_from_id = find_element_by_resource_id(
            ui_hierarchy=rich_hierarchy,
            resource_id=target.resource_id,
            index=target.resource_id_index,
            is_rich_hierarchy=False,
        )

    if elt_from_id and target.text:
        text_from_id_elt = get_element_text(elt_from_id)
        if not text_from_id_elt or target.text.lower() != text_from_id_elt.lower():
            logger.warning(
                f"ID '{target.resource_id}' and text '{target.text}' seem to be on different "
                "elements. Ignoring the resource_id and falling back to other locators."
            )
            elt_from_id = None

    if elt_from_id:
        if not is_element_focused(elt_from_id):
            await tap(
                ctx=ctx,
                selector_request=IdSelectorRequest(id=target.resource_id),  # type: ignore
                index=target.resource_id_index,
            )
            logger.debug(f"Focused (tap) on resource_id={target.resource_id}")
            rich_hierarchy = await controller.get_ui_elements()
            elt_from_id = find_element_by_resource_id(
                ui_hierarchy=rich_hierarchy,
                resource_id=target.resource_id,  # type: ignore
                index=target.resource_id_index,
                is_rich_hierarchy=False,
            )
        if elt_from_id and is_element_focused(elt_from_id):
            logger.debug(f"Text input is focused: {target.resource_id}")
            return "resource_id"
        logger.warning(f"Failed to focus using resource_id='{target.resource_id}'. Fallback...")

    if target.bounds:
        relative_point = target.bounds.get_center()
        await tap(
            ctx=ctx,
            selector_request=SelectorRequestWithCoordinates(
                coordinates=CoordinatesSelectorRequest(x=relative_point.x, y=relative_point.y)
            ),
        )
        logger.debug(f"Tapped on coordinates ({relative_point.x}, {relative_point.y}) to focus.")
        return "coordinates"

    if target.text:
        text_elt = find_element_by_text(rich_hierarchy, target.text, index=target.text_index)
        if text_elt:
            bounds = get_bounds_for_element(text_elt)
            if bounds:
                relative_point = bounds.get_center()
                await tap(
                    ctx=ctx,
                    selector_request=SelectorRequestWithCoordinates(
                        coordinates=CoordinatesSelectorRequest(
                            x=relative_point.x, y=relative_point.y
                        )
                    ),
                )
                logger.debug(f"Tapped on text element '{target.text}' to focus.")
                return "text"

    logger.error(
        "Failed to focus element. No valid locator (resource_id, coordinates, or text) succeeded."
    )
    return None


def validate_coordinates_bounds(
    target: Target, screen_width: int, screen_height: int
) -> str | None:
    """
    Validate that coordinates are within screen bounds.
    Returns error message if invalid, None if valid.
    """
    if not target.bounds:
        return None

    center = target.bounds.get_center()
    errors = []

    if center.x < 0 or center.x >= screen_width:
        errors.append(f"x={center.x} is outside screen width (0-{screen_width})")
    if center.y < 0 or center.y >= screen_height:
        errors.append(f"y={center.y} is outside screen height (0-{screen_height})")

    return "; ".join(errors) if errors else None


def has_valid_selectors(target: Target) -> bool:
    """Check if target has at least one valid selector."""
    has_coordinates = target.bounds is not None
    has_resource_id = target.resource_id is not None and target.resource_id != ""
    has_text = target.text is not None and target.text != ""
    # index / ocr_index 单独就能定位：它们还原成一个带 bounds 的具体元素
    has_index = target.index is not None
    has_ocr_index = target.ocr_index is not None
    return has_coordinates or has_resource_id or has_text or has_index or has_ocr_index


def _apply_element(target: Target, element: ScreenElement) -> Target:
    """把元素表里的一行填进 target 的兜底字段。

    **索引优先于模型自己抄的字段**：索引指向的是模型当时看到的那一行，
    坐标和 id 都是代码给的，不会抄错。
    """
    updates: dict = {"bounds": element.bounds}
    if element.resource_id:
        updates["resource_id"] = element.resource_id
        if target.resource_id_index is None:
            updates["resource_id_index"] = 0
    # UI 行的可见文字放在 label 里，OCR 行的放在 text 里 —— 两个都能当文本定位。
    visible_text = element.text or element.label
    if visible_text:
        updates["text"] = visible_text
        if target.text_index is None:
            updates["text_index"] = 0
    return target.model_copy(update=updates)


def resolve_target(state: State, target: Target) -> tuple[Target, str | None]:
    """把 `target.index` / `target.ocr_index` 解析成具体的元素。

    返回 `(解析后的 target, 错误信息)`。错误信息非空表示「模型给的索引用不了」，
    调用方**必须**把它回给模型 —— 绝不能静默退化成「点别的地方」。

    两条路径
    --------
    1. **有屏幕元素表**（Android，层级树能归一化）：`index` 和 `ocr_index`
       都索引这张表，**同一个编号空间**。UI 行和 OCR 行都在里面。
    2. **没有元素表**（iOS / 云设备，层级树形状不认识）：只有 `ocr_index`，
       它索引 `latest_ocr_results`，这是改造前的行为，保持不变。
    """
    elements = state.latest_screen_elements or []

    if target.index is not None:
        element = resolve_screen_index(elements, target.index)
        if element is None:
            return target, (
                f"index={target.index} is not a row in the screen element table"
                + (f" (the table had {len(elements)} rows)" if elements else " (no table)")
            )
        return _apply_element(target, element), None

    if target.ocr_index is not None:
        if elements:
            # 元素表世界：ocr_index 就是 index 的别名（一个编号空间）。
            element = resolve_screen_index(elements, target.ocr_index)
            if element is None:
                return target, (
                    f"ocr_index={target.ocr_index} is not a row in the screen element table"
                    f" (the table had {len(elements)} rows)"
                )
            return _apply_element(target, element), None
        # 老世界：没有元素表，ocr_index 索引的是 latest_ocr_results。
        # 报错文案要跟改造前逐字一致 —— `tests/mobile_use/test_tap_ocr_index.py`
        # 守着这些字样，模型也是靠它们知道索引错在哪。
        ocr_results = state.latest_ocr_results
        ocr_hit = resolve_ocr_index(ocr_results, target.ocr_index)
        if ocr_hit is None:
            if not ocr_results:
                return target, "no OCR results available for this screen"
            available = len(ocr_results)
            return target, (
                f"out of range; the OCR block had {available} line(s), "
                f"valid range is 1..{available}"
            )
        # 连文本一起填上：工具回显时要用，模型才知道索引点到了什么。
        return target.model_copy(update={"bounds": ocr_hit.bounds, "text": ocr_hit.text}), None

    return target, None


class IdSelectorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str

    def to_dict(self) -> dict[str, str | int]:
        return {"id": self.id}


class TextSelectorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str

    def to_dict(self) -> dict[str, str | int]:
        return {"text": self.text}


class SelectorRequestWithCoordinates(BaseModel):
    model_config = ConfigDict(extra="forbid")
    coordinates: CoordinatesSelectorRequest

    def to_dict(self) -> dict[str, str | int]:
        return {"point": self.coordinates.to_str()}


class SelectorRequestWithPercentages(BaseModel):
    model_config = ConfigDict(extra="forbid")
    percentages: PercentagesSelectorRequest

    def to_dict(self) -> dict[str, str | int]:
        return {"point": self.percentages.to_str()}


class IdWithTextSelectorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    text: str

    def to_dict(self) -> dict[str, str | int]:
        return {"id": self.id, "text": self.text}


SelectorRequest = (
    IdSelectorRequest
    | SelectorRequestWithCoordinates
    | SelectorRequestWithPercentages
    | TextSelectorRequest
    | IdWithTextSelectorRequest
)


def _extract_resource_id_and_text_from_selector(
    selector: SelectorRequest,
) -> tuple[str | None, str | None]:
    """Extract resource_id and text from a selector."""
    resource_id = None
    text = None

    if isinstance(selector, IdSelectorRequest):
        resource_id = selector.id
    elif isinstance(selector, TextSelectorRequest):
        text = selector.text
    elif isinstance(selector, IdWithTextSelectorRequest):
        resource_id = selector.id
        text = selector.text

    return resource_id, text


async def tap(
    ctx: MobileUseContext,
    selector_request: SelectorRequest,
    index: int | None = None,
):
    """
    Tap on a selector.
    Index is optional and is used when you have multiple views matching the same selector.
    ui_hierarchy is optional and used for ADB taps to find elements.
    """
    controller = UnifiedMobileController(ctx)
    if isinstance(selector_request, SelectorRequestWithCoordinates):
        result = await controller.tap_at(
            x=selector_request.coordinates.x, y=selector_request.coordinates.y
        )
        return result.error if result.error else None
    if isinstance(selector_request, SelectorRequestWithPercentages):
        coords = selector_request.percentages.to_coords(
            width=ctx.device.device_width,
            height=ctx.device.device_height,
        )
        return await controller.tap_at(coords.x, coords.y)

    # For other selectors, we need the UI hierarchy
    resource_id, text = _extract_resource_id_and_text_from_selector(selector_request)

    return await controller.tap_element(
        resource_id=resource_id,
        text=text,
        index=index if index is not None else 0,
    )
