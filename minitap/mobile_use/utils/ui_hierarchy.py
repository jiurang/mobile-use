import re
from typing import Any

from pydantic import BaseModel, Field

from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


def __find_element_by_ressource_id_in_rich_hierarchy(
    hierarchy: list[dict], resource_id: str
) -> dict | None:
    """
    Retrieves all the sibling elements for a given resource ID from a nested dictionary.

    Args:
      hierarchy (dict): The nested dictionary representing the UI hierarchy.
      resource_id (str): The resource-id to find.

    Returns:
      list: A list of the sibling elements, or None if the resource_id is not found.
    """
    if not hierarchy:
        return None

    for child in hierarchy:
        if child.get("attributes", {}).get("resource-id") == resource_id:
            return child.get("attributes", {})

    for child in hierarchy:
        result = __find_element_by_ressource_id_in_rich_hierarchy(
            child.get("children", []), resource_id
        )
        if result is not None:
            return result

    return None


def text_input_is_empty(text: str | None, hint_text: str | None) -> bool:
    return not text or text == hint_text


def find_element_by_resource_id(
    ui_hierarchy: list[dict],
    resource_id: str,
    index: int | None = None,
    is_rich_hierarchy: bool = False,
) -> dict | None:
    """
    Find a UI element by its resource-id in the UI hierarchy.

    Args:
        ui_hierarchy: List of UI element dictionaries
        resource_id: The resource-id to search for
            (e.g., "com.google.android.settings.intelligence:id/open_search_view_edit_text")

    Returns:
        The complete UI element dictionary if found, None otherwise
    """
    if is_rich_hierarchy:
        return __find_element_by_ressource_id_in_rich_hierarchy(ui_hierarchy, resource_id)

    def search_recursive(elements: list[dict]) -> dict | None:
        for element in elements:
            if isinstance(element, dict):
                if element.get("resourceId") == resource_id:
                    idx = index or 0
                    if idx == 0:
                        return element
                    idx -= 1
                    continue

                children = element.get("children", [])
                if children:
                    result = search_recursive(children)
                    if result:
                        return result
        return None

    return search_recursive(ui_hierarchy)


def is_element_focused(element: dict) -> bool:
    return element.get("focused", None) == "true"


def get_element_text(element: dict, hint_text: bool = False) -> str | None:
    if hint_text:
        return element.get("hintText", None)
    return element.get("text", None)


class Point(BaseModel):
    x: int
    y: int


class ElementBounds(BaseModel):
    x: int = Field(description="The x coordinate of the top-left corner of the element.")
    y: int = Field(description="The y coordinate of the top-left corner of the element.")
    width: int = Field(description="The width of the element.")
    height: int = Field(description="The height of the element.")

    def get_center(self) -> Point:
        return Point(x=self.x + self.width // 2, y=self.y + self.height // 2)

    def get_relative_point(self, x_percent: float, y_percent: float) -> Point:
        """
        Returns the coordinates of the point at x_percent of the width and y_percent
        of the height of the element.

        Ex if x_percent = 0.95 and y_percent = 0.95,
        the point is at the bottom right of the element:
        <------>
        |      |
        |     x|
        <------>
        """
        return Point(
            x=int(self.x + self.width * x_percent),
            y=int(self.y + self.height * y_percent),
        )


_ANDROID_BOUNDS_RE = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")


def _coord_to_int(value: Any) -> int:
    """坐标转 int；iOS 客户端给的是 float。非数字会抛异常。"""
    return int(round(float(value)))


def get_bounds_for_element(element: dict) -> ElementBounds | None:
    """返回元素的 bounds，读不出来时返回 None。

    覆盖各平台客户端实际会产出的全部形状：
      - Android：``"bounds": "[x1,y1][x2,y2]"``（是**字符串**，来自 uiautomator）
      - iOS / 云设备：``"frame": {"x", "y", "width", "height"}``（float）
      - 部分调用方/测试：``"bounds": {"x", "y", "width", "height"}``
    """
    for key in ("bounds", "frame"):
        raw = element.get(key)
        if not raw:
            continue

        # Android 的字符串形式给的是两个角点，不是 x/y/width/height。
        if isinstance(raw, str):
            match = _ANDROID_BOUNDS_RE.match(raw.strip())
            if not match:
                logger.error(f"Failed to parse bounds: {raw!r}")
                return None
            x1, y1, x2, y2 = (_coord_to_int(v) for v in match.groups())
            return ElementBounds(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

        if isinstance(raw, dict):
            try:
                return ElementBounds(
                    x=_coord_to_int(raw["x"]),
                    y=_coord_to_int(raw["y"]),
                    width=_coord_to_int(raw["width"]),
                    height=_coord_to_int(raw["height"]),
                )
            except (KeyError, TypeError, ValueError) as e:
                logger.error(f"Failed to validate bounds: {e}")
                return None

    return None
