"""屏幕「上下文指纹」—— 让一个元素在 app 改版后仍然能被找到。

为什么需要
----------
真机实测（2026-10-22，Redmi Turbo 4 / 快手极速版）：

* **id 会改**：同一个「搜索」入口，4 小时内从 ``id/right_tv`` 变成
  ``id/thanos_home_top_search``；点赞按钮从 ``id/like_button`` 变成
  ``id/like_element_click_layout``。
  ⇒ 靠 id 回放的宏，app 一发版/灰度就全废。
* **位置也会平移**：屏幕尺寸一天内 2530 → 2660 → 2712（导航栏模式变了），
  中下部的元素整体平移 230px；顶栏元素只差 15px。

⇒ 唯一还稳的是**结构关系**：元素**旁边**是谁。

* 搜索按钮的邻居是「热门 / 同城 / 商城 / 生活 / 关注 / 发现」（**文字**）
* 点赞按钮的邻居是「collect / negative / share / comment」（**id**）

指纹长什么样
------------
::

    {"t": "搜索", "i": 2, "w": ["热门", "同城", "搜索", "发现", "生活"]}

* ``t`` —— 目标自己的 token
* ``i`` —— 目标在窗口里的下标
* ``w`` —— 以目标为中心的窗口（前后各 3 个邻居）的 token 序列

**很短**（~60 字符），所以可以塞进工具结果文字里 —— 工具结果文字本来就在
下一轮 prompt 的历史里，也进 trace，**一条通道同时解决两件事**。

token 怎么算
------------
**文字优先，id 兜底**：

* 有可见文字 → 用文字（截断到 12 字）
* 没文字（图标/图片按钮）→ 用**归一化后的 id 尾段**
  （``like_button`` / ``like_element_click_layout`` → ``like``）

为什么不能只靠 id 归一化：``right_tv`` → ``thanos_home_top_search``
**根本不是一个词**，归一化救不了；但它们的**邻居**（那几个 tab 的文字）没变。

⚠️ **文字可能是会变的**（点赞数、时钟）。这不是问题 —— 变了的 token 在
LCS 对齐里就是「对不上」的那几个，会被跳过。**对齐的是结构，不是内容。**
"""

from __future__ import annotations

from typing import Any

#: id 尾段里这些后缀没有语义，去掉后剩下的部分才是「这个控件是干什么的」。
_ID_NOISE_SUFFIXES = (
    "_element_click_layout",
    "_click_layout",
    "_click_area",
    "_container",
    "_text_view",
    "_textview",
    "_layout",
    "_root",
    "_view",
    "_text",
    "_button",
    "_btn",
    "_image",
    "_img",
    "_icon",
    "_tv",
)

#: 指纹窗口：目标前后几个邻居。
#:
#: **为什么是 8 而不是 3**（真机实测 2026-10-22，APK 里的快手极速版）：
#: 同一个「搜索」入口在两次录制里的邻居完全不同（一次「传奇/查找/查找」，
#: 一次「关注/发现/查找/头像」）—— 窗口太窄时 LCS 没有足够公共锚点，对齐失败。
#:
#: ⚠️ 但**它仍不是最强的兜底**：那两次录制里同一个按钮的**文字都变了**
#: （「搜索」→「查找」，快手改了文案）⇒ 真正救了那一步的是**位置**兜底
#: （偏差 15px）。指纹的价值主要在「是不是同一屏」这个**前置条件**上。
DEFAULT_WINDOW = 8

#: token 的最大长度（文字可能很长）。
MAX_TOKEN_CHARS = 12


def normalize_resource_id(resource_id: str) -> str:
    """``com.ks:id/like_element_click_layout`` → ``like``。

    去掉包名、去掉 ``:id/``、去掉没有语义的后缀。全空就返回空串。
    """
    if not resource_id:
        return ""
    tail = resource_id.split(":", 1)[-1]
    tail = tail.rsplit("/", 1)[-1]
    tail = tail.strip().lower()
    changed = True
    while changed:
        changed = False
        for suffix in _ID_NOISE_SUFFIXES:
            if tail.endswith(suffix) and len(tail) > len(suffix):
                tail = tail[: -len(suffix)]
                changed = True
    return tail


def element_token(element: Any) -> str:
    """一个元素的短 token：**文字优先，id 兜底**。

    ``element`` 是 :class:`~minitap.mobile_use.utils.screen_elements.ScreenElement`
    或同形状的 dict。
    """
    if isinstance(element, dict):
        text = element.get("text") or element.get("label") or ""
        resource_id = element.get("resource_id") or ""
        class_name = element.get("class_name") or ""
    else:
        text = getattr(element, "text", "") or getattr(element, "label", "") or ""
        resource_id = getattr(element, "resource_id", "") or ""
        class_name = getattr(element, "class_name", "") or ""

    text = str(text).strip()
    if text:
        return text[:MAX_TOKEN_CHARS]
    normalized = normalize_resource_id(str(resource_id))
    if normalized:
        return normalized[:MAX_TOKEN_CHARS]
    return str(class_name).rsplit(".", 1)[-1][:MAX_TOKEN_CHARS]


def _resource_id_of(element: dict) -> str:
    """元素的 resource-id（``resource-id`` / ``resourceId`` / ``resource_id`` 都认）。"""
    for key in ("resource-id", "resourceId", "resource_id"):
        value = element.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _is_interactive(element: Any) -> bool:
    """只取可交互的元素（决策只关心这些）。

    **两种形状都要认**：

    * **元素表**（``ScreenElement`` 或同形状 dict）：看 ``flags``（C/L/F/R）
    * **原始层级树**（uiautomator 的 dict）：看 ``clickable`` / ``scrollable`` 等属性
    """
    if isinstance(element, dict):
        if "resource-id" in element or "resourceId" in element:
            return _is_interactive_tree(element)  # 原始树
        kind = element.get("kind")
        flags = str(element.get("flags") or "")
        resource_id = str(element.get("resource_id") or "")
        if kind == "ocr":
            return False  # OCR 行没有稳定身份
        return "C" in flags or "L" in flags or "F" in flags or "R" in flags or bool(resource_id)
    kind = getattr(element, "kind", None)
    flags = str(getattr(element, "flags", "") or "")
    resource_id = str(getattr(element, "resource_id", "") or "")
    if kind == "ocr":
        return False
    return "C" in flags or "L" in flags or "F" in flags or "R" in flags or bool(resource_id)


def _is_interactive_tree(element: dict) -> bool:
    """**原始层级树**里的可交互元素判定。

    和 :func:`_is_interactive` 的区别：树里没有 ``flags``，交互性直接看
    uiautomator 的属性。两边的口径必须一致，否则录制的序列和当屏的序列
    对不上，LCS 就没有锚点。
    """
    if _truthy(element.get("clickable")):
        return True
    if _truthy(element.get("long-clickable")):
        return True
    if _truthy(element.get("scrollable")):
        return True
    if _truthy(element.get("focusable")):
        return True
    cls = str(element.get("class") or "")
    return cls.endswith("EditText")


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1")


def tree_token(element: dict) -> str:
    """原始树里一个元素的 token（与 :func:`element_token` 同口径）。

    树里的可见文字在 ``text`` 或 ``content-desc``；元素表里那一份叫 ``label``。
    """
    text = str(element.get("text") or element.get("content-desc") or "").strip()
    if text:
        return text[:MAX_TOKEN_CHARS]
    normalized = normalize_resource_id(str(element.get("resource-id") or element.get("resourceId") or ""))
    if normalized:
        return normalized[:MAX_TOKEN_CHARS]
    return str(element.get("class") or "").rsplit(".", 1)[-1][:MAX_TOKEN_CHARS]


def tree_tokens(elements: list[dict] | None) -> list[str]:
    """当屏（原始树）的 token 序列 —— 回放时用 :func:`align_window` 对齐。"""
    if not elements:
        return []
    return [tree_token(e) for e in elements if _is_interactive_tree(e)]


def build_context(
    elements: list[Any] | None,
    target_resource_id: str,
    *,
    window: int = DEFAULT_WINDOW,
) -> dict[str, Any] | None:
    """算出目标的上下文指纹。

    Args:
        elements: 当屏的元素表（``ScreenElement`` 列表）。
        target_resource_id: 目标的**完整** resource-id。
        window: 前后各取几个邻居。

    Returns:
        ``{"t": 目标的 token, "i": 在窗口里的下标, "w": 窗口 token 列表}``；
        找不到目标返回 None。
    """
    if not elements or not target_resource_id:
        return None

    interactive = [e for e in elements if _is_interactive(e)]
    tokens = [element_token(e) for e in interactive]

    target_index = -1
    for i, element in enumerate(interactive):
        if isinstance(element, dict):
            rid = element.get("resource_id") or _resource_id_of(element)
        else:
            rid = getattr(element, "resource_id", "")
        if str(rid or "") == target_resource_id:
            target_index = i
            break
    if target_index < 0:
        return None

    start = max(0, target_index - window)
    end = min(len(tokens), target_index + window + 1)
    return {
        "t": tokens[target_index],
        "i": target_index - start,
        "w": tokens[start:end],
    }


def format_context(context: dict[str, Any] | None) -> str:
    """把指纹渲染成能塞进工具结果文字的短串。

    形如 ``ctx:{"t":"like","i":2,"w":["collect","like","comment"]}``（~70 字符）。

    **用 JSON 而不是逗号分隔**：token 来自控件的可见文字，**里面本来就有逗号**
    （「点赞, 收藏」），用分隔符切窗口就会切错位置。
    """
    if not context:
        return ""
    import json

    payload = {
        "t": context.get("t", ""),
        "i": int(context.get("i", 0)),
        "w": [str(t) for t in (context.get("w") or [])],
    }
    return "ctx:" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_context(text: str) -> dict[str, Any] | None:
    """从工具结果文字里把指纹抠出来（编译器用）。"""
    import json

    marker = text.find("ctx:")
    if marker < 0:
        return None
    chunk = text[marker + 4 :].strip()
    # JSON 优先；退回到旧的竖线格式（早期版本写的那种）。
    try:
        payload = json.loads(chunk)
        if isinstance(payload, dict) and "w" in payload:
            return {
                "t": str(payload.get("t") or ""),
                "i": int(payload.get("i") or 0),
                "w": [str(t) for t in (payload.get("w") or [])],
            }
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    parts = chunk.split("|", 2)
    if len(parts) < 3:
        return None
    token, index, window = parts[0], parts[1], parts[2]
    try:
        index_value = int(index)
    except ValueError:
        return None
    window_tokens = [t for t in window.split(",") if t]
    if index_value < 0 or index_value >= max(1, len(window_tokens)):
        return None
    return {"t": token, "i": index_value, "w": window_tokens}


def align_window(window: list[str], current: list[str]) -> int | None:
    """把录制的窗口对齐到当屏序列，返回**窗口第 0 个**在当屏里的下标。

    用 LCS（最长公共子序列）：对不上的 token（计数变了、元素增删）会被跳过，
    剩下能对上的部分给出锚点。

    ⚠️ 返回的是**窗口起点**的下标，不是目标的下标 —— 目标在窗口里的偏移
    由调用方加。
    """
    if not window or not current:
        return None
    n, m = len(window), len(current)

    # dp[i][j] = window[:i] 与 current[:j] 的 LCS 长度
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if window[i - 1] == current[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 回溯出对齐：窗口第 0 个对到当屏的哪个位置
    i, j = n, m
    anchor: int | None = None
    while i > 0 and j > 0:
        if window[i - 1] == current[j - 1]:
            if i == 1:
                anchor = j - 1
                break
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    if anchor is None:
        # 窗口第 0 个没对上（它自己变了）⇒ 退化：用「窗口里第一个能对上的」当锚点
        for offset, token in enumerate(window):
            for k, candidate in enumerate(current):
                if candidate == token:
                    return k - offset
    return anchor
