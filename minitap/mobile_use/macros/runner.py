"""宏的回放引擎：顺序执行一串工具调用，**不需要模型再想**。

它**不碰设备** —— 设备相关的两件事由 :class:`MacroEnv` 接进来：

* ``dispatch(tool, args)`` —— 执行一次工具调用
* ``read_tree()`` —— 读当屏的层级树（原始元素 dict 列表）

所以这个引擎在任何执行面（bridge / runner / 单测）里都能用。

降级链（三级）
------------
1. **``resource_id``** —— 最准，但**会变**（真机实测：快手 4 小时内把「搜索」
   从 ``right_tv`` 改成 ``thanos_home_top_search``）。
2. **录制时的位置** —— id 失效时用。但**位置也会平移**（屏幕尺寸一天内变了
   三次，中下部的元素跟着平移 230px；顶栏只差 15px）。
3. **上下文指纹（邻居序列 + LCS 对齐）** —— 前两级都救不了时靠它：
   邻居（相邻控件的文字或归一化 id）比 id 和坐标都稳。

每一次降级都会记进 :attr:`MacroRunResult.degradations` —— **降级必须上报**，
否则调用方不知道这个宏已经过期了。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from minitap.mobile_use.macros.model import Macro, MacroRunResult, MacroStep
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.screen_elements import foreground_package_from_tree
from minitap.mobile_use.utils.screen_fingerprint import align_window, tree_tokens

logger = get_logger(__name__)

#: 「元素不在屏上」类失败的判据 —— 这类失败**值得等**（元素可能马上出现）。
MISSING_REASONS = (
    "no longer on the screen",
    "not on the current screen",
    "could not be resolved",
    "no valid selector",
)

#: 位置兜底的最大偏差（像素）。1~2px 的漂移是常态；100px+ 是真的动了。
POSITION_TOLERANCE_PX = 40


@dataclass
class ToolOutcome:
    """一次工具调用的结果（执行面给回来的）。"""

    ok: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class MacroEnv:
    """执行环境：引擎唯一需要知道的两件事 + 两个可选钩子。"""

    #: 执行一次工具调用。
    dispatch: Callable[[str, dict[str, Any]], Awaitable[ToolOutcome]]
    #: 读当屏的层级树（原始元素 dict 列表）。
    read_tree: Callable[[], Awaitable[list[dict]]]
    #: 当前设备序列号（用来核对宏是不是录在这台设备上）。
    device_id: str | None = None
    #: 失败熔断账的清账钩子。
    #:
    #: 熔断（``check_failure_breaker``）是给 **LLM 循环**用的：模型连续报同一个
    #: 失败目标时顶回去让它换个目标。宏重试是**另一个循环** —— 它等的是
    #: 「元素马上会出现」，同一个目标失败两次是正常的。不清账的话第 3 次尝试
    #: 会被熔断直接拒掉。
    reset_failures: Callable[[], None] | None = None


def _looks_like_missing(message: str) -> bool:
    lowered = message.lower()
    return any(reason in lowered for reason in MISSING_REASONS)


def _resource_id_of(element: dict) -> str:
    for key in ("resource-id", "resourceId"):
        value = element.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _bounds_of(element: dict) -> tuple[int, int, int, int] | None:
    """元素矩形 ``(left, top, right, bottom)``。

    ⚠️ uiautomator 的 ``bounds`` 是 ``"[x1,y1][x2,y2]"``（**两个角点**），
    不是 ``(x, y, w, h)``。
    """
    bounds = element.get("bounds")
    if isinstance(bounds, str) and bounds:
        nums: list[int] = []
        for part in bounds.replace("][", ",").strip("[]").split(","):
            try:
                nums.append(int(part))
            except ValueError:
                return None
        if len(nums) == 4:
            return nums[0], nums[1], nums[2], nums[3]
    if isinstance(bounds, dict):
        x = int(bounds.get("x", 0))
        y = int(bounds.get("y", 0))
        return x, y, x + int(bounds.get("width", 0)), y + int(bounds.get("height", 0))
    return None


def _center_of(element: dict) -> tuple[int, int] | None:
    box = _bounds_of(element)
    if box is None:
        return None
    left, top, right, bottom = box
    return (left + right) // 2, (top + bottom) // 2


def _package_of(element: dict) -> str | None:
    package = element.get("package")
    return str(package) if package else None


async def _foreground(env: MacroEnv) -> str | None:
    try:
        elements = await env.read_tree()
    except Exception:  # noqa: BLE001
        return None
    if not elements:
        return None
    return foreground_package_from_tree(elements)


async def _wait_for_app(env: MacroEnv, package: str, timeout_s: float = 20.0) -> bool:
    """等目标 app 真的到前台。

    真机实测（2026-10-22）：``launch_app`` 返回 True 时 app **往往还没起来**，
    前台还是 launcher。不等就会在「还没进 app」的屏上执行后面的步骤。
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if await _foreground(env) == package:
            return True
        await asyncio.sleep(0.6)
    return False


async def _resolve_by_position(
    env: MacroEnv, step: MacroStep, *, app_package: str | None
) -> tuple[dict[str, Any] | None, str]:
    """降级链第 2 级：按**录制时的位置**找当屏的元素。

    ⚠️ **必须限定在目标 app 的元素里**：真机实测，树里同时有**别的 app 的窗口**
    （``com.android.browser:id/rightBtn`` 恰好落在「搜索」的录制位置附近），
    不限定就会点到别的 app 去 —— 比点空更糟。
    """
    recorded = step.fallback_bounds
    if not isinstance(recorded, dict):
        return None, "宏里没有录制时的位置"
    try:
        rx = int(recorded["x"]) + int(recorded["w"]) // 2
        ry = int(recorded["y"]) + int(recorded["h"]) // 2
    except (KeyError, TypeError, ValueError):
        return None, "录制位置格式不对"

    try:
        elements = await env.read_tree()
    except Exception:  # noqa: BLE001
        elements = None
    if not elements:
        return None, "拿不到当屏层级树"

    best: tuple[int, dict] | None = None
    for element in elements:
        if app_package and _package_of(element) and _package_of(element) != app_package:
            continue
        center = _center_of(element)
        if center is None:
            continue
        distance = abs(center[0] - rx) + abs(center[1] - ry)
        if distance > POSITION_TOLERANCE_PX:
            continue
        if best is None or distance < best[0]:
            best = (distance, element)
    if best is None:
        return None, f"录制位置 ({rx},{ry}) 附近没有目标 app 的可交互元素"

    distance, element = best
    box = _bounds_of(element)
    if box is None:
        return None, "命中的元素没有 bounds"
    left, top, right, bottom = box
    return (
        {"x": left, "y": top, "width": right - left, "height": bottom - top},
        f"按录制位置命中 {_resource_id_of(element) or '-'}（偏差 {distance}px）",
    )


async def _resolve_by_context(
    env: MacroEnv, step: MacroStep, *, app_package: str | None
) -> tuple[dict[str, Any] | None, str]:
    """降级链第 3 级：**上下文指纹 + LCS 对齐**。

    真机实测：id 变了、位置也平移了 230px（屏幕尺寸变了）—— 前两级都救不了。
    救得了的是**邻居**：搜索按钮旁边是「热门/同城/商城/生活/关注/发现」，
    点赞按钮旁边是「collect/negative/share/comment」—— 那些没变。
    """
    context = step.fallback_context
    if not isinstance(context, dict):
        return None, "宏里没有上下文指纹"
    window = context.get("w")
    if not isinstance(window, list) or not window:
        return None, "上下文指纹没有窗口"
    try:
        offset = int(context.get("i", 0))
    except (TypeError, ValueError):
        return None, "上下文指纹的偏移量不对"

    try:
        elements = await env.read_tree()
    except Exception:  # noqa: BLE001
        elements = None
    if not elements:
        return None, "拿不到当屏层级树"

    candidates = [
        e for e in elements if not app_package or not _package_of(e) or _package_of(e) == app_package
    ]
    tokens = tree_tokens(candidates)
    anchor = align_window([str(t) for t in window], tokens)
    if anchor is None:
        return None, "当屏序列和录制的窗口对不上"

    live_index = anchor + offset
    if live_index < 0 or live_index >= len(candidates):
        return None, f"对齐后的下标 {live_index} 越界（当屏 {len(candidates)} 个元素）"
    element = candidates[live_index]
    box = _bounds_of(element)
    if box is None:
        return None, "对齐到的元素没有 bounds"
    left, top, right, bottom = box
    return (
        {"x": left, "y": top, "width": right - left, "height": bottom - top},
        f"按上下文指纹命中 {_resource_id_of(element) or '-'}（窗口对齐到 {anchor}+{offset}）",
    )


async def run_macro(
    env: MacroEnv,
    macro: Macro,
    *,
    stop_on_fail: bool = True,
) -> MacroRunResult:
    """回放一个宏。"""
    result = MacroRunResult(ok=True, steps_total=len(macro.steps))

    # ---- 前置条件 0：设备对不对 ----
    if macro.device_serial and env.device_id and env.device_id != macro.device_serial:
        result.ok = False
        result.reason = (
            f"宏是在设备 {macro.device_serial} 上录的，当前设备是 {env.device_id}。"
            "换设备要重新录制这个宏。"
        )
        return result

    # ---- 前置条件 1：前台 app 对不对 ----
    if macro.app_package:
        current = await _foreground(env)
        if current and current != macro.app_package:
            # 第一步是 launch_app 的话交给它切换（下面会等 app 就绪）
            if not (macro.steps and macro.steps[0].tool == "launch_app"):
                result.ok = False
                result.reason = (
                    f"宏属于 {macro.app_package}，但当前前台是 {current}。"
                    "先切到目标 app，或让宏第一步是 launch_app。"
                )
                return result

    for index, step in enumerate(macro.steps, 1):
        wanted = step.wanted_selector()
        attempts = 0
        deadline = time.time() + step.wait_timeout_s
        outcome: ToolOutcome | None = None
        local_degradations: list[dict[str, Any]] = []

        while True:
            outcome = await env.dispatch(step.tool, step.args)
            if outcome.ok:
                break

            # 定位不到 → 试降级链 → 还没有就等它出现
            if (
                step.needs_target()
                and _looks_like_missing(outcome.message)
                and attempts < step.max_retries
                and time.time() < deadline
            ):
                attempts += 1
                if env.reset_failures is not None:
                    env.reset_failures()

                for resolver in (
                    lambda: _resolve_by_position(env, step, app_package=macro.app_package),
                    lambda: _resolve_by_context(env, step, app_package=macro.app_package),
                ):
                    bounds, why = await resolver()
                    if bounds is None:
                        continue
                    fallback = MacroStep(
                        tool=step.tool,
                        args={**step.args, "target": {"bounds": bounds}},
                        optional=step.optional,
                        max_retries=0,
                        wait_timeout_s=0,
                    )
                    fallback_outcome = await env.dispatch(fallback.tool, fallback.args)
                    if fallback_outcome.ok:
                        outcome = fallback_outcome
                        local_degradations.append(
                            {
                                "step": index,
                                "tool": step.tool,
                                "wanted": wanted or "",
                                "got": why,
                            }
                        )
                        break
                if outcome.ok:
                    break

                await asyncio.sleep(0.6)
                continue
            break

        ok = outcome.ok
        message = outcome.message
        data = outcome.data or {}

        # 降级检测：宏要的是 resource_id，工具却用别的路径命中了。
        # ⚠️ **降级链兜底命中时不要再记一次** —— 兜底那一层已经报过原因了，
        # 记两次会让「有几处降级」这个数字对不上。
        if ok and wanted and not local_degradations:
            got = str(data.get("selector") or "")
            if got and not got.startswith("resource_id"):
                local_degradations.append(
                    {"step": index, "tool": step.tool, "wanted": wanted, "got": got}
                )
        result.degradations.extend(local_degradations)
        result.steps.append(
            {
                "step": index,
                "tool": step.tool,
                "ok": ok,
                "message": message[:200],
                "retries": attempts,
            }
        )

        if not ok:
            result.failed_index = index
            result.failed_tool = step.tool
            result.reason = message
            if step.optional:
                result.steps_skipped += 1
                continue
            result.ok = False
            if stop_on_fail:
                return result
            continue

        result.steps_executed += 1

        # 启动完 app 要等它真的到前台，否则下一步会打在 launcher 上
        if step.tool == "launch_app":
            package = macro.app_package or step.args.get("package")
            if isinstance(package, str) and package:
                if not await _wait_for_app(env, package):
                    result.ok = False
                    result.failed_index = index
                    result.failed_tool = step.tool
                    result.reason = f"launch_app 之后等 {package} 到前台超时"
                    if stop_on_fail:
                        return result

    return result
