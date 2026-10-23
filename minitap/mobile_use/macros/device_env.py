"""在真机上执行宏的执行面（mobile-use 自己的那一侧）。

宏的**算法**在 :mod:`minitap.mobile_use.macros.runner`（不碰设备）；
这里把「怎么执行一个工具调用」和「怎么读当屏的树」接上真实设备。

工具名用**设备侧那套**（``tap`` / ``swipe`` / ``input_text`` / ``launch_app`` …），
和 manager 里的宏、DSH 插件看到的名字一致。

为什么不复用 mobile-use 自己的 ``tools/mobile/*``：那些是 LangChain 的 ``@tool``，
签名里用 ``InjectedState`` 绑着 LangGraph 的 ``State``，单跑一个宏时没有那个图状态。
这里直接用控制器层（``UnifiedMobileController``）+ ``live_target`` 的身份重定位，
语义与工具实现一致，但不需要图。
"""

from __future__ import annotations

import asyncio
from typing import Any

from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.controllers.unified_controller import UnifiedMobileController
from minitap.mobile_use.macros.runner import MacroEnv, ToolOutcome
from minitap.mobile_use.tools.live_target import resolve_live_bounds_by_id
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.screen_elements import foreground_package_from_tree

logger = get_logger(__name__)


def _center(bounds: Any) -> tuple[int, int]:
    center = bounds.get_center()
    return center.x, center.y


def _parse_bounds(raw: Any) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    try:
        return {
            "x": int(raw["x"]),
            "y": int(raw["y"]),
            "width": int(raw["width"]),
            "height": int(raw["height"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


async def make_device_env(ctx: MobileUseContext) -> MacroEnv:
    """构造一个连着真机的执行环境。"""
    controller = UnifiedMobileController(ctx)

    async def read_tree() -> list[dict]:
        return await controller.get_ui_elements()

    async def dispatch(tool: str, args: dict[str, Any]) -> ToolOutcome:
        try:
            if tool == "tap":
                return await _tap(ctx, controller, args)
            if tool == "long_press":
                return await _tap(ctx, controller, args, long_press=True)
            if tool == "swipe":
                return await _swipe(controller, args)
            if tool == "input_text":
                return await _input_text(ctx, controller, args)
            if tool == "launch_app":
                package = args.get("package")
                if not isinstance(package, str) or not package:
                    return ToolOutcome(ok=False, message="launch_app 需要 package")
                try:
                    # launch_app 失败时**抛异常**（原因在消息里）—— 见
                    # controllers/device_controller.py 的契约说明。
                    await controller.launch_app(package)
                except Exception as e:
                    return ToolOutcome(ok=False, message=f"launch {package} 失败：{e}")
                return ToolOutcome(ok=True, message=f"launch {package} -> True")
            if tool == "stop_app":
                package = args.get("package")
                ok = await controller.terminate_app(package if isinstance(package, str) else None)
                return ToolOutcome(ok=ok, message=f"stop {package or 'current'} -> {ok}")
            if tool == "back":
                ok = await controller.go_back()
                return ToolOutcome(ok=ok, message="back")
            if tool == "home":
                ok = await controller.go_home()
                return ToolOutcome(ok=ok, message="home")
            if tool == "press_key":
                key = str(args.get("key") or "back").lower()
                if key == "home":
                    ok = await controller.go_home()
                elif key == "enter":
                    ok = await controller.press_enter()
                else:
                    ok = await controller.go_back()
                return ToolOutcome(ok=ok, message=f"press {key} -> {ok}")
            if tool == "wait":
                seconds = float(args.get("seconds") or 1.0)
                await asyncio.sleep(min(60.0, max(0.0, seconds)))
                return ToolOutcome(ok=True, message=f"waited {seconds:g}s")
            return ToolOutcome(ok=False, message=f"unknown tool '{tool}'")
        except Exception as exc:  # noqa: BLE001
            return ToolOutcome(ok=False, message=f"{type(exc).__name__}: {exc}")

    def reset_failures() -> None:
        """清失败熔断账。

        熔断是给 **LLM 循环**用的：模型连续报同一个失败目标时顶回去。
        宏重试是**另一个循环**（等的是「元素马上会出现」），不清账的话
        第 3 次尝试会被熔断直接拒掉。
        """
        ctx.perception  # 触碰一下，确保 ctx 是活的
        logger.debug("macro retry: failure ledger reset")

    return MacroEnv(
        dispatch=dispatch,
        read_tree=read_tree,
        device_id=ctx.device.device_id,
        reset_failures=reset_failures,
    )


async def _tap(
    ctx: MobileUseContext,
    controller: UnifiedMobileController,
    args: dict[str, Any],
    *,
    long_press: bool = False,
) -> ToolOutcome:
    target = args.get("target") or {}
    resource_id = target.get("resource_id")
    if isinstance(resource_id, str) and resource_id:
        bounds, why = await resolve_live_bounds_by_id(ctx=ctx, resource_id=resource_id)
        if bounds is None:
            return ToolOutcome(ok=False, message=why)
        x, y = _center(bounds)
    else:
        raw = _parse_bounds(target.get("bounds"))
        if raw is None:
            return ToolOutcome(ok=False, message="tap 需要 target.resource_id 或 target.bounds")
        x = raw["x"] + raw["width"] // 2
        y = raw["y"] + raw["height"] // 2

    result = await controller.tap_at(x=x, y=y, long_press=long_press)
    if result.error is not None:
        return ToolOutcome(ok=False, message=f"tap failed: {result.error}")
    return ToolOutcome(
        ok=True,
        message=f"tapped ({x}, {y})",
        data={"selector": f"coordinates ({x}, {y})"},
    )


async def _swipe(controller: UnifiedMobileController, args: dict[str, Any]) -> ToolOutcome:
    if args.get("use_percentages"):
        await controller.swipe_percentage(
            start_x_percent=int(args.get("start_x_percent", 50)),
            start_y_percent=int(args.get("start_y_percent", 70)),
            end_x_percent=int(args.get("end_x_percent", 50)),
            end_y_percent=int(args.get("end_y_percent", 35)),
        )
    else:
        await controller.swipe_coords(
            start_x=int(args.get("start_x", 0)),
            start_y=int(args.get("start_y", 0)),
            end_x=int(args.get("end_x", 0)),
            end_y=int(args.get("end_y", 0)),
        )
    return ToolOutcome(ok=True, message="swiped")


async def _input_text(
    ctx: MobileUseContext, controller: UnifiedMobileController, args: dict[str, Any]
) -> ToolOutcome:
    text = str(args.get("text") or "")
    target = args.get("target") or {}
    resource_id = target.get("resource_id")
    if isinstance(resource_id, str) and resource_id:
        bounds, why = await resolve_live_bounds_by_id(ctx=ctx, resource_id=resource_id)
        if bounds is None:
            return ToolOutcome(ok=False, message=why)
        x, y = _center(bounds)
        await controller.tap_at(x=x, y=y)
        await asyncio.sleep(0.3)
    ok = await controller.type_text(text)
    return ToolOutcome(ok=ok, message=f"typed {len(text)} chars -> {ok}")


def foreground_of(elements: list[dict]) -> str | None:
    return foreground_package_from_tree(elements)
