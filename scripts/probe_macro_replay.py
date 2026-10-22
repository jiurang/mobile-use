"""在真机上回放一个编译出来的宏 —— 用来证伪「动作序列能不能重放」。

回放机制 = 宏真正该用的那套（**不是**拿旧坐标去赌）：

* 每一步用**稳定选择器**（resource_id）在**当屏**重新定位，再点它的**当前位置**
  —— 和 ``tools/live_target.py`` / ``tools/mobile/tap.py`` 走的是同一条路。
* **前置条件**：每步跑之前先确认「屏还是那一屏」（前台包名 + 屏上的 id 集合）。
  不满足就**中止**并说清原因 —— 绝不在错的屏上照样点。
* 定位不到就**明确失败**（不猜、不点空）。
* 坐标必须落在屏内 —— ``adb tap`` 对屏外坐标**不报错**，不校验就是静默点空气。

用法::

    python scripts/probe_macro_replay.py --macro probe_out/macro_exec816.json
    python scripts/probe_macro_replay.py --macro ... --package com.kuaishou.nebula
    python scripts/probe_macro_replay.py --macro ... --dry-run
    python scripts/probe_macro_replay.py --macro ... --continue-on-miss --fuzzy
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from adbutils import AdbClient  # noqa: E402

from minitap.mobile_use.clients.ui_automator_client import UIAutomatorClient  # noqa: E402
from minitap.mobile_use.config import initialize_llm_config  # noqa: E402
from minitap.mobile_use.context import (  # noqa: E402
    DeviceContext,
    DevicePlatform,
    MobileUseContext,
    PerceptionConfig,
)
from minitap.mobile_use.controllers.unified_controller import UnifiedMobileController  # noqa: E402
from minitap.mobile_use.utils.screen_elements import foreground_package_from_tree  # noqa: E402
from minitap.mobile_use.utils.screen_stability import structural_digest  # noqa: E402
from minitap.mobile_use.utils.ui_hierarchy import get_bounds_for_element  # noqa: E402


def build_ctx(serial: str) -> MobileUseContext:
    import uiautomator2 as u2

    ui = UIAutomatorClient(device_id=serial)
    info = u2.connect(serial).info
    device = DeviceContext(
        host_platform="WINDOWS",
        mobile_platform=DevicePlatform.ANDROID,
        device_id=serial,
        device_width=int(info["displayWidth"]),
        device_height=int(info["displayHeight"]),
    )
    return MobileUseContext(
        trace_id="macro-replay",
        device=device,
        adb_client=AdbClient(),
        ui_adb_client=ui,
        llm_config=initialize_llm_config(),
        perception=PerceptionConfig(),
    )


def resource_id_of(element: dict) -> str:
    for key in ("resource-id", "resourceId"):
        value = element.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def bounds_of(element: dict) -> tuple[int, int, int, int] | None:
    """元素在屏上的矩形 ``(left, top, right, bottom)``。

    ⚠️ uiautomator 的 ``bounds`` 字符串是 ``"[x1,y1][x2,y2]"``（**两个角点**），
    **不是** ``(x, y, w, h)``。我一开始按 x,y,w,h 算中心，结果算出 x=1650
    （屏宽才 1220）—— 那是我的 bug，不是 app 的。

    直接用仓库里的实现（``utils/ui_hierarchy.py``），别自己再写一遍。
    """
    b = get_bounds_for_element(element)
    if b is None:
        return None
    return (b.x, b.y, b.x + b.width, b.y + b.height)


def center_of(element: dict) -> tuple[int, int] | None:
    b = get_bounds_for_element(element)
    if b is None:
        return None
    return b.get_center().x, b.get_center().y


def live_ids(elements: list[dict]) -> set[str]:
    return {resource_id_of(e).rsplit("/", 1)[-1] for e in elements if resource_id_of(e)}


def live_package(elements: list[dict]) -> str | None:
    """当前前台 app 的包名。

    ⚠️ **不能取「第一个带 package 的节点」** —— MIUI 的树第一个节点是**桌面**
    （``com.miui.home``），哪怕前台明明是快手。真机上就是这么误判的：
    回放器报了「等 20s 后仍未到前台」，而 ``dumpsys`` 说前台就是快手。

    直接用仓库里已有的实现（``utils/screen_elements.py``，它取「元素最多的那个包」，
    排除 systemui）—— 判据和主流程完全一致，不自己另写一套。
    """
    return foreground_package_from_tree(elements)


def check_precondition(
    step: dict, elements: list[dict], *, app_package: str | None, app_established: bool
) -> tuple[bool, str]:
    """这一步是不是在**同一屏 / 同一个 app** 上？不像就别动。

    两个判据：
    * **前台 app** —— 任务目标 app（不是录制瞬间的前台，那个可能是 runner 自己）。
    * **屏上的 id 集合** —— 录制时能看到的 id，现在还剩多少（≥ 一半算同一屏）。
    """
    if app_package and app_established:
        now_pkg = live_package(elements)
        if now_pkg and now_pkg != app_package:
            return False, f"前台 app 不对：任务目标是 {app_package}，现在是 {now_pkg}"

    expect_ids = set(step.get("expect_ids") or [])
    if expect_ids:
        now_ids = live_ids(elements)
        kept = expect_ids & now_ids
        ratio = len(kept) / len(expect_ids)
        if ratio < 0.5:
            missing = sorted(expect_ids - now_ids)[:4]
            return False, (
                f"屏不是录制那一屏：{len(kept)}/{len(expect_ids)} 个 id 还在"
                f"（{ratio:.0%}），少了 {missing}"
            )
    return True, "前置条件满足"


def resolve_by_selector(
    elements: list[dict], selector: str, *, fuzzy: bool = False
) -> tuple[tuple[int, int] | None, str]:
    """按稳定选择器在**当屏**找元素，返回它的中心坐标。

    * 完整 resource_id → 直接匹配
    * 尾段（``like_button``）→ 匹配 ``*/like_button``
    * ``fuzzy=True`` 才启用**子串**降级 —— 默认关着。

      ⚠️ 真机实测（2026-10-22）：子串降级会把动作打到**错误元素**上而且
      **不报错**（``id/text`` 找不到 → 命中 ``id/textView`` → 点到顶栏一个
      无关元素）。所以它只能是**显式开启**的实验开关，不能是默认行为。
    """
    exact = []
    loose = []
    for e in elements:
        rid = resource_id_of(e)
        if not rid:
            continue
        if rid == selector or rid.endswith("/" + selector):
            exact.append((rid, e))
        elif fuzzy and selector and selector in rid:
            loose.append((rid, e))
    if exact:
        rid, e = exact[0]
        extra = f"（另有 {len(exact) - 1} 个同名节点）" if len(exact) > 1 else ""
        return center_of(e), f"命中 {rid}{extra}"
    if loose:
        rid, e = loose[0]
        return center_of(e), f"[模糊] 命中 {rid}（精确 id '{selector}' 不存在）"
    return None, f"当屏没有 resource_id 含 '{selector}' 的元素"


async def wait_for_app(
    controller: UnifiedMobileController, package: str | None, timeout_s: float = 20.0
) -> tuple[bool, str]:
    """等目标 app 真的到前台。

    真机实测（2026-10-22）：``launch_app`` 返回 True 时，app **往往还没起来**
    —— 前台还是 ``com.miui.home``（launcher）。不等就会在**还没进 app** 的屏上
    执行后面的步骤。
    """
    if not package:
        return True, "没指定包名，跳过等待"
    deadline = time.time() + timeout_s
    seen = ""
    while time.time() < deadline:
        try:
            elements = await controller.get_ui_elements()
            seen = live_package(elements) or seen
            if seen == package:
                return True, f"{package} 已到前台"
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(0.6)
    return False, f"等 {timeout_s:.0f}s 后 {package} 仍未到前台（当前前台 {seen or '未知'}）"


def resolve_by_position(
    elements: list[dict],
    step: dict,
    tolerance: int = 40,
    *,
    app_package: str | None = None,
) -> tuple[tuple[int, int] | None, str]:
    """降级链的第 2 级：**位置**（**只在目标 app 的元素里找**）。

    真机实测（2026-10-22）：快手 4 小时内把「搜索」入口的 id 从
    ``right_tv`` 改成 ``thanos_home_top_search``，但**位置几乎没变**（中心差 15px）。
    ⇒ 对高频改版的 app，位置比 id 稳。

    ⚠️ **必须限定包名**：真机实测，树里同时有**别的 app 的窗口**元素
    （``com.android.browser:id/rightBtn`` 恰好落在「搜索」的录制位置附近），
    不限定就会点到**别的 app** 去 —— 比点空更糟。
    """
    recorded = step.get("recorded_bounds")
    if not recorded:
        return None, "宏里没有录制时的位置"
    rx = recorded["x"] + recorded["w"] // 2
    ry = recorded["y"] + recorded["h"] // 2
    want_cls = (step.get("recorded_class") or "").strip()

    best = None
    for e in elements:
        if app_package and e.get("package") and e.get("package") != app_package:
            continue  # 不是目标 app 的元素 → 不许碰
        c = center_of(e)
        if c is None:
            continue
        dist = abs(c[0] - rx) + abs(c[1] - ry)
        if dist > tolerance:
            continue
        cls = str(e.get("class") or "").rsplit(".", 1)[-1]
        if want_cls and cls and cls != want_cls:
            # class 不一致 → 降级惩罚，排在后面
            dist += 500
        if best is None or dist < best[0]:
            best = (dist, e, c)
    if best is None:
        return None, f"录制位置 ({rx},{ry}) 附近没有目标 app 的可交互元素"
    dist, e, c = best
    rid = resource_id_of(e) or "-"
    return c, f"[位置降级] 命中 {rid}（录制位置 ({rx},{ry})，偏差 {dist}px）"


async def resolve_step_target(
    controller: UnifiedMobileController,
    step: dict,
    *,
    fuzzy: bool = False,
    use_position: bool = True,
    app_package: str | None = None,
    timeout_s: float = 8.0,
) -> tuple[tuple[int, int] | None, str]:
    """等这一步的目标**出现**并返回它的中心坐标。

    真机实测：``launch_app`` 之后 app 已经在前台了，但**顶栏还没渲染出来**
    —— 立刻找元素会「找不到」。所以要**轮询等待**（Appium 的 implicit wait
    那种），而不是找不到就立刻判失败。

    降级链：精确 resource_id → 录制时的位置 → 放弃。
    """
    selector = step.get("stable_selector")
    deadline = time.time() + timeout_s
    last_why = "没试过"
    while True:
        elements = await controller.get_ui_elements()
        if selector:
            center, why = resolve_by_selector(elements, selector, fuzzy=fuzzy)
            if center is not None:
                return center, why
            last_why = why
        if use_position:
            center, why = resolve_by_position(elements, step, app_package=app_package)
            if center is not None:
                return center, why
            last_why = why
        if time.time() >= deadline:
            return None, f"等 {timeout_s:.0f}s 也没找到目标（{last_why}）"
        await asyncio.sleep(0.6)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--macro", required=True)
    ap.add_argument("--serial", default="89R8ZXNBGQIRJRLF")
    ap.add_argument("--package", default=None, help="launch_app 步要启动的包名")
    ap.add_argument("--dry-run", action="store_true", help="只解析，不动手机")
    ap.add_argument(
        "--continue-on-miss",
        action="store_true",
        help="元素找不到时跳过这步继续跑（而不是中止整个宏）",
    )
    ap.add_argument(
        "--fuzzy",
        action="store_true",
        help="允许子串降级匹配（**危险**：会打到错误元素上，默认关）",
    )
    ap.add_argument(
        "--use-position",
        action="store_true",
        default=True,
        help="id 找不到时按「录制时的位置」降级（默认开：实测位置比 id 稳）",
    )
    args = ap.parse_args()

    macro = json.loads(Path(args.macro).read_text(encoding="utf-8"))
    steps = macro["steps"]
    ctx = build_ctx(args.serial)
    controller = UnifiedMobileController(ctx)
    screen_w = ctx.device.device_width
    screen_h = ctx.device.device_height

    print("=" * 74)
    print("回放宏:", macro.get("name"), "| 来源执行:", macro.get("source_exec_id"))
    print("总步数:", len(steps), "| 模式:", "DRY-RUN" if args.dry_run else "真机")
    print("=" * 74)

    executed = skipped = 0
    failed_at = None
    t_start = time.perf_counter()
    app_package = macro.get("app_package")
    app_established = False

    for i, step in enumerate(steps, 1):
        action = step.get("action")
        args_d = step.get("args") or {}
        selector = step.get("stable_selector")
        need_selector = action in ("tap", "focus_and_input_text", "long_press_on")

        if need_selector and not selector:
            skipped += 1
            print(
                "  %2d. %-22s ⏭  跳过：没有稳定选择器（录制时是按 index/OCR 点的，跨运行不可靠）"
                % (i, action)
            )
            continue

        if args.dry_run:
            print("  %2d. %-22s sel=%-44s args=%s" % (i, action, selector, args_d))
            continue

        t0 = time.perf_counter()
        try:
            # ---- 前置条件：这一步是在哪一屏上做的 ----
            elements = await controller.get_ui_elements()
            ok_screen, why_screen = check_precondition(
                step,
                elements,
                app_package=app_package,
                app_established=app_established,
            )
            if not ok_screen:
                print("  %2d. %-22s 🛑 中止：%s" % (i, action, why_screen))
                failed_at = (i, action, why_screen)
                break

            if action == "launch_app":
                pkg = args.package or app_package or args_d.get("app_name")
                ok = await controller.launch_app(pkg)
                ready, ready_why = await wait_for_app(controller, pkg)
                app_established = bool(ok and ready)
                detail = f"launch {pkg} -> {ok}；{ready_why}"
                if not ready:
                    print("  %2d. %-22s ❌ %s" % (i, action, detail))
                    failed_at = (i, action, ready_why)
                    break
            elif action == "swipe_percentages":
                ok = True
                await controller.swipe_percentage(
                    start_x_percent=args_d.get("start_x_percent", 50),
                    start_y_percent=args_d.get("start_y_percent", 70),
                    end_x_percent=args_d.get("end_x_percent", 50),
                    end_y_percent=args_d.get("end_y_percent", 35),
                )
                detail = "swipe"
            elif action in ("tap", "long_press_on"):
                center, why = await resolve_step_target(
                    controller,
                    step,
                    fuzzy=args.fuzzy,
                    use_position=args.use_position,
                    app_package=app_package,
                )
                if center is None:
                    print("  %2d. %-22s ⚠️  跳过：%s" % (i, action, why))
                    skipped += 1
                    if not args.continue_on_miss:
                        failed_at = (i, action, why)
                        break
                    continue
                if not (0 <= center[0] < screen_w and 0 <= center[1] < screen_h):
                    msg = f"坐标 ({center[0]},{center[1]}) 落在屏外（{screen_w}x{screen_h}）"
                    print("  %2d. %-22s ❌ %s" % (i, action, msg))
                    failed_at = (i, action, msg)
                    break
                res = await controller.tap_at(x=center[0], y=center[1])
                ok = res.error is None
                detail = "%s -> tap(%d,%d) %s" % (why, center[0], center[1], ok)
            elif action == "focus_and_input_text":
                center, why = await resolve_step_target(
                    controller,
                    step,
                    fuzzy=args.fuzzy,
                    use_position=args.use_position,
                    app_package=app_package,
                )
                if center is None:
                    print("  %2d. %-22s ⚠️  跳过：%s" % (i, action, why))
                    skipped += 1
                    if not args.continue_on_miss:
                        failed_at = (i, action, why)
                        break
                    continue
                if not (0 <= center[0] < screen_w and 0 <= center[1] < screen_h):
                    msg = f"坐标 ({center[0]},{center[1]}) 落在屏外（{screen_w}x{screen_h}）"
                    print("  %2d. %-22s ❌ %s" % (i, action, msg))
                    failed_at = (i, action, why)
                    break
                await controller.tap_at(x=center[0], y=center[1])
                await asyncio.sleep(0.3)
                typed = await controller.type_text(args_d.get("text") or "")
                ok = typed
                detail = "%s -> tap(%d,%d) + type(%d 字) %s" % (
                    why,
                    center[0],
                    center[1],
                    len(args_d.get("text") or ""),
                    typed,
                )
            else:
                skipped += 1
                print("  %2d. %-22s ⏭  未支持的动作" % (i, action))
                continue
        except Exception as e:  # noqa: BLE001
            print("  %2d. %-22s ❌ 异常 %s: %s" % (i, action, type(e).__name__, e))
            failed_at = (i, action, "%s: %s" % (type(e).__name__, e))
            break

        executed += 1
        print(
            "  %2d. %-22s %s %s  (%.0fms)"
            % (i, action, "✅" if ok else "⚠️", detail, (time.perf_counter() - t0) * 1000)
        )
        try:
            print("        结构指纹:", structural_digest(elements)[:16])
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(0.6)

    total = time.perf_counter() - t_start
    print("-" * 74)
    print(
        "执行 %d 步 / 跳过 %d 步 / 耗时 %.1fs%s"
        % (executed, skipped, total, "" if not failed_at else "  ← 第 %d 步失败" % failed_at[0])
    )
    if failed_at:
        print("失败原因:", failed_at[2])
    return 0 if not failed_at else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
