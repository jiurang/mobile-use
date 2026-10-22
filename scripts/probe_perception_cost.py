"""**感知段（Contextor）的逐项耗时实测** + **沉降等待能不能去掉**的判定实验。

问三个问题
----------

1. **感知段的钱花在哪？** ``contextor`` 节点实测中位 3,746ms（27 次，合计 73.1s），
   其中沉降等待 51.0s。剩下 22.1s 里，每一项各占多少？

   代码路径（``agents/contextor/contextor.py``）是**串行**的::

       沉降等待 → [抓屏 ∥ dumpsys power] → dumpsys window | grep mCurrentFocus
                → date → normalize_hierarchy → build_screen_elements

   ⚠️ 注意后面两个：``get_current_foreground_package_async`` 和 ``get_device_date``
   的函数体是**同步的** ``device.shell(...)``（名字带 async 但并没有），
   它们**阻塞事件循环**，而且排在抓屏**之后** ⇒ 全额计入关键路径。

2. **它们能不能并发掉？** 三个 adb 往返互不依赖，能不能和抓屏重叠？
   而 ``dumpsys window`` 那个往返能不能**根本不做** ——
   抓回来的层级树里**每个元素都带 ``package`` 字段**（``_parse_hierarchy_xml_to_elements``），
   前台包名就在手里。

3. **3 秒沉降下限能不能去掉？** 这是感知段最大的单笔（51.0s = 墙钟 7.1%）。
   它的存在理由是「控件延迟挂载，抓早了拿到半成品」。
   但 ``dump_hierarchy`` 本身要 ~1.2s（uiautomator 自带 waitForIdle），
   所以「抓早」到底会拿到什么，**必须实测**：导航动作之后连打几次 dump，
   看**第一次** dump 回来时新页面的控件在不在。

跑法（要真机在线）::

    uv run python scripts/probe_perception_cost.py
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("MOBILE_USE_TELEMETRY_ENABLED", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

from adbutils import AdbClient  # noqa: E402

from minitap.mobile_use.clients.ui_automator_client import (  # noqa: E402
    UIAutomatorClient,
    _parse_hierarchy_xml_to_elements,
)
from minitap.mobile_use.context import (  # noqa: E402
    DeviceContext,
    DevicePlatform,
    MobileUseContext,
)
from minitap.mobile_use.controllers.android_controller import (  # noqa: E402
    AndroidDeviceController,
)
from minitap.mobile_use.controllers.platform_specific_commands_controller import (  # noqa: E402
    get_current_foreground_package_async,
    get_device_date,
)
from minitap.mobile_use.services.ocr import OcrConfig, create_ocr_engine  # noqa: E402
from minitap.mobile_use.utils.screen_elements import build_screen_elements  # noqa: E402
from minitap.mobile_use.utils.ui_hierarchy_view import normalize_hierarchy  # noqa: E402

SERIAL = "89R8ZXNBGQIRJRLF"
ROUNDS = 8


def _med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def _fmt(name: str, xs: list[float], note: str = "") -> str:
    return (
        f"  {name:<38} 中位 {_med(xs):>7.0f} ms   "
        f"最小 {min(xs):>6.0f}   最大 {max(xs):>6.0f}  {note}"
    )


def _make_ctx(adb: AdbClient, ui: UIAutomatorClient) -> MobileUseContext:
    """最小 context。这个探针一条 LLM 都不调，但 ``llm_config`` 是必填字段。"""
    from minitap.mobile_use.config import get_default_llm_config, initialize_llm_config

    try:
        llm_config = initialize_llm_config()
    except Exception:
        llm_config = get_default_llm_config()

    return MobileUseContext(
        trace_id="probe_perception",
        device=DeviceContext(
            host_platform="WINDOWS",
            mobile_platform=DevicePlatform.ANDROID,
            device_id=SERIAL,
            device_width=1220,
            device_height=2712,
        ),
        llm_config=llm_config,
        adb_client=adb,
        ui_adb_client=ui,
    )


def _fingerprint(elements: list[dict]) -> str:
    """树内容指纹：只看 resource-id + text + class 的集合。

    只看**数量**不够 —— 一棵半成品的树可能元素数一样但内容不同（控件还没挂上）。
    """
    items = sorted(
        f"{e.get('resource-id', '')}|{e.get('text', '')}|{e.get('class', '')}"
        for e in elements
    )
    return f"{len(items)}:" + "|".join(items)


# ---------------------------------------------------------------------------
# 1. 逐项成本
# ---------------------------------------------------------------------------


async def part1_itemized(ctrl, ctx, ui, engine) -> dict:
    print("=" * 96)
    print("1) 感知段每一项的真实成本")
    print("=" * 96)

    shot: list[float] = []
    dump: list[float] = []
    ocr: list[float] = []
    parse: list[float] = []
    capture: list[float] = []
    capture_noocr: list[float] = []
    power: list[float] = []
    foreground: list[float] = []
    devdate: list[float] = []
    build: list[float] = []

    for _ in range(ROUNDS):
        dev = ui._ensure_connected()

        t0 = time.perf_counter()
        dev.screenshot()
        shot.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        xml = dev.dump_hierarchy(compressed=True)
        dump.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        _parse_hierarchy_xml_to_elements(xml)
        parse.append((time.perf_counter() - t0) * 1000)

        # OCR 单独量（拿一张现图的 base64）
        from minitap.mobile_use.clients.ui_automator_client import _pil_to_base64

        img = dev.screenshot()
        t0 = time.perf_counter()
        engine.recognize(_pil_to_base64(img))
        ocr.append((time.perf_counter() - t0) * 1000)

        # 走生产路径：整段 get_screen_data（dump ∥ screenshot，OCR 内嵌）
        t0 = time.perf_counter()
        data = await ctrl.get_screen_data(on_screenshot=lambda b: engine.recognize(b))
        capture.append((time.perf_counter() - t0) * 1000)

        # 不带 OCR 的整段（隔离 OCR 的净贡献）
        t0 = time.perf_counter()
        await ctrl.get_screen_data()
        capture_noocr.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        await ctrl.get_screen_power_state()
        power.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        await get_current_foreground_package_async(ctx)
        foreground.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        get_device_date(ctx)
        devdate.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        rows = normalize_hierarchy(data.elements)
        build_screen_elements(
            tree_rows=rows,
            ocr_results=None,
            foreground_package="com.android.settings",
        )
        build.append((time.perf_counter() - t0) * 1000)

    print()
    print(_fmt("截图 screenshot()", shot))
    print(_fmt("层级树 dump_hierarchy()", dump, "← 感知段最大单笔"))
    print(_fmt("  └ XML 解析（CPU）", parse))
    print(_fmt("OCR 推理（CPU）", ocr))
    print(_fmt("抓屏整段 get_screen_data（含 OCR）", capture, "← 生产路径"))
    print(_fmt("抓屏整段（不含 OCR）", capture_noocr, "← 对照"))
    print(_fmt("dumpsys power（已并发）", power))
    print(_fmt("dumpsys window | grep mCurrentFocus", foreground, "🔴 串行阻塞"))
    print(_fmt("date", devdate, "🔴 串行阻塞"))
    print(_fmt("建元素表 normalize+build（CPU）", build))

    return {
        "screenshot": shot,
        "dump": dump,
        "parse": parse,
        "ocr": ocr,
        "capture": capture,
        "capture_noocr": capture_noocr,
        "power": power,
        "foreground": foreground,
        "device_date": devdate,
        "build": build,
    }


# ---------------------------------------------------------------------------
# 2. 关键路径：三种排布
# ---------------------------------------------------------------------------


def _dominant_package(elements: list[dict]) -> str | None:
    """从层级树推前台包名。

    判据：元素最多、且不是系统 UI 的那个包。
    这只是用来决定「要不要保留 systemui 行」，精度要求不高
    （见 ``utils/screen_elements.build_screen_elements`` 的说明）。
    """
    counts: dict[str, int] = {}
    for e in elements:
        pkg = e.get("package")
        if pkg:
            counts[pkg] = counts.get(pkg, 0) + 1
    if not counts:
        return None
    systemui = {"com.android.systemui", "miui.systemui.plugin"}
    app_counts = {k: v for k, v in counts.items() if k not in systemui}
    pool = app_counts or counts
    return max(pool, key=lambda k: pool[k])


async def part2_arrangements(ctrl, ctx, ui, engine) -> dict:
    print()
    print("=" * 96)
    print("2) 感知段关键路径：三种排布（端到端，不含沉降等待）")
    print("=" * 96)

    async def ocr_cb(b64: str) -> None:
        engine.recognize(b64)

    # A —— 现状：抓屏（∥power）→ foreground → date → 建表
    a: list[float] = []
    # B —— 三个 adb 全部与抓屏并发
    b: list[float] = []
    # C —— foreground 从树里推、date 用本机时钟，power 并发
    c: list[float] = []

    for _ in range(ROUNDS):
        # --- A 现状 ---
        t0 = time.perf_counter()
        power_task = asyncio.create_task(ctrl.get_screen_power_state())
        await asyncio.sleep(0)
        data = await ctrl.get_screen_data(on_screenshot=ocr_cb)
        await power_task
        await get_current_foreground_package_async(ctx)
        get_device_date(ctx)
        rows = normalize_hierarchy(data.elements)
        build_screen_elements(
            tree_rows=rows,
            ocr_results=None,
            foreground_package="com.android.settings",
        )
        a.append((time.perf_counter() - t0) * 1000)

        # --- B 全并发 ---
        t0 = time.perf_counter()
        power_task = asyncio.create_task(ctrl.get_screen_power_state())
        fg_task = asyncio.create_task(get_current_foreground_package_async(ctx))
        date_task = asyncio.to_thread(get_device_date, ctx)
        await asyncio.sleep(0)
        data = await ctrl.get_screen_data(on_screenshot=ocr_cb)
        await power_task
        await fg_task
        await date_task
        rows = normalize_hierarchy(data.elements)
        build_screen_elements(
            tree_rows=rows,
            ocr_results=None,
            foreground_package="com.android.settings",
        )
        b.append((time.perf_counter() - t0) * 1000)

        # --- C 免 adb ---
        t0 = time.perf_counter()
        power_task = asyncio.create_task(ctrl.get_screen_power_state())
        await asyncio.sleep(0)
        data = await ctrl.get_screen_data(on_screenshot=ocr_cb)
        await power_task
        fg = _dominant_package(data.elements)
        _ = fg  # 用本机时钟代替 device.shell("date")
        rows = normalize_hierarchy(data.elements)
        build_screen_elements(
            tree_rows=rows,
            ocr_results=None,
            foreground_package=fg,
        )
        c.append((time.perf_counter() - t0) * 1000)

    print()
    print(_fmt("A 现状（串行 fg + date）", a))
    print(_fmt("B 三个 adb 全部并发", b))
    print(_fmt("C 免 adb（fg 从树推）", c))
    base = _med(a)
    print()
    for name, xs in (("B", b), ("C", c)):
        m = _med(xs)
        print(f"  {name} 相对现状：**{(base - m) / base * 100:+.1f}%**（省 {base - m:+.0f} ms/轮）")

    return {"A_current": a, "B_concurrent": b, "C_no_adb": c}


async def part1b_capture_internals(ctrl, ctx, ui, engine) -> dict:
    """把抓屏这一段拆到毫秒级。

    ``get_screen_data`` 里其实有 4 件事叠在一起：截图、**PNG 编码**、OCR、层级树 dump。
    之前只知道整段 814ms，不知道谁在占。
    """
    print()
    print("=" * 96)
    print("1b) 抓屏内部拆分（截图 / PNG 编码 / OCR / 层级树）")
    print("=" * 96)

    from minitap.mobile_use.clients.ui_automator_client import _pil_to_base64
    from minitap.mobile_use.utils.image_compress import compress_screenshot_base64

    dev = ui._ensure_connected()
    t_screenshot: list[float] = []
    t_png: list[float] = []
    t_ocr: list[float] = []
    t_dump: list[float] = []
    t_compress: list[float] = []

    b64 = None
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        img = dev.screenshot()
        t_screenshot.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        b64 = _pil_to_base64(img)
        t_png.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        engine.recognize(b64)
        t_ocr.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        dev.dump_hierarchy(compressed=True)
        t_dump.append((time.perf_counter() - t0) * 1000)

        # executor 的「二问」那条路：把 base64 PNG 解码 → 缩放 → 重编码成 JPEG
        t0 = time.perf_counter()
        compress_screenshot_base64(b64)
        t_compress.append((time.perf_counter() - t0) * 1000)

    print()
    print(_fmt("① device.screenshot()", t_screenshot))
    print(_fmt("② _pil_to_base64（PNG 编码）", t_png, "🔴 全分辨率 PNG"))
    print(_fmt("③ engine.recognize（OCR 推理）", t_ocr))
    print(_fmt("④ dump_hierarchy（层级树）", t_dump))
    print(_fmt("⑤ compress_screenshot_base64（二问用）", t_compress, "＝解码PNG+缩放+JPEG重编码"))

    serial = _med(t_screenshot) + _med(t_png) + _med(t_ocr)
    print()
    print(f"  ⇒ 截图这条链（①②③ 串行）= **{serial:.0f} ms**")
    print(
        f"     ④ 层级树与之并发（{_med(t_dump):.0f} ms）"
        f"⇒ 抓屏关键路径 ≈ {max(serial, _med(t_dump)):.0f} ms"
    )
    print(
        f"     其中 **PNG 编码 ② 占 {_med(t_png) / serial * 100:.0f}%**"
        f" —— 而它只为「按需截图」那一条路服务"
    )

    return {
        "screenshot": t_screenshot,
        "png_encode": t_png,
        "ocr": t_ocr,
        "dump": t_dump,
        "compress": t_compress,
    }


# ---------------------------------------------------------------------------
# 3. 沉降等待能不能去掉
# ---------------------------------------------------------------------------


def _center(el: dict) -> tuple[int, int] | None:
    b = el.get("bounds", "")
    try:
        (x1, y1), (x2, y2) = [
            tuple(int(v) for v in seg.split(",")) for seg in b.strip("[]").split("][")
        ]
    except Exception:
        return None
    return ((x1 + x2) // 2, (y1 + y2) // 2)


async def _settle_trial(dev, ui, action: str, label: str) -> dict:
    """跑一次「动作 → 连续 dump」，看最终内容在第几个 dump 出现。"""
    t_action = time.perf_counter()
    if action == "tap_wlan":
        dev.shell(f"input tap {action}")
    else:
        dev.shell(action)

    samples = []
    for i in range(5):
        t_start = time.perf_counter()
        xml = dev.dump_hierarchy(compressed=True)
        t_end = time.perf_counter()
        els = _parse_hierarchy_xml_to_elements(xml)
        titles = [
            (e.get("text") or "").strip()
            for e in els
            if (e.get("text") or "").strip()
        ]
        samples.append(
            {
                "i": i,
                "start_ms": (t_start - t_action) * 1000,
                "end_ms": (t_end - t_action) * 1000,
                "n_elements": len(els),
                "has_wlan_title": any(t.upper() in ("WLAN", "WI-FI", "WIFI") for t in titles),
                "fp": _fingerprint(els),
            }
        )

    final_fp = samples[-1]["fp"]
    first_full = next((s["i"] for s in samples if s["fp"] == final_fp), None)
    ready_ms = samples[first_full]["end_ms"] if first_full is not None else None
    print(
        f"    {label:<28} 最终内容出现在 dump#{first_full}"
        f"（回来于 {ready_ms:.0f}ms）   元素数序列 {[s['n_elements'] for s in samples]}"
        if first_full is not None
        else f"    {label:<28} ⚠ 5 次 dump 都没拿到最终内容"
    )
    return {"label": label, "first_full": first_full, "ready_ms": ready_ms, "samples": samples}


async def part3_settle(ctrl, ui) -> dict:
    print()
    print("=" * 96)
    print("3) 沉降下限能不能去掉：动作后连打 5 次 dump，看最终内容第几个 dump 出现")
    print("=" * 96)

    dev = ui._ensure_connected()
    dev.shell("am start -a android.settings.SETTINGS")
    await asyncio.sleep(2.5)

    # 找到 WLAN 行的中心点
    data = await ctrl.get_screen_data()
    target = None
    for e in data.elements:
        if (e.get("text") or "").strip().upper() in ("WLAN", "WIFI", "WI-FI"):
            target = e
            break
    if target is None:
        print("  ⚠ 找不到 WLAN 行，跳过")
        return {}
    pt = _center(target)
    print(f"  目标行: {target.get('text')!r} bounds={target.get('bounds')} → 点击 {pt}")
    print()

    trials = []
    for rep in range(2):
        # --- 场景 A：页面导航（点进 WLAN）---
        dev.shell("am start -a android.settings.SETTINGS")
        await asyncio.sleep(2.0)
        trials.append(
            await _settle_trial(
                dev, ui, f"input tap {pt[0]} {pt[1]}", f"A 页面导航 第{rep + 1}次"
            )
        )

        # --- 场景 B：列表滚动（对应文档里「控件延迟挂载」的场景）---
        await asyncio.sleep(1.0)
        trials.append(
            await _settle_trial(
                dev, ui, "input swipe 600 2000 600 700 400", f"B 列表滚动 第{rep + 1}次"
            )
        )

        # --- 场景 C：返回上一页 ---
        await asyncio.sleep(1.0)
        trials.append(await _settle_trial(dev, ui, "input keyevent 4", f"C 返回键 第{rep + 1}次"))
        await asyncio.sleep(1.0)

    print()
    ok = [t["ready_ms"] for t in trials if t.get("ready_ms") is not None]
    if ok:
        print(f"  ⇒ {len(ok)}/{len(trials)} 次试验里，最终内容在动作后 "
              f"**{min(ok):.0f} ~ {max(ok):.0f} ms** 就已经拿到（最坏 {max(ok):.0f}ms）")
        print(f"  ⇒ 而 3 秒下限让 dump 最早也要在 3000ms 才发起 ⇒ "
              f"**每次多等 {3000 - max(ok):.0f} ~ {3000 - min(ok):.0f} ms**")
    else:
        print("  ⚠ 没有一次试验拿到最终内容 —— 5 次 dump 不够，需要更多采样")

    return {"trials": trials}



async def main() -> int:
    adb = AdbClient(host="localhost", port=5037)
    ui = UIAutomatorClient(device_id=SERIAL)
    ctrl = AndroidDeviceController(
        device_id=SERIAL,
        adb_client=adb,
        ui_adb_client=ui,
        device_width=1220,
        device_height=2712,
    )
    ctx = _make_ctx(adb, ui)
    engine = create_ocr_engine(OcrConfig())  # 预热，模型加载不计入

    print(f"探针: 感知段逐项耗时 + 沉降等待可替代性   设备 {SERIAL}")
    print()

    items = await part1_itemized(ctrl, ctx, ui, engine)
    internals = await part1b_capture_internals(ctrl, ctx, ui, engine)
    arr = await part2_arrangements(ctrl, ctx, ui, engine)
    settle = await part3_settle(ctrl, ui)

    out = ROOT / "probe_out" / "perception_cost.json"
    out.write_text(
        json.dumps(
            {
                "rounds": ROUNDS,
                "items": {k: [round(x, 1) for x in v] for k, v in items.items()},
                "items_median": {k: round(_med(v), 1) for k, v in items.items()},
                "capture_internals": {k: [round(x, 1) for x in v] for k, v in internals.items()},
                "capture_internals_median": {
                    k: round(_med(v), 1) for k, v in internals.items()
                },
                "arrangements": {k: [round(x, 1) for x in v] for k, v in arr.items()},
                "arrangements_median": {k: round(_med(v), 1) for k, v in arr.items()},
                "settle": settle,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n=> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
