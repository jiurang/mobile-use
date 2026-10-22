"""**验证感知段的三处改动**（2026-10-22）。

改了三件事，每一件都要用真机对一遍：

1. **前台包名从树里推**（`utils/screen_elements.foreground_package_from_tree`）
   —— 原来是一次串行的 ``dumpsys window | grep mCurrentFocus``（实测 127ms）。
   ⇒ **逐屏和 dumpsys 对账**，不一致就是回归。

2. **沉降算法**：下限 3000 → 200 + 树稳定确认。
   ⇒ 量新的感知段端到端耗时，和旧账本（contextor 中位 3,746ms / 合计 73.1s）比。

3. **树级重抓**（``controller.get_hierarchy_only``）
   ⇒ 它必须**不截图、不跑 OCR**，且比整段 get_screen_data 便宜得多。

跑法（要真机在线）::

    uv run python scripts/verify_perception_changes.py
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

from minitap.mobile_use.clients.ui_automator_client import UIAutomatorClient  # noqa: E402
from minitap.mobile_use.context import DeviceContext, DevicePlatform, MobileUseContext  # noqa: E402
from minitap.mobile_use.controllers.android_controller import (  # noqa: E402
    AndroidDeviceController,
)
from minitap.mobile_use.utils.screen_elements import foreground_package_from_tree  # noqa: E402
from minitap.mobile_use.utils.screen_stability import tree_fingerprint  # noqa: E402

SERIAL = "89R8ZXNBGQIRJRLF"
ROUNDS = 6


def _med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def _dumpsys_focus(dev) -> str | None:
    out = str(dev.shell("dumpsys window | grep mCurrentFocus"))
    if "mCurrentFocus=" not in out:
        return None
    segment = out.split("mCurrentFocus=")[-1]
    for token in segment.split():
        if "." in token and not token.startswith("Window"):
            return token.split("/")[0].rstrip("}")
    return None


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
    dev = ui._ensure_connected()

    # ---- 1. 前台包名对账 ----
    print("=" * 90)
    print("1) 前台包名：从树里推 vs dumpsys window | grep mCurrentFocus")
    print("=" * 90)

    scenes = [
        ("设置首页", ["am start -a android.settings.SETTINGS"]),
        ("WLAN 页", ["am start -a android.settings.SETTINGS", "input tap 300 1179"]),
        ("通知栏", ["cmd statusbar expand-notifications"]),
        ("回到设置", ["input keyevent 4", "cmd statusbar collapse"]),
    ]

    mismatches = 0
    checked = 0
    for name, cmds in scenes:
        for cmd in cmds:
            dev.shell(cmd)
        await asyncio.sleep(1.8)
        data = await ctrl.get_screen_data()
        from_tree = foreground_package_from_tree(data.elements)
        from_dumpsys = _dumpsys_focus(dev)
        same = from_tree == from_dumpsys
        checked += 1
        if not same:
            mismatches += 1
        print(
            f"  {name:<10} 树里推={str(from_tree):<32} dumpsys={str(from_dumpsys):<32} "
            f"{'✅ 一致' if same else '❌ 不一致'}"
        )
    print(f"\n  ⇒ {checked - mismatches}/{checked} 一致")

    # ---- 2. 树级重抓的代价 ----
    print()
    print("=" * 90)
    print("2) 树级重抓 get_hierarchy_only 的代价（必须远小于整段 get_screen_data）")
    print("=" * 90)

    tree_only: list[float] = []
    full: list[float] = []
    from minitap.mobile_use.services.ocr import OcrConfig, create_ocr_engine

    engine = create_ocr_engine(OcrConfig())

    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        await ctrl.get_hierarchy_only()
        tree_only.append((time.perf_counter() - t0) * 1000)

        # ⚠️ 必须带 OCR 回调才是**生产路径**：Contextor 每次抓屏都跑 OCR。
        # 不带的话整段只有 271ms（截图被 dump 的窗口藏住了），那不是真实成本。
        t0 = time.perf_counter()
        await ctrl.get_screen_data(on_screenshot=lambda b: engine.recognize(b))
        full.append((time.perf_counter() - t0) * 1000)

    print(f"  树级重抓（只 dump）        中位 {_med(tree_only):>6.0f} ms")
    print(f"  整段 get_screen_data+OCR   中位 {_med(full):>6.0f} ms")
    print(f"  ⇒ 便宜了 {_med(full) - _med(tree_only):.0f} ms/次（省掉截图 + PNG 编码 + OCR）")

    # ---- 3. 指纹能不能区分「变了」和「没变」 ----
    print()
    print("=" * 90)
    print("3) 树指纹：同屏两次采样相同吗？动一下之后变了吗？")
    print("=" * 90)

    dev.shell("am start -a android.settings.SETTINGS")
    await asyncio.sleep(1.8)
    a = tree_fingerprint(await ctrl.get_hierarchy_only())
    b = tree_fingerprint(await ctrl.get_hierarchy_only())
    print(f"  静止屏两次采样: {'✅ 相同' if a == b else '❌ 不同'}  ({a[:12]} / {b[:12]})")

    dev.shell("input swipe 600 2000 600 900 400")
    await asyncio.sleep(0.2)
    c = tree_fingerprint(await ctrl.get_hierarchy_only())
    await asyncio.sleep(1.2)
    d = tree_fingerprint(await ctrl.get_hierarchy_only())
    print(f"  滚动后立即采样 vs 停稳后: {'✅ 不同（抓到了变化）' if c != d else '❌ 相同（漏判）'}")

    # ---- 4. 新的感知段端到端耗时 ----
    print()
    print("=" * 90)
    print("4) 新的感知段端到端（含 200ms 下限 + 稳定确认）")
    print("=" * 90)

    from minitap.mobile_use.agents.contextor.contextor import ContextorNode

    node = ContextorNode.__new__(ContextorNode)
    node.ctx = MobileUseContext(
        trace_id="verify",
        device=DeviceContext(
            host_platform="WINDOWS",
            mobile_platform=DevicePlatform.ANDROID,
            device_id=SERIAL,
            device_width=1220,
            device_height=2712,
        ),
        llm_config=_dummy_llm_config(),
        adb_client=adb,
        ui_adb_client=ui,
    )

    class _State:
        last_action_at = time.time()

        async def asanitize_update(self, *, ctx, update, agent):
            return update

    rounds: list[float] = []
    stability_calls: list[int] = []
    original = ctrl.get_hierarchy_only
    counter = {"n": 0}

    async def counting_hierarchy_only():
        counter["n"] += 1
        return await original()

    ctrl.get_hierarchy_only = counting_hierarchy_only  # type: ignore[method-assign]

    for _ in range(ROUNDS):
        counter["n"] = 0
        t0 = time.perf_counter()
        await node._wait_for_settle(_State())
        await node._confirm_tree_stable(ctrl, _State())
        rounds.append((time.perf_counter() - t0) * 1000)
        stability_calls.append(counter["n"])

    print(f"  沉降 + 稳定确认  中位 {_med(rounds):.0f} ms   逐次 {[round(r) for r in rounds]}")
    print(f"  每次的树级 dump 次数 {stability_calls}")
    print()
    print("  对照旧账本：contextor 节点中位 3,746ms（其中沉降等待 3,000ms）")
    print(f"  ⇒ 沉降部分从 3,000ms 降到 ~{_med(rounds):.0f}ms")

    out = ROOT / "probe_out" / "perception_changes_verify.json"
    out.write_text(
        json.dumps(
            {
                "foreground_mismatches": mismatches,
                "foreground_checked": checked,
                "tree_only_ms": [round(x) for x in tree_only],
                "full_capture_ms": [round(x) for x in full],
                "settle_plus_stability_ms": [round(x) for x in rounds],
                "stability_dumps": stability_calls,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n=> {out}")
    return 0 if mismatches == 0 else 1


def _dummy_llm_config():
    from minitap.mobile_use.config import get_default_llm_config, initialize_llm_config

    try:
        return initialize_llm_config()
    except Exception:
        return get_default_llm_config()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
