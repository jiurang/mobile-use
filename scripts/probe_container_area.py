"""真机确认：中心点判据的 max_area_fraction 该定多少。

要划开的是「真控件」与「纯容器」。难点：一个真控件也可能很大
（底部 Tab 栏、整块卡片），而容器也可能不大。

做法：在多屏真机上把节点按面积排序，看可操作节点与不可操作容器的
面积分布重叠情况，取一个既能容下大控件、又能排除大容器的分界。
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minitap"))

from minitap.mobile_use.clients.ui_automator_client import (  # noqa: E402
    UIAutomatorClient,
    _parse_hierarchy_xml_to_elements,
)
from minitap.mobile_use.utils.ui_hierarchy import get_bounds_for_element  # noqa: E402

DEVICE = "89R8ZXNBGQIRJRLF"
W, H = 1220, 2712
SCREEN = W * H

SCREENS = {
    "home": ["shell", "input", "keyevent", "3"],
    "settings": ["shell", "am", "start", "-a", "android.settings.SETTINGS"],
    "recents": ["shell", "input", "keyevent", "187"],
    "notif": ["shell", "cmd", "statusbar", "expand-notifications"],
}


def truthy(v) -> bool:
    return v is True or (isinstance(v, str) and v.strip().lower() == "true")


def capture(name: str) -> list[dict]:
    subprocess.run(["adb", "-s", DEVICE, *SCREENS[name]], capture_output=True, timeout=30)
    time.sleep(2.0)
    return _parse_hierarchy_xml_to_elements(
        UIAutomatorClient(device_id=DEVICE).get_hierarchy()
    )


def main() -> None:
    all_actionable: list[tuple[float, str]] = []  # (面积占比, label)
    all_container: list[tuple[float, str]] = []

    for name in SCREENS:
        try:
            elements = capture(name)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: capture failed {e}")
            continue
        if not elements:
            continue

        n_act = n_con = 0
        for e in elements:
            b = get_bounds_for_element(e)
            if not b:
                continue
            frac = (b.width * b.height) / SCREEN
            label = (
                (e.get("text") or "").strip()
                or (e.get("content-desc") or "").strip()
                or e.get("resource-id")
                or (e.get("class") or "").rsplit(".", 1)[-1]
                or "?"
            )
            if not truthy(e.get("visible-to-user", True)):
                continue
            act = (
                truthy(e.get("clickable"))
                or truthy(e.get("long-clickable"))
                or truthy(e.get("focusable"))
                or truthy(e.get("checkable"))
            )
            if act:
                all_actionable.append((frac, f"{name}/{label[:24]}"))
                n_act += 1
            else:
                all_container.append((frac, f"{name}/{label[:24]}"))
                n_con += 1
        print(f"{name:9s} nodes={len(elements):4d} actionable={n_act:3d} other={n_con:3d}")

    all_actionable.sort(reverse=True)
    all_container.sort(reverse=True)

    print(f"\n可操作节点 {len(all_actionable)} 个，其它节点 {len(all_container)} 个")
    print("\n--- 面积最大的 10 个「可操作」节点（真控件可以很大）---")
    for frac, lab in all_actionable[:10]:
        print(f"  {frac * 100:6.2f}%  {lab}")
    print("\n--- 面积最大的 10 个「不可操作」节点（这些是容器，不能当点击目标）---")
    for frac, lab in all_container[:10]:
        print(f"  {frac * 100:6.2f}%  {lab}")

    print("\n--- 门限扫描：阈值以下才算「真控件」 ---")
    print(f"{'max_frac':>9s} {'控件保留':>10s} {'容器混入':>10s}  说明")
    for thr in [0.50, 0.30, 0.20, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02]:
        kept = sum(1 for f, _ in all_actionable if f < thr)
        leak = sum(1 for f, _ in all_container if f < thr)
        verdict = ""
        if kept == len(all_actionable) and leak == 0:
            verdict = "<= 无漏、无混"
        print(
            f"{thr:9.2f} {kept:6d}/{len(all_actionable):<3d} "
            f"{leak:6d}/{len(all_container):<3d}  {verdict}"
        )

    # 最小的那个大容器，给出真实分界
    big_containers = [f for f, _ in all_container if f >= 0.05]
    if big_containers:
        print(f"\n真实最小「大容器」面积占比: {min(big_containers) * 100:.2f}%")
    big_controls = [f for f, _ in all_actionable if f >= 0.05]
    if big_controls:
        print(f"真实最大「可操作但很大」面积占比: {max(big_controls) * 100:.2f}%")


if __name__ == "__main__":
    main()
