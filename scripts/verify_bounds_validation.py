"""在真机数据上验证坐标校验判据 —— 这是唯一可信的评判方式。

前几轮在 5 元素的小假 dump 上调判据，调出 4 条测试互相打架：放宽尺寸漂移
就会放进坏坐标。那是因为假 dump 缺了真机上最要命的东西：**大量互相嵌套的
大容器**。真机实测 31 个节点覆盖 >=95% 屏幕，另有面积 31.6% 屏幕的中间容器。

所以这里不再用假数据，直接连真机跑多屏，输出判据的混淆矩阵：
    真负例 good  = 层级树原值坐标（必须全部接受）
    真阳例 bad   = 模拟模型从缩放截图目测的坐标（必须全部拒绝）

跑法：
    .venv\\Scripts\\python.exe scripts\\verify_bounds_validation.py
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
from minitap.mobile_use.tools.bounds_validation import (  # noqa: E402
    coordinates_look_like_hierarchy_derived_from_elements,
)
from minitap.mobile_use.tools.types import Target  # noqa: E402
from minitap.mobile_use.utils.ui_hierarchy import get_bounds_for_element  # noqa: E402

DEVICE = "89R8ZXNBGQIRJRLF"

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
    tot_good = tot_good_ok = 0
    tot_bad = tot_bad_caught = 0
    failures: list[str] = []
    leaks: list[str] = []

    for name in SCREENS:
        try:
            elements = capture(name)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: capture failed: {e}")
            continue
        if not elements:
            continue

        # 真负例：所有可见可操作节点的原始 bounds
        cases = []
        for e in elements:
            if not truthy(e.get("visible-to-user", True)):
                continue
            b = get_bounds_for_element(e)
            if not b:
                continue
            if (
                truthy(e.get("clickable"))
                or truthy(e.get("long-clickable"))
                or truthy(e.get("focusable"))
                or truthy(e.get("checkable"))
            ):
                cases.append(b)

        good_ok = bad_caught = 0
        for b in cases:
            # 好坐标：原样
            t = Target(bounds=b)
            ok, _ = coordinates_look_like_hierarchy_derived_from_elements(t, elements)
            tot_good += 1
            if ok:
                good_ok += 1
                tot_good_ok += 1
            else:
                failures.append(f"{name}: good rejected ({b.x},{b.y},{b.width}x{b.height})")

            # 坏坐标：模拟从 0.5 倍截图目测 —— 中心和尺寸都缩一半，且忘了换算回来
            for scale in (0.5,):
                t2 = Target(
                    bounds=type(b)(
                        x=b.x + b.width // 2 // 2 - (b.width // 2) // 2,
                        y=b.y + b.height // 2 // 2 - (b.height // 2) // 2,
                        width=max(b.width // 2, 1),
                        height=max(b.height // 2, 1),
                    )
                )
                tot_bad += 1
                ok2, _ = coordinates_look_like_hierarchy_derived_from_elements(t2, elements)
                if not ok2:
                    bad_caught += 1
                    tot_bad_caught += 1
                else:
                    leaks.append(f"{name}: bad passed ({t2.bounds.x},{t2.bounds.y})")

        print(f"{name:9s} cases={len(cases):3d} good_ok={good_ok:3d} bad_caught={bad_caught:3d}")

    print()
    print(f"好坐标接受: {tot_good_ok}/{tot_good} = {tot_good_ok / max(tot_good, 1) * 100:.1f}%")
    print(f"坏坐标拦截: {tot_bad_caught}/{tot_bad} = {tot_bad_caught / max(tot_bad, 1) * 100:.1f}%")

    if failures:
        print(f"\n误拒 {len(failures)} 条，前 10:")
        for f in failures[:10]:
            print(f"  - {f}")
    if leaks:
        print(f"\n漏放 {len(leaks)} 条，前 10:")
        for f in leaks[:10]:
            print(f"  - {f}")

    print()
    ok = not failures and not leaks
    if ok:
        print("PASS: 零误拒 + 零漏放")
    else:
        print("FAIL: 判据需要调整")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
