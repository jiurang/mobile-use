"""第三版判据的探索：在真机数据上比较几种方案，找真正能用的那个。

已排除的方案（有真机数据支撑，不是拍脑袋）
------------------------------------------
1. 「点是否落在某节点内」        -> 拦截 0%。dump 里有 31 个 >=95% 屏幕的容器，恒为真。
2. 「中心点是否落在最小包含节点内」 -> 拦截 0%。同样的容器问题。
3. 「中心点命中 IoU 最佳匹配节点」  -> 拦截 0%。半尺寸的框会匹配到大容器，
                                       中心点自然落在里面（本轮实测 68/68 全漏）。
4. 纯 IoU 门槛 0.9              -> 拦截 100%、零误拒，但外扩 10% 就掉到 0.826，
                                       与坏坐标上界 0.826 撞车。

本轮要试的：归一化 IoU
---------------------
先按「最佳匹配节点」把模型给的框做尺度对齐（消除尺寸漂移），
再比 IoU —— 这样 IoU 只反映位置差异，不再被框大小干扰。

跑法：
    .venv\\Scripts\\python.exe scripts\\probe_validation_criteria2.py
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
from minitap.mobile_use.utils.ui_hierarchy import (  # noqa: E402
    ElementBounds,
    get_bounds_for_element,
)

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


def iou(a: ElementBounds, b: ElementBounds) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.width, a.y + a.height
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.width, b.y + b.height
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    if inter <= 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def best_iou(b: ElementBounds, nodes: list[ElementBounds]) -> tuple[float, ElementBounds | None]:
    best, bb = 0.0, None
    for n in nodes:
        v = iou(b, n)
        if v > best:
            best, bb = v, n
    return best, bb


def iou_scaled_to(b: ElementBounds, ref: ElementBounds) -> float:
    """把 b 的尺寸缩放到 ref 再算 IoU —— 消除尺寸漂移，只看位置。"""
    sx = ref.width / b.width if b.width else 1.0
    sy = ref.height / b.height if b.height else 1.0
    # 以 b 的中心为锚点缩放
    cx, cy = b.x + b.width / 2, b.y + b.height / 2
    scaled = ElementBounds(
        x=int(cx - ref.width / 2), y=int(cy - ref.height / 2),
        width=ref.width, height=ref.height,
    )
    del sx, sy
    return iou(scaled, ref)


def main() -> None:
    all_good_scaled: list[float] = []
    all_bad_scaled: list[float] = []
    all_good_raw: list[float] = []
    all_bad_raw: list[float] = []

    for name in SCREENS:
        try:
            elements = capture(name)
        except Exception:
            continue
        if not elements:
            continue
        nodes = [
            b
            for b in (get_bounds_for_element(e) for e in elements)
            if b is not None
        ]
        cases = []
        for e in elements:
            if not truthy(e.get("visible-to-user", True)):
                continue
            b = get_bounds_for_element(e)
            if b and (
                truthy(e.get("clickable"))
                or truthy(e.get("long-clickable"))
                or truthy(e.get("focusable"))
                or truthy(e.get("checkable"))
            ):
                cases.append(b)
        if not cases:
            continue

        for b in cases:
            cx, cy = b.x + b.width // 2, b.y + b.height // 2
            # 好坐标：原样
            g_raw, _ = best_iou(b, nodes)
            g_scaled, ref = best_iou(b, nodes)
            all_good_raw.append(g_raw)
            all_good_scaled.append(iou_scaled_to(b, ref) if ref else 0.0)

            # 坏坐标：0.5 倍目测（中心和尺寸都缩一半，忘换算）
            bad = ElementBounds(
                x=cx // 2 - (b.width // 2) // 2, y=cy // 2 - (b.height // 2) // 2,
                width=max(b.width // 2, 1), height=max(b.height // 2, 1),
            )
            bd_raw, _ = best_iou(bad, nodes)
            bd_scaled, ref2 = best_iou(bad, nodes)
            all_bad_raw.append(bd_raw)
            all_bad_scaled.append(iou_scaled_to(bad, ref2) if ref2 else 0.0)

    def stats(name, vals):
        v = sorted(vals)
        return (f"{name:22s} n={len(v):3d} min={v[0]:.3f} p05={v[len(v)//20]:.3f} "
                f"p50={v[len(v)//2]:.3f} p95={v[len(v)*19//20]:.3f} max={v[-1]:.3f}")

    print("=== 原始 IoU 分布 ===")
    print(stats("good(层级树原值)", all_good_raw))
    print(stats("bad(0.5倍目测)", all_bad_raw))
    print()
    print("=== 归一化 IoU 分布（按最佳匹配节点缩放后）===")
    print(stats("good", all_good_scaled))
    print(stats("bad", all_bad_scaled))

    print("\n=== 混淆矩阵（阈值扫描）===")
    print(f"{'thr':>6s} {'good accept':>12s} {'bad caught':>12s} {'verdict':>18s}")
    print("-" * 54)
    for thr in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        ga = sum(1 for v in all_good_scaled if v >= thr) / max(len(all_good_scaled), 1)
        bc = sum(1 for v in all_bad_scaled if v < thr) / max(len(all_bad_scaled), 1)
        if ga == 1.0 and bc >= 0.95:
            verdict = "<= 可用"
        elif ga == 1.0:
            verdict = f"漏{bc * 100:.0f}%"
        else:
            verdict = f"误拒{(1 - ga) * 100:.0f}%"
        print(f"{thr:6.2f} {ga * 100:11.1f}% {bc * 100:11.1f}% {verdict:>18s}")

    # 原始 IoU 在有尺寸漂移下的表现也要一并看
    print("\n=== 对照：原始 IoU 门限在真机上的混淆矩阵 ===")
    for thr in [0.95, 0.9, 0.85]:
        ga = sum(1 for v in all_good_raw if v >= thr) / max(len(all_good_raw), 1)
        bc = sum(1 for v in all_bad_raw if v < thr) / max(len(all_bad_raw), 1)
        print(f"thr={thr:.2f}  good {ga * 100:5.1f}%  bad caught {bc * 100:5.1f}%")


if __name__ == "__main__":
    main()
