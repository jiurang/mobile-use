"""比较两种坐标校验判据，挑一个既不怕框漂移、又能拦下目测坐标的。

判据 A：IoU（当前实现）
    要求模型给的**整个框**与某个真实节点高度重合。
    问题：框尺寸对 IoU 极其敏感 —— 外扩 10% 就掉到 0.826，
    而坏坐标（0.5 倍目测）的上界也正好是 0.826，两者撞车。

判据 B：中心点命中 + 尺寸合理性（中心点必须落在某个**较小**的真实节点里）
    只关心「点在哪」，不关心「框多大」，天然免疫尺寸漂移。

真机实测输入：
    快手首页 1220x2712，247 节点，164 个可操作节点，
    31 个覆盖 >=95% 屏幕面积的容器（必须排除，否则判据 B 也会失效）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minitap"))

from minitap.mobile_use.tools.bounds_validation import (  # noqa: E402
    coordinates_look_like_hierarchy_derived_from_elements,
)
from minitap.mobile_use.tools.types import Target  # noqa: E402
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds, get_bounds_for_element  # noqa: E402

W, H = 1220, 2712
SCREEN = W * H

# 假 dump：含全屏容器 + 真实控件
DUMP = [
    {"bounds": "[0,0][1220,2712]", "class": "android.widget.FrameLayout"},
    {"bounds": "[0,0][1220,2660]", "class": "android:id/content"},
    {"bounds": "[0,1000][1220,1700]", "class": "big_container"},  # 31% 屏幕，中间容器
    {"bounds": "[40,2500][1180,2600]", "text": "发送", "class": "Button"},
    {"bounds": "[60,100][300,180]", "text": "搜索", "class": "EditText"},
    {"bounds": "[1100,1300][1200,1400]", "class": "ImageView"},
]


def centers():
    """返回 [(label, x1,y1,x2,y2)]，含容器与控件。"""
    out = []
    for e in DUMP:
        b = get_bounds_for_element(e)
        out.append((e.get("text") or e.get("class", "?"), b.x, b.y, b.x + b.width, b.y + b.height))
    return out


def iou_check(target: Target) -> bool:
    ok, _ = coordinates_look_like_hierarchy_derived_from_elements(target, DUMP, min_iou=0.9)
    return ok


def center_check(target: Target, max_area_fraction: float = 0.05) -> tuple[bool, str]:
    """中心点是否落在一个「足够小」的真实节点里（即真控件，而不是容器）。"""
    c = target.bounds.get_center()
    best_label, best_area = None, None
    for e in DUMP:
        b = get_bounds_for_element(e)
        if b is None:
            continue
        if b.x <= c.x <= b.x + b.width and b.y <= c.y <= b.y + b.height:
            area = b.width * b.height
            if best_area is None or area < best_area:
                best_area, best_label = area, (e.get("text") or e.get("class", "?"))
    if best_area is None:
        return False, "no node"
    if best_area >= SCREEN * max_area_fraction:
        return False, f"only container {best_label!r}"
    return True, str(best_label)


print("=== 1. IoU 判据对尺寸漂移的敏感性 ===")
print(f"{'drift':>12s} {'IoU':>8s} {'A: IoU>=0.9':>12s} {'B: center':>10s}")
print("-" * 46)
base = [(40, 2500, 1140, 100), (60, 100, 240, 80), (1100, 1300, 100, 100)]
for name, fx, fy in [
    ("原样", 1.0, 1.0), ("外扩5%", 1.05, 1.05), ("外扩10%", 1.1, 1.1),
    ("外扩20%", 1.2, 1.2), ("外扩30%", 1.3, 1.3), ("收缩10%", 0.9, 0.9),
    ("收缩30%", 0.7, 0.7),
]:
    a = b = 0
    for x1, y1, w, h in base:
        nw, nh = int(w * fx), int(h * fy)
        t = Target(
            bounds=ElementBounds(
                x=int(x1 - (nw - w) / 2),
                y=int(y1 - (nh - h) / 2),
                width=nw,
                height=nh,
            )
        )
        if iou_check(t):
            a += 1
        ok, _ = center_check(t)
        if ok:
            b += 1
    print(f"{name:>12s} {'':>8s} {a}/3{'':>8s} {b}/3")

print("\n=== 2. 坏坐标（0.5 倍目测）拦截率 ===")
a = b = 0
total = 0
for label, x1, y1, x2, y2 in centers():
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    # 目测坐标：中心和尺寸都按 0.5 缩，且忘了换算回来
    t = Target(
        bounds=ElementBounds(
            x=cx // 2,
            y=cy // 2,
            width=max(w // 2, 1),
            height=max(h // 2, 1),
        )
    )
    total += 1
    if iou_check(t):
        a += 1
    ok, _ = center_check(t)
    if ok:
        b += 1
print(f"  A: IoU>=0.9  放行(漏网) {a}/{total}")
print(f"  B: center    放行(漏网) {b}/{total}")

print("\n=== 3. 关键对照：容器中心点 ===")
for label, x1, y1, x2, y2 in centers():
    t = Target(bounds=ElementBounds(x=x1, y=y1, width=x2 - x1, height=y2 - y1))
    ok_b, why = center_check(t)
    print(f"  {label[:28]:28s} B={'过' if ok_b else '拦'} ({why})")

print("\n=== 4. B 判据：max_area_fraction 扫描（对好坐标要全过）===")
for frac in [0.30, 0.10, 0.05, 0.02, 0.01, 0.005]:
    acc = sum(1 for label, x1, y1, x2, y2 in centers() if center_check(
        Target(bounds=ElementBounds(x=x1, y=y1, width=x2 - x1, height=y2 - y1)), frac)[0])
    print(f"  frac={frac:<6}: 好坐标接受 {acc}/{len(centers())}")
