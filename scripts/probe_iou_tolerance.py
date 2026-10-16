"""量一下：模型给的 bounds 相对真实节点有尺寸漂移时，IoU 还剩多少。

用途：为 MIN_IOU 定门限 —— 要同时满足「不误拒有轻微漂移的正常坐标」和
「拦下从缩放截图目测的坐标」。真机实测坏坐标 IoU max=0.826，好坐标 IoU=1.000。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minitap"))

from minitap.mobile_use.tools.bounds_validation import best_iou_for_bounds  # noqa: E402
from minitap.mobile_use.tools.types import Target  # noqa: E402
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds  # noqa: E402

DUMP = [
    {"bounds": "[40,2500][1180,2600]", "text": "发送"},
    {"bounds": "[60,100][300,180]", "text": "搜索"},
    {"bounds": "[1100,1300][1200,1400]", "text": "头像"},
    {"bounds": "[200,800][800,1100]", "text": "卡片"},
]

def _origin_and_size(name: str) -> tuple[int, int, int, int]:
    """按控件名给出它在假 dump 里的实际位置与尺寸。"""
    return {
        "底部大按钮": (40, 2500, 1140, 100),
        "顶部搜索框": (60, 100, 240, 80),
        "小图标": (1100, 1300, 100, 100),
        "卡片": (200, 800, 600, 300),
    }[name]


_SIZES = ["底部大按钮", "顶部搜索框", "小图标", "卡片"]


def measure(x1: int, y1: int, w: int, h: int, fx: float, fy: float) -> float:
    nw, nh = int(w * fx), int(h * fy)
    target = Target(
        bounds=ElementBounds(
            x=int(x1 - (nw - w) / 2), y=int(y1 - (nh - h) / 2), width=nw, height=nh
        )
    )
    iou, _ = best_iou_for_bounds(target.bounds, DUMP)
    return iou


def main() -> None:
    drifts = [
        ("原样", 1.0, 1.0),
        ("外扩5%", 1.05, 1.05),
        ("外扩10%", 1.10, 1.10),
        ("外扩15%", 1.15, 1.15),
        ("外扩20%", 1.20, 1.20),
        ("外扩25%", 1.25, 1.25),
        ("外扩30%", 1.30, 1.30),
        ("收缩5%", 0.95, 0.95),
        ("收缩10%", 0.90, 0.90),
        ("收缩20%", 0.80, 0.80),
        ("收缩30%", 0.70, 0.70),
        ("宽外扩20%", 1.20, 1.0),
        ("宽外扩30%", 1.30, 1.0),
        ("高外扩30%", 1.0, 1.30),
    ]

    print(f"{'drift':>12s} | " + " | ".join(f"{n:>10s}" for n in _SIZES))
    print("-" * (14 + 13 * len(_SIZES)))

    worst_by_drift: dict[str, float] = {}
    for name, fx, fy in drifts:
        vals = [measure(*_origin_and_size(n), fx, fy) for n in _SIZES]
        worst_by_drift[name] = min(vals)
        cells = " | ".join(f"{v:10.3f}" for v in vals)
        print(f"{name:>12s} | {cells}")

    print()
    print("最小 IoU（所有控件尺寸中最差的那个）:")
    for name, v in worst_by_drift.items():
        mark = ""
        if v >= 0.9:
            mark = "  <- 0.9 门限下仍接受"
        print(f"  {name:>12s}: {v:.3f}{mark}")

    print()
    print("坏坐标实测上界 (0.5 倍目测, 164 节点): 0.826")
    print("=> 门限必须落在 [最差好坐标, 0.826] 区间内")


if __name__ == "__main__":
    main()
