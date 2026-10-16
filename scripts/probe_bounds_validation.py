"""实测：坐标一致性校验到底能不能判别「层级树抄来的坐标」和「从缩小图目测的坐标」。

背景
----
`tap.py` 现在只用「是否在屏幕内」校验坐标，半分辨率目测的坐标照样通过，
于是点错地方却报告成功。想加一层校验：拿坐标去层级树里找对应元素。

难点
----
uiautomator 的 dump 里有一个全屏根节点（bounds 覆盖整屏），
所以「点是否落在某个节点内」这个条件**恒为真**，毫无判别力。

本脚本用真机 dump 实测三件事：
  1. 全屏根节点到底有多大
  2. 好坐标（从层级树抄的中心点）能否被正确识别
  3. 坏坐标（半分辨率目测，即真坐标/2）能否被拒绝

跑法：
    .venv\\Scripts\\python.exe scripts\\probe_bounds_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minitap"))

from mobile_use.clients.ui_automator_client import (  # noqa: E402
    UIAutomatorClient,
    _parse_hierarchy_xml_to_elements,
)
from mobile_use.utils.ui_hierarchy import get_bounds_for_element  # noqa: E402

DEVICE = "89R8ZXNBGQIRJRLF"


def _area(b: tuple[int, int, int, int]) -> int:
    """(x1,y1,x2,y2) 的面积。"""
    return (b[2] - b[0]) * (b[3] - b[1])


def dump() -> list[dict]:
    return _parse_hierarchy_xml_to_elements(
        UIAutomatorClient(device_id=DEVICE).get_hierarchy()
    )


def smallest_node_containing(
    nodes: list[tuple[tuple[int, int, int, int], str]], x: int, y: int
) -> tuple[tuple[int, int, int, int], str] | None:
    """返回包含 (x,y) 的最小节点；没有则 None。"""
    hits = [
        (b, label)
        for b, label in nodes
        if b[0] <= x <= b[2] and b[1] <= y <= b[3]
    ]
    if not hits:
        return None
    return min(hits, key=lambda bl: (bl[0][2] - bl[0][0]) * (bl[0][3] - bl[0][1]))


def main() -> None:
    elements = dump()
    if not elements:
        print("空 dump，退出")
        sys.exit(1)

    # 收集所有可解析 bounds 的节点
    nodes: list[tuple[tuple[int, int, int, int], str]] = []
    for e in elements:
        b = get_bounds_for_element(e)
        if not b:
            continue
        label = (e.get("text") or "").strip() or (e.get("content-desc") or "").strip()
        rid = e.get("resource-id") or ""
        nodes.append(
            ((b.x, b.y, b.x + b.width, b.y + b.height), label or rid or e.get("class", ""))
        )

    print(f"dump 节点数: {len(elements)}, 可解析 bounds: {len(nodes)}")

    # 1. 全屏根节点
    screen_area = 1220 * 2712
    full = [(b, lab) for b, lab in nodes if (b[2] - b[0]) * (b[3] - b[1]) >= screen_area * 0.95]
    print(f"\n全屏根节点(>=95% 屏幕面积): {len(full)} 个")
    for b, lab in full[:5]:
        print(f"  {b}  {lab[:40]}")

    areas = sorted((b[2] - b[0]) * (b[3] - b[1]) for b, _ in nodes)
    print(f"节点面积分位: min={areas[0]:,}  p50={areas[len(areas) // 2]:,}  max={areas[-1]:,}")
    print(f"屏幕面积={screen_area:,}")

    # 2 / 3. 好坐标 vs 坏坐标
    # 「可操作」的节点：clickable / long-clickable / 有 resource-id / 是输入类
    actionable = [
        (b, lab)
        for (b, lab), e in zip(
            nodes,
            [e for e in elements if get_bounds_for_element(e)],
        )
        if e.get("clickable") == "true"
        or e.get("long-clickable") == "true"
        or e.get("focusable") == "true"
        or e.get("resource-id")
    ]
    print(f"\n用于实验的候选节点(可操作/有 id): {len(actionable)}")

    MAX_AREA_FRACTION = 0.6  # 判定「背景容器」的门限

    good_ok = good_rejected = 0
    bad_caught = bad_passed = 0
    bad_examples: list[str] = []
    bad_undetected: list[str] = []

    for b, lab in actionable:
        x1, y1, x2, y2 = b
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # 好坐标：就是该节点自己的中心
        hit = smallest_node_containing(nodes, cx, cy)
        ok = hit is not None and _area(hit[0]) < screen_area * MAX_AREA_FRACTION
        if ok:
            good_ok += 1
        else:
            good_rejected += 1

        # 坏坐标：模型在 0.5 倍图上目测，且忘了换算回来
        bx, by = cx // 2, cy // 2
        bhit = smallest_node_containing(nodes, bx, by)
        is_background = bhit is not None and _area(bhit[0]) >= screen_area * MAX_AREA_FRACTION
        if bhit is None or is_background:
            bad_caught += 1
            if len(bad_examples) < 5:
                bad_examples.append(f"    {lab[:24]!r} 真({cx},{cy}) -> 猜({bx},{by}) 已拦下")
        else:
            bad_passed += 1
            if len(bad_undetected) < 5:
                bad_undetected.append(
                    f"    {lab[:24]!r} 真({cx},{cy}) -> 猜({bx},{by}) "
                    f"落在 {bhit[1][:24]!r} 面积{(bhit[0][2]-bhit[0][0])*(bhit[0][3]-bhit[0][1]):,}"
                )

    total = good_ok + good_rejected
    print("\n--- 好坐标（层级树原值）是否被误拒 ---")
    print(f"  接受 {good_ok}/{total} = {good_ok / total * 100:.1f}%   误拒 {good_rejected}")

    total2 = bad_caught + bad_passed
    print("\n--- 坏坐标（0.5 倍目测）是否被拦下 ---")
    print(f"  拦下 {bad_caught}/{total2} = {bad_caught / total2 * 100:.1f}%   漏放 {bad_passed}")
    if bad_examples:
        print("  拦下样例:")
        print("\n".join(bad_examples))
    if bad_undetected:
        print("  漏放样例（这些会静默点错）:")
        print("\n".join(bad_undetected))

    # ---- 阈值扫描：有没有任何门限能同时做到「不误拒 + 能拦下」 ----
    print("\n=== 面积门限扫描（要求：好坐标接受率=100% 的前提下，坏坐标拦得越多越好）===")
    print(f"{'threshold':>10s} {'good accept':>12s} {'bad caught':>12s}")
    results = []
    for th in [0.60, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02, 0.01, 0.005]:
        acc = sum(
            1
            for b, _ in actionable
            if (h := smallest_node_containing(nodes, (b[0] + b[2]) // 2, (b[1] + b[3]) // 2))
            is not None
            and (h[0][2] - h[0][0]) * (h[0][3] - h[0][1]) < screen_area * th
        )
        cat = sum(
            1
            for b, _ in actionable
            if (
                (h := smallest_node_containing(nodes, (b[0] + b[2]) // 4, (b[1] + b[3]) // 4))
                is None
                or (h[0][2] - h[0][0]) * (h[0][3] - h[0][1]) >= screen_area * th
            )
        )
        print(
            f"{th:10.3f} {acc / len(actionable) * 100:11.1f}% "
            f"{cat / len(actionable) * 100:11.1f}%"
        )
        results.append((th, acc / len(actionable), cat / len(actionable)))

    best = max((r for r in results if r[1] == 1.0), key=lambda r: r[2], default=None)
    print()
    if best is None:
        print("没有任何门限能在不误拒的前提下拦下坏坐标。")
    else:
        print(f"最佳门限 {best[0]}: 接受 {best[1] * 100:.1f}% / 拦下 {best[2] * 100:.1f}%")

    # ---- 换个思路：不看「最小包含节点」，直接看「坐标是否精确命中某个已渲染节点」 ----
    # 真机上层级树里有些节点极小(min=339)，半分辨率坐标几乎不可能正好落在小节点里
    print("\n=== 替代思路：绝对面积上限（小节点才是真控件，大的是容器）===")
    for abs_area in [500_000, 200_000, 100_000, 50_000, 20_000]:
        acc = sum(
            1
            for b, _ in actionable
            if (h := smallest_node_containing(nodes, (b[0] + b[2]) // 2, (b[1] + b[3]) // 2))
            is not None
            and (h[0][2] - h[0][0]) * (h[0][3] - h[0][1]) < abs_area
        )
        cat = sum(
            1
            for b, _ in actionable
            if (
                (h := smallest_node_containing(nodes, (b[0] + b[2]) // 4, (b[1] + b[3]) // 4))
                is None
                or (h[0][2] - h[0][0]) * (h[0][3] - h[0][1]) >= abs_area
            )
        )
        print(
            f"  绝对面积<{abs_area:>9,}: 接受 {acc / len(actionable) * 100:5.1f}%  "
            f"拦下 {cat / len(actionable) * 100:5.1f}%"
        )

    # ---- 关键对照：模型到底是「抄层级树」还是「目测缩略图」？ ----
    # 如果模型抄层级树，那 bounds 一定和某个真实节点**高度重合**。
    # 用「与最佳匹配节点的 IoU」比「是否被包含」判别力更强。
    print("\n=== IoU 判别力（好坐标 vs 坏坐标与最近节点的 IoU 分布）===")

    def best_iou(x: int, y: int, w: int, h: int) -> float:
        bestv = 0.0
        for b, _ in nodes:
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(x, bx1), max(y, by1)
            ix2, iy2 = min(x + w, bx2), min(y + h, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = w * h + (bx2 - bx1) * (by2 - by1) - inter
            bestv = max(bestv, inter / union if union else 0.0)
        return bestv

    goods, bads = [], []
    for b, _ in actionable:
        x1, y1, x2, y2 = b
        w, h = x2 - x1, y2 - y1
        goods.append(best_iou(x1, y1, w, h))
        bads.append(best_iou(x1 // 2, y1 // 2, w // 2, h // 2))

    goods.sort()
    bads.sort()
    q = lambda a, p: a[min(len(a) - 1, int(len(a) * p))]  # noqa: E731
    print(f"  好坐标 IoU: min={goods[0]:.3f} p05={q(goods, 0.05):.3f} p50={q(goods, 0.5):.3f}")
    print(f"  坏坐标 IoU: p50={q(bads, 0.5):.3f} p95={q(bads, 0.95):.3f} max={bads[-1]:.3f}")
    for th in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        g = sum(1 for v in goods if v >= th) / len(goods)
        bd = sum(1 for v in bads if v < th) / len(bads)
        print(
            f"  IoU>={th:.1f}: 好坐标接受 {g * 100:5.1f}%   坏坐标拦下 {bd * 100:5.1f}%"
        )


if __name__ == "__main__":
    main()
