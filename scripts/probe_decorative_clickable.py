"""实测：「不可交互 + 无文字」的行里，有多少**实际上真的能点**。

为什么要问这个
--------------
裁掉纯装饰行是降本手段（L4）。但判据有个前提假设：
**「uiautomator dump 里没有 clickable」== 「这个位置点了没反应」**。

这个前提在自定义 View / Canvas 画的 App 上会破掉 —— 那类可点区域在
accessibility 树里既没有 clickable=true，文字又是画上去的（dump 里也没有）。
按判据会被误裁，模型就再也点不到它了。

所以这里**不推理，直接点**。

怎么算「实际上能点」
------------------
1. 先按判据挑出候选行（不可交互 + 无文字）。
2. 对每一行，算出它**中心点**上最顶层的那个原始节点（原始 dump 的后序顺序
   里最后一个包含该点的节点 = 画在最上面的）。
   - 如果那个顶层节点是 clickable 的，但**它不在元素表里**（被
     `normalize_hierarchy` 的去重/裁剪合并掉了）→ 裁掉这行会**连带丢掉一个
     真的可点目标**。这是最危险的一类。
   - 如果顶层节点可点、且**在表里** → 裁掉这行不损失什么（模型点的是表里那行）。
   - 如果顶层节点就是这行自己（没被盖住）→ 真正的测试对象，点它看屏幕变没变。
3. 屏幕变化用**像素差**判定（Canvas 画的 App，accessibility 树可能压根不变，
   只有像素差能看出来）。

用法
----
    uv run python v/probe_decorative_clickable.py            # 跑全部页面
    uv run python v/probe_decorative_clickable.py --pages 2  # 只跑前 2 个
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "minitap"))

from minitap.mobile_use.utils.ui_hierarchy_view import (  # noqa: E402
    _INTERACTIVE_CLASSES,
    normalize_hierarchy,
)

SERIAL = "89R8ZXNBGQIRJRLF"

# 判据里的「可交互」标记，和 _HEADER 里给模型看的那套一致。
INTERACTIVE = set("CLFKSR")

_SYSTEMUI_PACKAGE = "com.android.systemui:"

# 像素差阈值：变化像素占比超过它就算「屏幕变了」。
# 状态栏时钟自己走一格大约占 0.09%，所以 0.3% 能把噪音挡在外面。
PIXEL_DIFF_RATIO = 0.003

_BOUNDS_RE = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")

# 页面清单：包名 + 一个便于人看的名字。
PAGES: list[tuple[str, str]] = [
    ("com.android.settings", "系统设置"),
    ("com.tencent.mm", "微信"),
    ("com.ss.android.ugc.aweme", "抖音"),
    ("com.xingin.xhs", "小红书"),
    ("com.sina.weibo", "微博"),
    ("com.taobao.taobao", "淘宝"),
    ("com.miui.weather2", "小米天气"),
    ("com.miui.gallery", "小米相册"),
    ("com.tencent.mobileqq", "QQ"),
    ("com.jingdong.app.mall", "京东"),
    ("com.autonavi.minimap", "高德地图"),
    ("com.zhihu.android", "知乎"),
]


def _box_of(element: dict) -> tuple[int, int, int, int] | None:
    """原始 dump 的 bounds（字符串）-> (x, y, w, h)，与表里的取整方式一致。"""
    raw = element.get("bounds")
    if not isinstance(raw, str):
        return None
    match = _BOUNDS_RE.match(raw.strip())
    if not match:
        return None
    x1, y1, x2, y2 = (int(g) for g in match.groups())
    return x1, y1, x2 - x1, y2 - y1


def _is_clickable(element: dict) -> bool:
    return element.get("clickable") == "true" or element.get("long-clickable") == "true"


def _contains(box: tuple[int, int, int, int], cx: int, cy: int) -> bool:
    x, y, w, h = box
    return x <= cx < x + w and y <= cy < y + h


def _topmost_at(elements: list[dict], cx: int, cy: int) -> dict | None:
    """中心点上画在最上面的原始节点。

    `_parse_hierarchy_xml_to_elements` 是 pre-order DFS，后面的节点画在上面，
    所以「最后一个包含该点的节点」就是最顶层的那个。
    """
    found = None
    for element in elements:
        box = _box_of(element)
        if box and _contains(box, cx, cy):
            found = element
    return found


def _changed_ratio(before, after, threshold: int = 24) -> float:
    """两张图的像素差占比。

    阈值取 24（不是 0）：uiautomator2 走的是 `takeScreenshot` 的 JPEG(quality=80)，
    同一屏两次截的字节本来就不完全一样。噪底由 `_noise_floor()` 实测出来。
    """
    from PIL import ImageChops

    diff = ImageChops.difference(before.convert("RGB"), after.convert("RGB")).convert("L")
    histogram = diff.histogram()
    changed = sum(histogram[threshold:])
    return changed / (before.width * before.height)


def _noise_floor(device) -> float:
    """静止屏连拍两次的像素差占比 —— 判定「屏幕变了」的噪底线。"""
    a = device.screenshot()
    time.sleep(0.4)
    b = device.screenshot()
    return _changed_ratio(a, b)


_SHOT_DIR = Path(__file__).resolve().parent.parent / "probe_out" / "decorative_shots"
_ROWS_FILE = Path(__file__).resolve().parent.parent / "probe_out" / "decorative_rows.jsonl"


def _save_shots(before, after, tag: str) -> None:
    """把「点了有反应」的前后截图存下来，人工复核用。"""
    _SHOT_DIR.mkdir(exist_ok=True)
    before.save(_SHOT_DIR / f"{tag}_before.png")
    after.save(_SHOT_DIR / f"{tag}_after.png")


def _record(row: dict) -> None:
    """把每个装饰候选的完整属性和实测结果写进 JSONL，供离线换判据复算。"""
    with _ROWS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _classify_decor_rows(
    tree_rows: list[dict],
    raw_elements: list[dict],
    table_boxes: set[tuple[int, int, int, int]],
) -> list[dict]:
    """把候选装饰行分三类。

    * `dead`         —— 中心点上就是它自己，点它应该没反应（真正的装饰）
    * `covered`      —— 中心点上盖着一个可点节点，且那个节点**在元素表里**
                        （裁掉这行不损失可点目标）
    * `covered_lost` —— 中心点上盖着一个可点节点，但那个节点**不在元素表里**
                        （被 normalize_hierarchy 合并掉了；裁掉这行会连带
                        丢一个真的可点目标 —— 最危险）
    """
    classified: list[dict] = []
    for row in tree_rows:
        flags = set(row.get("flags") or "")
        label = (row.get("label") or "").strip()
        if flags & INTERACTIVE or label:
            continue  # 不是候选装饰行
        # 类名本身就意味着可操作（EditText / ImageButton …）的也要留 ——
        # 它的可操作性没写进 flags（flags 只有 C/L/F/K/S/R）。
        if row.get("class_name") in _INTERACTIVE_CLASSES:
            continue

        box = row.get("bounds")
        if not box:
            continue
        cx = int(box["x"]) + int(box["width"]) // 2
        cy = int(box["y"]) + int(box["height"]) // 2

        top = _topmost_at(raw_elements, cx, cy)
        kind = "dead"
        if top is not None and _is_clickable(top):
            top_box = _box_of(top)
            kind = "covered" if top_box in table_boxes else "covered_lost"

        classified.append(
            {
                "kind": kind,
                "row": row,
                "cx": cx,
                "cy": cy,
                "top_class": (top or {}).get("class", ""),
                "top_id": (top or {}).get("resource-id", ""),
                "w": int(box["width"]),
                "h": int(box["height"]),
            }
        )
    return classified


def _launch(device, package: str) -> None:
    # 先回桌面：前面几轮点完按 BACK 之后，通知栏/弹窗可能还盖在上面，
    # 那时 dump 出来的全是 systemui，量到的不是这个 App 的页面。
    device.shell("input keyevent KEYCODE_HOME")
    time.sleep(1.0)
    device.shell(f"monkey -p {package} -c android.intent.category.LAUNCHER 1")
    time.sleep(3.5)


def _tap(device, cx: int, cy: int) -> None:
    device.shell(f"input tap {cx} {cy}")


def _back(device) -> None:
    device.shell("input keyevent KEYCODE_BACK")
    time.sleep(1.2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=len(PAGES))
    parser.add_argument("--tap-test", action="store_true", default=True)
    parser.add_argument("--no-tap", dest="tap_test", action="store_false")
    args = parser.parse_args()

    from minitap.mobile_use.clients.ui_automator_client import (
        UIAutomatorClient,
        _parse_hierarchy_xml_to_elements,
    )

    client = UIAutomatorClient(SERIAL)
    device = client._ensure_connected()

    noise = _noise_floor(device)
    print(f"噪底（静止屏连拍两次的像素差）= {noise:.4%}   判定阈值 = {PIXEL_DIFF_RATIO:.2%}")
    print()

    grand_total = {"decor": 0, "dead": 0, "covered": 0, "covered_lost": 0, "responded": 0}

    for package, title in PAGES[: args.pages]:
        print("=" * 78)
        print(f"[{title}] {package}")
        _launch(device, package)
        time.sleep(1.0)

        xml = device.dump_hierarchy(compressed=True)
        raw_elements = _parse_hierarchy_xml_to_elements(xml)
        tree_rows = normalize_hierarchy(raw_elements)
        if tree_rows is None:
            print("  层级树形状不认识，跳过")
            continue

        # 复刻 build_screen_elements 的 systemui 过滤，得到**真实的**表内容。
        kept = [
            row
            for row in tree_rows
            if not (row.get("resource_id") or "").startswith(_SYSTEMUI_PACKAGE)
        ]
        table_boxes = {
            (
                int(r["bounds"]["x"]),
                int(r["bounds"]["y"]),
                int(r["bounds"]["width"]),
                int(r["bounds"]["height"]),
            )
            for r in kept
            if r.get("bounds")
        }

        classified = _classify_decor_rows(kept, raw_elements, table_boxes)
        counts = {"dead": 0, "covered": 0, "covered_lost": 0}
        for item in classified:
            counts[item["kind"]] += 1

        print(f"  表行数 = {len(kept)}   装饰候选 = {len(classified)}")
        print(
            f"    dead(自己就是顶层)={counts['dead']}  "
            f"covered(盖着可点且在表里)={counts['covered']}  "
            f"covered_lost(盖着可点但不在表里)={counts['covered_lost']}"
        )

        grand_total["decor"] += len(classified)
        for key in ("dead", "covered", "covered_lost"):
            grand_total[key] += counts[key]

        if not args.tap_test:
            continue

        # ---- 逐个真点：只有 dead 那一类值得点 ----
        responded: list[dict] = []
        tree_only: list[dict] = []
        for item in classified:
            if item["kind"] != "dead":
                continue
            cx, cy = item["cx"], item["cy"]
            row = item["row"]

            # 点之前先确认屏幕**静止**。刚打开的页面还在转场时，
            # 「点完屏幕变了」是转场造成的，不是这一行造成的 —— 必须挡掉。
            before_shot = None
            for _attempt in range(4):
                shot_a = device.screenshot()
                time.sleep(0.35)
                shot_b = device.screenshot()
                if _changed_ratio(shot_a, shot_b) <= PIXEL_DIFF_RATIO:
                    before_shot = shot_a
                    break
                time.sleep(0.8)
            if before_shot is None:
                continue  # 页面一直在动，这一行没法干净地测

            # 重新 dump 一遍，确认这行还在（前面几次点击可能把页面弄跑了）
            before_xml = device.dump_hierarchy(compressed=True)
            before_rows = normalize_hierarchy(_parse_hierarchy_xml_to_elements(before_xml))
            if before_rows is None:
                break
            still_there = any(
                r.get("bounds")
                and (
                    int(r["bounds"]["x"]),
                    int(r["bounds"]["y"]),
                    int(r["bounds"]["width"]),
                    int(r["bounds"]["height"]),
                )
                == (
                    int(row["bounds"]["x"]),
                    int(row["bounds"]["y"]),
                    int(row["bounds"]["width"]),
                    int(row["bounds"]["height"]),
                )
                for r in before_rows
            )
            if not still_there:
                continue  # 页面变了，这行已经不在，跳过

            _tap(device, cx, cy)
            time.sleep(1.2)
            after_shot = device.screenshot()
            after_xml = device.dump_hierarchy(compressed=True)

            ratio = _changed_ratio(before_shot, after_shot)
            tree_changed = before_xml != after_xml
            record = {
                "page": title,
                "package": package,
                "kind": item["kind"],
                "cx": cx,
                "cy": cy,
                "x": int(row["bounds"]["x"]),
                "y": int(row["bounds"]["y"]),
                "w": item["w"],
                "h": item["h"],
                "area": item["w"] * item["h"],
                "ratio": ratio,
                "tree_changed": tree_changed,
                "id": row.get("resource_id", ""),
                "cls": row.get("class_name", ""),
                "flags": row.get("flags", ""),
                "label": row.get("label", ""),
                "outcome": "responded" if ratio > PIXEL_DIFF_RATIO else "no_change",
            }
            _record(record)
            if ratio > PIXEL_DIFF_RATIO:
                responded.append(record)
                print(
                    f"    ⚠️ 点了有反应: ({cx},{cy}) id={row.get('resource_id', '')} "
                    f"cls={row.get('class_name', '')} 像素差={ratio:.3%} 树变={tree_changed}"
                )
                _save_shots(before_shot, after_shot, f"{package}_{cx}_{cy}")
                _back(device)
                time.sleep(0.6)
                # 回到原页面了吗？
                restored = device.screenshot()
                if _changed_ratio(before_shot, restored) > PIXEL_DIFF_RATIO:
                    _back(device)
                    time.sleep(0.6)
            elif tree_changed:
                # 只有 accessibility 树变了、像素没变 —— 大概率是焦点/选中的
                # 抖动，不是真的「点开了东西」。单独记一笔，不计入「可点」。
                tree_only.append(record)
                print(
                    f"    ~ 树变像素没变: ({cx},{cy}) id={row.get('resource_id', '')} "
                    f"像素差={ratio:.3%}"
                )

        grand_total["responded"] += len(responded)
        grand_total["tree_only"] = grand_total.get("tree_only", 0) + len(tree_only)
        print(
            f"  → 实点后有反应: {len(responded)} / {counts['dead']} (dead 类)"
            f"   [另有 {len(tree_only)} 个只变了树、没变像素]"
        )
        print()

    print("=" * 78)
    print("汇总")
    print(
        f"  装饰候选 {grand_total['decor']}  "
        f"dead={grand_total['dead']}  covered={grand_total['covered']}  "
        f"covered_lost={grand_total['covered_lost']}  实点有反应={grand_total['responded']}"
        f"  [只变树={grand_total.get('tree_only', 0)}]"
    )
    if grand_total["dead"]:
        print(
            f"  dead 类里真能点的比例 = {grand_total['responded'] / grand_total['dead']:.1%}"
            f"  （占全部装饰候选 {grand_total['responded'] / max(grand_total['decor'], 1):.1%}）"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
