"""量化裁掉「纯装饰行」的**收益**（token）和**代价**（会裁掉多少行）。

和 `probe_decorative_clickable.py` 配套：那个脚本负责**实测**「点了有没有反应」
（结果写进 `decorative_rows.jsonl`），这个脚本负责**算账** ——
每个候选判据各能省多少 token、各会裁掉多少行、各裁掉多少「有 id 的行」。

判据候选
--------
* `A` 原始判据：flags 里没有 C/L/F/K/S，且无文字
* `B` A + R 也算可交互（executor 提示词说 R 行的 bounds 用来做 swipe）
* `C` B + 类名本身可操作的控件也算（EditText / ImageButton …）
* `D` C + 「有 resource_id 的行不裁」（仓库自己的既有结论：id 名字本身就是线索）
* `E` C + 只裁面积极小的行（width*height < 阈值）

用法
----
    uv run python v/probe_decorative_crop_impact.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "minitap"))

from minitap.mobile_use.utils.screen_elements import (  # noqa: E402
    build_screen_elements,
    render_screen_elements,
)
from minitap.mobile_use.utils.ui_hierarchy_view import (  # noqa: E402
    _INTERACTIVE_CLASSES,
    normalize_hierarchy,
)

SERIAL = "89R8ZXNBGQIRJRLF"

_INTERACTIVE_NO_R = frozenset("CLFKS")
_INTERACTIVE_WITH_R = frozenset("CLFKSR")

# 面积阈值候选（像素²）。手机屏 1220x2712 = 330 万像素。
AREA_THRESHOLDS = (400, 1600, 6400, 25000)

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

OUT = Path(__file__).resolve().parent.parent / "probe_out" / "decorative_impact.json"


def _criteria(element) -> dict[str, bool]:
    """每一行在每个候选判据下「会不会被裁」。True = 会被裁掉。"""
    flags = set(element.flags or "")
    has_text = bool((element.label or "").strip() or (element.text or "").strip())
    has_id = bool(element.resource_id)
    is_ocr = element.kind == "ocr"
    class_interactive = element.class_name in _INTERACTIVE_CLASSES
    area = element.bounds.width * element.bounds.height

    def decor(interactive: frozenset[str]) -> bool:
        return (not is_ocr) and (not (flags & interactive)) and (not has_text)

    out = {
        "A": decor(_INTERACTIVE_NO_R),
        "B": decor(_INTERACTIVE_WITH_R),
        "C": decor(_INTERACTIVE_WITH_R) and (not class_interactive),
    }
    out["D"] = out["C"] and (not has_id)
    for threshold in AREA_THRESHOLDS:
        out[f"E{threshold}"] = out["C"] and area < threshold
    return out


def main() -> int:
    import tiktoken

    from minitap.mobile_use.clients.ui_automator_client import (
        UIAutomatorClient,
        _parse_hierarchy_xml_to_elements,
    )

    client = UIAutomatorClient(SERIAL)
    device = client._ensure_connected()
    enc = tiktoken.get_encoding("o200k_base")

    names = ["A", "B", "C", "D"] + [f"E{t}" for t in AREA_THRESHOLDS]
    cropped = {n: 0 for n in names}
    cropped_with_id = {n: 0 for n in names}
    tok_before = 0
    tok_after = {n: 0 for n in names}
    total_rows = 0
    pages_used = 0
    detail: list[dict] = []

    for package, title in PAGES:
        device.shell("input keyevent KEYCODE_HOME")
        time.sleep(1.0)
        device.shell(f"monkey -p {package} -c android.intent.category.LAUNCHER 1")
        time.sleep(3.5)

        xml = device.dump_hierarchy(compressed=True)
        raw = _parse_hierarchy_xml_to_elements(xml)
        tree_rows = normalize_hierarchy(raw)
        if tree_rows is None:
            print(f"[{title}] 形状不认识，跳过")
            continue
        screen = build_screen_elements(
            tree_rows=tree_rows, ocr_results=None, foreground_package=package
        )
        if len(screen) < 5:
            print(f"[{title}] 表只有 {len(screen)} 行（App 没起来），跳过")
            continue

        pages_used += 1
        total_rows += len(screen)
        before = render_screen_elements(screen)
        tok_before += len(enc.encode(before))

        for name in names:
            kept = [e for e in screen if not _criteria(e)[name]]
            for position, element in enumerate(kept, 1):
                element.index = position
            tok_after[name] += len(enc.encode(render_screen_elements(kept)))
            cut = [e for e in screen if _criteria(e)[name]]
            cropped[name] += len(cut)
            cropped_with_id[name] += sum(1 for e in cut if e.resource_id)

        for element in screen:
            detail.append(
                {
                    "package": package,
                    "page": title,
                    "kind": element.kind,
                    "label": element.label,
                    "text": element.text,
                    "resource_id": element.resource_id,
                    "class_name": element.class_name,
                    "flags": element.flags,
                    "x": element.bounds.x,
                    "y": element.bounds.y,
                    "w": element.bounds.width,
                    "h": element.bounds.height,
                    "area": element.bounds.width * element.bounds.height,
                    "criteria": _criteria(element),
                }
            )

        print(
            f"[{title:<8}] 行 {len(screen):>3}  "
            + "  ".join(
                f"{n}:裁{cropped[n] - cropped[n] + len([e for e in screen if _criteria(e)[n]])}"
                for n in names
            )
        )

    OUT.write_text(
        json.dumps(
            {
                "pages": pages_used,
                "total_rows": total_rows,
                "tok_before": tok_before,
                "tok_after": tok_after,
                "cropped": cropped,
                "cropped_with_id": cropped_with_id,
                "rows": detail,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(f"{pages_used} 个页面，共 {total_rows} 行，元素表 {tok_before} tok")
    print(f"{'判据':<8}{'裁掉行':>8}{'其中有id':>10}{'剩 tok':>10}{'省':>8}{'占比':>8}")
    for name in names:
        after = tok_after[name]
        print(
            f"{name:<8}{cropped[name]:>8}{cropped_with_id[name]:>10}{after:>10}"
            f"{tok_before - after:>8}{(tok_before - after) / tok_before:>8.0%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
