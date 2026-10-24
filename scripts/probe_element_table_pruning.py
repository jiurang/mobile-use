"""真机实测：Executor 元素表**裁剪方案**的体积与安全性。

要回答的问题
------------
Executor 的缓存尾巴就是元素表（真机平均 ≈1,150 tok）。它每轮都变 ⇒
永远满价重算 ⇒ 命中率天花板 = (固定段+历史) / 总长。

所以「元素表能不能小一点」是**唯一真正有效的杠杆**。但裁剪有硬约束：

1. **不能漏掉可点元素** —— 漏了模型就点不到那个按钮（召回率必须 100%）；
2. **不能破坏索引契约** —— 提示词里的 `[i]` 和 ``resolve_screen_index``
   吃的是同一个下标。裁剪后必须让**两侧用同一份列表**（不是只裁提示词）。
3. **不能破坏位置语义** —— 行是按上→下、左→右排序的，模型靠顺序判断
   「这个按钮在底部」。

本脚本在**真机**上抓屏，走**生产渲染器**，对每屏量多个裁剪方案的：
行数 / token / 可交互元素召回率 / 剪枝后的索引契约是否还成立。

跑法::

    .venv\\Scripts\\python.exe scripts\\probe_element_table_pruning.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minitap"))
sys.stdout.reconfigure(encoding="utf-8")

from minitap.mobile_use.clients.ui_automator_client import (  # noqa: E402
    UIAutomatorClient,
)
from minitap.mobile_use.utils.screen_elements import (  # noqa: E402
    ScreenElement,
    build_screen_elements,
)
from minitap.mobile_use.utils.ui_hierarchy_view import normalize_hierarchy  # noqa: E402

DEVICE = "89R8ZXNBGQIRJRLF"

# 抓屏动作：覆盖「稀疏」到「密集」各种形态
CAPTURES = {
    "home": ["shell", "input", "keyevent", "3"],
    "notif": ["shell", "cmd", "statusbar", "expand-notifications"],
    "settings": ["shell", "am", "start", "-a", "android.settings.SETTINGS"],
    "wlan": ["shell", "am", "start", "-a", "android.settings.WIFI_SETTINGS"],
    "recents": ["shell", "input", "keyevent", "187"],
}

HEADER = (
    'Screen elements, one per line: [i] "text" | id=<id tail> | Class | (x,y,WxH) | flags\n'
    "flags: C=clickable L=long-press F=focusable K=checkbox S=selected R=scrollable\n"
    "One numbering for all rows. Rows without id= are OCR text. id= is the tail of the\n"
    "resource-id (package omitted), shown only when the text is missing or repeats."
)


def count_tokens(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:
        return len(text) // 4


# --------------------------------------------------------------------------
# 裁剪方案
# --------------------------------------------------------------------------


@dataclass
class Variant:
    name: str
    drop_class: bool = False
    drop_bounds: bool = False
    only_useful: bool = False  # 只保留「可交互 or 有文字」的行


# 「可交互」的判定：flags 里含可点的标记。
INTERACTIVE = set("CLFKS")


def is_useful(e: ScreenElement) -> bool:
    """这个元素值得发给模型吗？

    * 有文字 → 值得（模型可能要点它、读它、判断位置）
    * 可交互 → 值得（能点的都是目标）
    * 其余（纯装饰：分隔线、图标底板、Layout、无字 TextView）→ 不值得
    """
    if e.label:
        return True
    return bool(set(e.flags or "") & INTERACTIVE)


def render(elements: list[ScreenElement], v: Variant) -> str:
    """按方案渲染（**保留原 index**，所以工具侧仍能用同一个下标解析）。"""
    lines = []
    for e in elements:
        if v.only_useful and not is_useful(e):
            continue
        parts: list[str] = []
        if e.label:
            parts.append(f'"{e.label}"')
        if e.display_id:
            parts.append(f"id={e.display_id}")
        if e.class_name and not v.drop_class:
            parts.append(e.class_name)
        if not v.drop_bounds:
            parts.append(
                f"({e.bounds.x},{e.bounds.y},{e.bounds.width}x{e.bounds.height})"
            )
        if e.flags:
            parts.append(e.flags)
        if e.kind == "ocr" and e.confidence is not None:
            parts.append(f"{e.confidence:.2f}")
        lines.append(f"[{e.index}] " + " ".join(parts))
    return "\n".join(lines)


VARIANTS = [
    Variant("V0 现状"),
    Variant("V1 去Class", drop_class=True),
    Variant("V2 去Class+bounds", drop_class=True, drop_bounds=True),
    Variant("V3 只留有用行", only_useful=True),
    Variant("V4 去Class+只留有用行", drop_class=True, only_useful=True),
    Variant("V5 去Class+bounds+只留有用行", drop_class=True, drop_bounds=True, only_useful=True),
]


def capture(name: str) -> str:
    cmd = ["adb", "-s", DEVICE, *CAPTURES[name]]
    proc = subprocess.run(cmd, capture_output=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"adb 失败 rc={proc.returncode}: {proc.stderr[:120]!r}")
    time.sleep(3.0)
    return UIAutomatorClient(device_id=DEVICE).get_hierarchy()


def _parse(xml: str):
    from minitap.mobile_use.clients.ui_automator_client import (
        _parse_hierarchy_xml_to_elements,
    )

    return _parse_hierarchy_xml_to_elements(xml)


def main() -> int:
    print("=" * 108)
    print("真机实测：Executor 元素表裁剪方案对比")
    print("=" * 108)
    hdr_tok = count_tokens(HEADER)
    print(f"  表头固定部分: {hdr_tok} tok（所有方案都一样，不参与对比）")
    print()

    screens: dict[str, list[ScreenElement]] = {}
    for name in CAPTURES:
        try:
            xml = capture(name)
            tree_rows = normalize_hierarchy(_parse(xml))
            elements = build_screen_elements(tree_rows=tree_rows, ocr_results=None)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: 抓屏/解析失败 {exc}")
            continue
        if elements:
            screens[name] = elements
            print(f"  抓到 {name}: {len(elements)} 行")

    if not screens:
        print("  没抓到任何屏")
        return 1

    print()
    print("=" * 108)
    print("① 体积（每屏 token / 平均 token）")
    print("=" * 108)
    print(f"  {'方案':<34}" + "".join(f"{n:>12}" for n in screens) + f"{'平均':>12}")
    tok_table: dict[str, list[int]] = {}
    for v in VARIANTS:
        toks = []
        for name, els in screens.items():
            t = count_tokens(render(els, v))
            toks.append(t)
            tok_table.setdefault(v.name, []).append(t)
        avg = sum(toks) / len(toks)
        print(f"  {v.name:<34}" + "".join(f"{t:>12,}" for t in toks) + f"{avg:>12,.0f}")
    base = sum(tok_table[VARIANTS[0].name]) / len(screens)

    print()
    print("=" * 108)
    print("② 安全性：可交互元素召回率（必须 100%）")
    print("=" * 108)
    print(f"  {'方案':<34}{'可交互元素数':>14}{'召回':>10}{'漏掉的':>10}")
    for v in VARIANTS:
        tot = 0
        kept = 0
        missed: list[str] = []
        for name, els in screens.items():
            for e in els:
                if not (set(e.flags or "") & INTERACTIVE):
                    continue
                tot += 1
                if not v.only_useful or is_useful(e):
                    kept += 1
                else:
                    missed.append(f"{name}:{e.index}:{e.label or e.resource_id}")
        rate = kept / tot * 100 if tot else 100.0
        print(f"  {v.name:<34}{tot:>14}{rate:>9.1f}%{len(missed):>10}")
        if missed:
            for m in missed[:3]:
                print(f"       漏: {m}")

    print()
    print("=" * 108)
    print("③ 索引契约：裁剪后提示词里的 [i] 能否仍解析到同一个元素")
    print("=" * 108)
    for v in VARIANTS:
        ok = True
        detail = ""
        for name, els in screens.items():
            by_index = {e.index: e for e in els}
            for line in render(els, v).splitlines():
                idx = int(line[1 : line.index("]")])
                if idx not in by_index:
                    ok = False
                    detail = f"{name} 的 {idx} 不在原表里"
                    break
            if not ok:
                break
        print(f"  {v.name:<34}{'✓ 仍能逐个解析到原元素' if ok else '✗ ' + detail}")

    print()
    print("=" * 108)
    print("④ 命中率天花板（固定段+历史 ≈1,100 tok；表头另计）")
    print("=" * 108)
    print(f"  {'方案':<34}" + "".join(f"{'历史'+str(h):>12}" for h in (0, 500, 1500, 3000)))
    for v in VARIANTS:
        avg = sum(tok_table[v.name]) / len(screens)
        row = f"  {v.name:<34}"
        for h in (0, 500, 1500, 3000):
            prefix = 1100 + h
            total = prefix + avg + hdr_tok
            row += f"{prefix / total:>11.1%} "
        print(row)
    print()
    print(f"  （V0 平均 {base:,.0f} tok）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
