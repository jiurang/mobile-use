"""量真机上 **Executor 元素表**的真实体积（缓存尾巴的地面真相）。

为什么需要它
------------
缓存命中率 = 公共前缀 / 总长。Executor 的「易变尾巴」就是元素表 ——
它每轮都变 ⇒ 永远满价。所以「尾巴到底多大」直接决定命中率的天花板。

之前文档里的行数（5/30/60 → 190/665/1,235 tok）是**估算**。这份脚本直接
在真机上抓屏、走**生产渲染器**（``build_screen_elements`` +
``render_screen_rows`` + ``executor.md`` 的表头），量真值。

跑法::

    .venv\\Scripts\\python.exe scripts\\probe_real_element_table_size.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minitap"))
sys.stdout.reconfigure(encoding="utf-8")

from minitap.mobile_use.clients.ui_automator_client import (  # noqa: E402
    UIAutomatorClient,
)
from minitap.mobile_use.utils.screen_elements import (  # noqa: E402
    build_screen_elements,
    render_screen_rows,
)
from minitap.mobile_use.utils.ui_hierarchy_view import normalize_hierarchy  # noqa: E402

DEVICE = "89R8ZXNBGQIRJRLF"

CAPTURES = {
    "home": ["shell", "input", "keyevent", "3"],
    "recents": ["shell", "input", "keyevent", "187"],
    "settings": ["shell", "am", "start", "-a", "android.settings.SETTINGS"],
    "notif": ["shell", "cmd", "statusbar", "expand-notifications"],
    "back": ["shell", "input", "keyevent", "4"],
}


def count_tokens(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:
        return len(text) // 4


def capture(name: str) -> str:
    cmd = ["adb", "-s", DEVICE, *CAPTURES[name]]
    proc = subprocess.run(cmd, capture_output=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"adb 失败 rc={proc.returncode}: {proc.stderr[:120]!r}")
    time.sleep(3.0)
    return UIAutomatorClient(device_id=DEVICE).get_hierarchy()


def main() -> int:
    # executor.md 的表头（screen_elements_message 宏的固定部分）
    header = (
        'Screen elements, one per line: [i] "text" | id=<id tail> | Class | (x,y,WxH) | flags\n'
        "flags: C=clickable L=long-press F=focusable K=checkbox S=selected R=scrollable\n"
        "One numbering for all rows. Rows without id= are OCR text. id= is the tail of the\n"
        "resource-id (package omitted), shown only when the text is missing or repeats."
    )
    sys_tokens = count_tokens(header)
    print("=" * 92)
    print("真机元素表体积（Executor 的缓存尾巴）")
    print("=" * 92)
    print(f"  表头固定部分: {sys_tokens} tok")
    print()
    print(f"  {'屏幕':<10} {'原始节点':>8} {'入表行数':>8} {'表 tok':>8} {'字符':>8}  {'行均 tok':>7}")
    print("  " + "-" * 60)

    rows_all = []
    for name in CAPTURES:
        try:
            xml = capture(name)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<10} 抓屏失败: {exc}")
            continue
        if not xml:
            print(f"  {name:<10} 空 dump")
            continue
        try:
            tree_rows = normalize_hierarchy(_parse(xml))
            elements = build_screen_elements(tree_rows=tree_rows, ocr_results=None)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<10} 解析失败: {exc}")
            continue
        text = render_screen_rows(elements)
        if not text:
            print(f"  {name:<10} 表为空")
            continue
        tok = count_tokens(text)
        sample = " / ".join(text.splitlines()[:2])[:70]
        rows_all.append((name, len(elements), tok))
        print(
            f"  {name:<10} {len(tree_rows):>8} {len(elements):>8} {tok:>8,} "
            f"{len(text):>8,}  {tok / max(len(elements), 1):>7.1f}   {sample}"
        )

    if rows_all:
        print()
        avg_tok = sum(t for _, _, t in rows_all) / len(rows_all)
        print(f"  平均尾巴 ≈ {avg_tok:,.0f} tok（60 行表头+内容另计 {sys_tokens} tok）")
        print()
        print("  换算成命中率天花板（system+goal+历史 固定 ≈ 1,100 tok）：")
        for hist in (0, 500, 1_500, 3_000):
            prefix = 1_100 + hist
            total = prefix + avg_tok + sys_tokens
            print(f"    历史 {hist:>5,} tok 时: 命中率 ≈ {prefix / total:>5.1%}")
    return 0


def _parse(xml: str):
    from minitap.mobile_use.clients.ui_automator_client import (
        _parse_hierarchy_xml_to_elements,
    )

    return _parse_hierarchy_xml_to_elements(xml)


if __name__ == "__main__":
    raise SystemExit(main())
