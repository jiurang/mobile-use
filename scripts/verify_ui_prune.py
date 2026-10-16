"""Regression check for the shipped renderer (not the prototype).

Verifies on real device dumps:
  - token reduction vs. the legacy json.dumps(indent=2) payload
  - recall of actionable (clickable/long-clickable/focusable) nodes  -> must be 100%
  - recall of distinct text labels
  - the fallback fires on a non-Android element shape

Run: .venv\\Scripts\\python.exe scripts\\verify_ui_prune.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minitap"))

from mobile_use.clients.ui_automator_client import UIAutomatorClient  # noqa: E402
from mobile_use.clients.ui_automator_client import _parse_hierarchy_xml_to_elements  # noqa: E402
from mobile_use.utils.ui_hierarchy_view import render_ui_hierarchy  # noqa: E402

DEVICE = "89R8ZXNBGQIRJRLF"
CAPTURES = {
    "home": ["shell", "input", "keyevent", "3"],
    "settings": ["shell", "am", "start", "-a", "android.settings.SETTINGS"],
    "recents": ["shell", "input", "keyevent", "187"],
    "notif": ["shell", "cmd", "statusbar", "expand-notifications"],
    "douyin": [
        "shell",
        "monkey",
        "-p",
        "com.ss.android.ugc.aweme.lite",
        "-c",
        "android.intent.category.LAUNCHER",
        "1",
    ],
}

BOUNDS_RE = __import__("re").compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")


def count_tokens(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:
        return len(text) // 4


def box(raw: str):
    m = BOUNDS_RE.match((raw or "").strip())
    return tuple(map(float, m.groups())) if m else None


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def capture(name: str) -> str:
    subprocess.run(["adb", "-s", DEVICE, *CAPTURES[name]], capture_output=True, timeout=30)
    time.sleep(2.5)
    return UIAutomatorClient(device_id=DEVICE).get_hierarchy()


def main() -> None:
    hdr = (
        f"{'screen':10s} {'raw':>5s} {'rows':>5s} {'legacy':>8s} {'new':>7s} "
        f"{'saved':>7s} {'actionable':>10s} {'labels':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))

    failures: list[str] = []
    for name in CAPTURES:
        try:
            elements = _parse_hierarchy_xml_to_elements(capture(name))
        except Exception as exc:  # noqa: BLE001
            print(f"{name:10s} capture failed: {exc}")
            continue
        if not elements:
            print(f"{name:10s} empty dump, skipped")
            continue

        legacy = "Here is the UI hierarchy:\n" + json.dumps(elements, indent=2, ensure_ascii=False)
        new = render_ui_hierarchy(elements)
        lt, nt = count_tokens(legacy), count_tokens(new)
        rows = new.count("\n") - 1  # minus the 2-line header

        # actionable recall: geometry of the rendered rows vs. the raw dump
        want = [
            box(e.get("bounds") or "")
            for e in elements
            if e.get("enabled") == "true"
            and (
                e.get("clickable") == "true"
                or e.get("long-clickable") == "true"
                or e.get("focusable") == "true"
            )
        ]
        want = [w for w in want if w]
        got = [
            box(m.group(0))
            for m in __import__("re").finditer(r"\((-?\d+),(-?\d+),(\d+)x(\d+)\)", new)
        ]
        got = [
            (g[0], g[1], g[0] + g[2], g[1] + g[3])
            for g in [
                tuple(map(float, m.groups()))
                for m in __import__("re").finditer(r"\((-?\d+),(-?\d+),(\d+)x(\d+)\)", new)
            ]
        ]
        recall = (
            sum(1 for w in want if any(iou(w, g) > 0.5 for g in got)) / len(want) if want else 1.0
        )

        def lbl(e):
            return (e.get("text") or "").strip() or (e.get("content-desc") or "").strip()

        want_l = {lbl(e) for e in elements if lbl(e)}
        got_l = {
            json.loads(m.group(1))
            for m in __import__("re").finditer(r'^\[\d+\] ("(?:[^"\\]|\\.)*")', new, 8)
        }
        lab = len(want_l & got_l) / len(want_l) if want_l else 1.0

        print(
            f"{name:10s} {len(elements):5d} {rows:5d} {lt:8,} {nt:7,} "
            f"{(1 - nt / lt) * 100:6.1f}% {recall * 100:9.1f}% {lab * 100:6.1f}%"
        )

        if recall < 1.0:
            failures.append(f"{name}: actionable recall {recall:.3f} < 1.0")
        if nt > lt:
            failures.append(f"{name}: new payload ({nt}) larger than legacy ({lt})")

    # fallback must fire for a non-Android shape
    ios_like = [
        {
            "type": "Button",
            "label": "OK",
            "value": "",
            "frame": {"x": 1, "y": 2, "width": 3, "height": 4},
            "enabled": True,
            "visible": True,
        }
    ]
    out = render_ui_hierarchy(ios_like)
    fell_back = out.startswith("Here is the UI hierarchy:")
    print()
    print(f"non-Android shape falls back to legacy dump: {fell_back}")
    if not fell_back:
        failures.append("iOS-shaped input did not hit the fallback")

    # get_bounds_for_element must resolve real Android elements (string bounds)
    from mobile_use.utils.ui_hierarchy import get_bounds_for_element

    total = resolved = 0
    for name in CAPTURES:
        try:
            elements = _parse_hierarchy_xml_to_elements(capture(name))
        except Exception:  # noqa: BLE001
            continue
        for e in elements:
            if not e.get("bounds"):
                continue
            total += 1
            if get_bounds_for_element(e) is not None:
                resolved += 1
    print()
    print(f"real Android elements with resolvable bounds: {resolved}/{total}")
    if total and resolved != total:
        failures.append(f"get_bounds_for_element resolved only {resolved}/{total} real elements")

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
