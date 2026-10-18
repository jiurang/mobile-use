"""量一次真实 Cortex prefill 的构成 + 本地感知 I/O 耗时。

为什么需要这个
--------------
Cortex 是每一轮都跑的热节点，但它的时间构成从来没被拆开量过：
system prompt / 历史 thoughts / 层级树 / OCR / 截图 各占多少，只有实测才知道。
拍脑袋优化 prefill 很容易砍错地方（砍掉 14% 的 OCR，省不下 14% 的时间）。

量到什么
--------
A. 本地 I/O（Contextor 阶段，真机串行）：
   screenshot / dump_hierarchy / XML 解析 / OCR 推理，四段分别计时。
B. Cortex prefill 构成（tiktoken o200k_base）：
   system prompt / device info / thoughts / 层级树渲染 / OCR 渲染，
   外加截图按 tile 规则折算的 image token。

注意
----
thoughts 用的是**估算**：仓库里没有留存的历史 state，真跑一轮的 thought
长度无从取。这里按典型长度合成，标记为 ESTIMATED，不要当实测引用。
层级树 / OCR / system prompt / 截图全是当前这一屏的实测值。

跑法：
    uv run python scripts/probe_cortex_prefill.py [device_serial]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "minitap"))

import tiktoken

from minitap.mobile_use.clients.ui_automator_client import (
    UIAutomatorClient,
    _parse_hierarchy_xml_to_elements,
)
from minitap.mobile_use.services.ocr import OcrConfig, create_ocr_engine
from minitap.mobile_use.utils.image_compress import compress_screenshot_base64
from minitap.mobile_use.utils.ocr_view import render_ocr_results
from minitap.mobile_use.utils.ui_hierarchy_view import render_hierarchy_and_labels

ENC = tiktoken.get_encoding("o200k_base")

# image_compress.py 的实测结论：640x1536 上限 = 1x2 tile = 1548 image token。
# 换成别的 tile 数时按 (W/640)*(H/1536)*1548 折算。
IMG_TOKEN_PER_TILE = 1548


def ntok(text: str) -> int:
    return len(ENC.encode(text))


def timed(label: str, fn):
    started = time.perf_counter()
    result = fn()
    elapsed = (time.perf_counter() - started) * 1000
    print(f"  {label:<22} {elapsed:8.0f} ms")
    return result, elapsed


def main() -> int:
    serial = sys.argv[1] if len(sys.argv) > 1 else "89R8ZXNBGQIRJRLF"
    client = UIAutomatorClient(serial)
    device = client._ensure_connected()

    width, height = device.window_size()
    print(f"\n=== 设备 {serial}  {width}x{height} ===")

    # ---------- A. 本地 I/O ----------
    print("\n=== A. 本地感知 I/O（串行，真机）===")
    screenshot, t_shot = timed("device.screenshot()", device.screenshot)
    hierarchy_xml, t_dump = timed("dump_hierarchy()", lambda: device.dump_hierarchy(compressed=True))
    elements, t_parse = timed(
        "_parse_hierarchy_xml_to_elements", lambda: _parse_hierarchy_xml_to_elements(hierarchy_xml)
    )

    screenshot_b64 = client._pil_to_base64(screenshot) if hasattr(client, "_pil_to_base64") else None
    if screenshot_b64 is None:
        from minitap.mobile_use.clients.ui_automator_client import _pil_to_base64

        screenshot_b64 = _pil_to_base64(screenshot)

    ocr_results, ocr_ms = None, 0.0
    try:
        engine = create_ocr_engine(OcrConfig())
        ocr_results, ocr_ms = timed("OCR recognize (cached)", lambda: engine.recognize(screenshot_b64))
    except Exception as e:
        print(f"  OCR                    SKIPPED: {type(e).__name__}: {e}")

    local_ms = t_shot + t_dump + t_parse + ocr_ms
    print(f"  {'--- 本地合计':<20} {local_ms:8.0f} ms")

    # ---------- B. Cortex prefill 构成 ----------
    print("\n=== B. Cortex prefill 构成（tiktoken o200k_base）===")
    cortex_md = Path("minitap/mobile_use/agents/cortex/cortex.md").read_text(encoding="utf-8")
    system_prompt = cortex_md.replace("{{ platform }}", "android")
    system_tokens = ntok(system_prompt)

    tree_render, tree_labels = render_hierarchy_and_labels(elements)
    tree_tokens = ntok(tree_render)

    # OCR 不再去重：整块照渲染，[i] 就是 Target.ocr_index 的取值。
    # tree_labels 保留只为报告「有多少条其实元素表里已经有了」。
    ocr_render = render_ocr_results(ocr_results) if ocr_results else ""
    ocr_tokens = ntok(ocr_render) if ocr_render else 0
    redundant = 0
    if ocr_results and tree_labels:
        from minitap.mobile_use.utils.ui_hierarchy_view import _normalize_label

        redundant = sum(1 for t in ocr_results if _normalize_label(t.text) in tree_labels)

    compressed = compress_screenshot_base64(screenshot_b64)
    scaled_w, scaled_h = 640, min(1536, int(640 * height / width))
    img_tokens = IMG_TOKEN_PER_TILE * (1 if scaled_w <= 640 else 1) * max(
        1, -(-scaled_h // 1536)
    )

    # thoughts 是估算，见模块 docstring
    est_thought = (
        "Decisions reason: I will tap the search field at the top of the screen to focus it, "
        "then the keyboard should appear and I can type the query.\n\n"
        "Goals completion reason: none yet"
    )
    est_thought_tokens = ntok(est_thought) * 25

    rows = [
        ("system prompt (cortex.md)", system_tokens, "measured"),
        ("device info + date", ntok("Here are my device info:\nRedmi\nDevice date: 2026-10-17"), "measured"),
        ("thoughts x25 [ESTIMATED]", est_thought_tokens, "estimated"),
        ("UI hierarchy render", tree_tokens, "measured"),
        ("OCR render", ocr_tokens, "measured"),
        ("screenshot (image tokens)", img_tokens, "measured*"),
    ]
    for name, val, kind in rows:
        print(f"  {name:<30} {val:7d} tok   [{kind}]")

    text_total = system_tokens + tree_tokens + ocr_tokens + est_thought_tokens
    grand_total = text_total + img_tokens
    print(f"  {'--- 合计':<30} {grand_total:7d} tok")
    print(f"      其中 OCR 占比            {ocr_tokens / grand_total * 100:5.1f}%")
    print(f"      其中截图占比            {img_tokens / grand_total * 100:5.1f}%")
    print(f"      其中层级树占比          {tree_tokens / grand_total * 100:5.1f}%")

    print("\n=== OCR 块 ===")
    print(f"  整块渲染 {ocr_tokens:5d} tok ({len(ocr_results or [])} 条，全量不去重)")
    print(f"  其中 {redundant} 条文字在层级树里也出现过 —— 按旧逻辑会被剔掉，")
    print("  现在保留，因为 [i] 是 Target.ocr_index 的取值，跳号会让索引对不上。")

    print("\n=== C. 压缩后的截图 ===")
    print(f"  原图 {width}x{height} -> 压缩 {scaled_w}x{scaled_h}")
    print(f"  base64 {len(screenshot_b64) // 1024} KB -> {len(compressed) // 1024} KB")
    print(f"  估算 image token {img_tokens} (tile 规则, 实测基准 {IMG_TOKEN_PER_TILE}/tile)")

    # ---------- D. 层级树压缩比 ----------
    print("\n=== D. 层级树：原始 vs 裁剪 ===")
    raw_tokens = ntok(json.dumps(elements, ensure_ascii=False))
    print(f"  原始 {len(elements)} 节点 {raw_tokens} tok -> 裁剪 {tree_tokens} tok "
          f"({(1 - tree_tokens / raw_tokens) * 100:.1f}% 降幅)")

    out = {
        "serial": serial,
        "screen": f"{width}x{height}",
        "io_ms": {
            "screenshot": round(t_shot),
            "dump_hierarchy": round(t_dump),
            "parse_xml": round(t_parse),
            "ocr": round(ocr_ms),
            "local_total": round(local_ms),
        },
        "prefill_tokens": {
            "system": system_tokens,
            "thoughts_est": est_thought_tokens,
            "tree": tree_tokens,
            "ocr": ocr_tokens,
            "image": img_tokens,
            "total": grand_total,
        },
        "ocr_share_pct": round(ocr_tokens / grand_total * 100, 1),
        "ocr_count": len(ocr_results) if ocr_results else 0,
    }
    Path("probe_out/cortex_prefill.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=> probe_out/cortex_prefill.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
