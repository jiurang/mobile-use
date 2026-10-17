"""量 Cortex prefill 里最重的两项：层级树渲染 和 OCR 渲染。

probe_cortex_prefill.py 已经量到：
    UI 树 3288 tok (40.1%)  <- prefill 最大头
    OCR     522 tok ( 6.4%)
    截图   1548 tok (18.9%)

这里往下钻两个问题，决定到底能砍哪一块：

1. **树的 token 花在哪个字段上？** 每行是 `"label" | id=xxx | Class | (x,y,WxH) | flags`。
   如果 `id=` 全名占了大头、而模型极少用 id 定位，那就是明确的压缩空间
   （但 ui_hierarchy_view.py 的不变量 #2 明确禁止截断 resource-id —— 所以要
   先知道数字，再决定是砍字段还是砍行数）。

2. **OCR 有多少字是树里已经有的？** ocr_view.py 的 docstring 说 OCR 换来的是
   「层级树完全没有的文本」。那反过来：OCR 里有多少条其实是**重复**的？
   重复的那部分可以整块砍掉，模型一点信息都不损失。

跑法：
    uv run python scripts/probe_cortex_field_fat.py [device_serial]
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "minitap"))

import tiktoken

from minitap.mobile_use.clients.ui_automator_client import (
    UIAutomatorClient,
    _parse_hierarchy_xml_to_elements,
    _pil_to_base64,
)
from minitap.mobile_use.services.ocr import OcrConfig, create_ocr_engine
from minitap.mobile_use.utils.ocr_view import MAX_TEXT_CHARS, render_ocr_results
from minitap.mobile_use.utils.ui_hierarchy_view import _normalize, _prune, render_ui_hierarchy

ENC = tiktoken.get_encoding("o200k_base")


def ntok(text: str) -> int:
    return len(ENC.encode(text)) if text else 0


def main() -> int:
    serial = sys.argv[1] if len(sys.argv) > 1 else "89R8ZXNBGQIRJRLF"
    client = UIAutomatorClient(serial)
    device = client._ensure_connected()
    screenshot = device.screenshot()
    hierarchy_xml = device.dump_hierarchy(compressed=True)
    elements = _parse_hierarchy_xml_to_elements(hierarchy_xml)

    engine = create_ocr_engine(OcrConfig())
    t0 = time.perf_counter()
    ocr_results = engine.recognize(_pil_to_base64(screenshot))
    ocr_ms = (time.perf_counter() - t0) * 1000

    # ---------- 1. 树的 token 构成 ----------
    print(f"\n=== 1. 层级树渲染构成 ===")
    nodes = [n for n in (_normalize(e) for e in elements) if n is not None]
    pruned = _prune(nodes)
    lines = render_ui_hierarchy(elements).split("\n")[1:]  # 去掉 header
    print(f"  原始节点 {len(elements)} -> 归一化 {len(nodes)} -> 最终渲染 {len(pruned)} 行")

    # 逐字段累加 token：把每行按字段切开，分别量
    field_cost = {"label": 0, "id": 0, "class": 0, "box": 0, "flags": 0, "index": 0}
    has = {"label": 0, "id": 0, "class": 0, "box": 0, "flags": 0}
    for i, node in enumerate(pruned, 1):
        field_cost["index"] += ntok(f"[{i}] ")
        if node.label:
            has["label"] += 1
            field_cost["label"] += ntok(json.dumps(node.label, ensure_ascii=False) + " ")
        if node.rid:
            has["id"] += 1
            field_cost["id"] += ntok(f"id={node.rid} ")
        if node.cls:
            has["class"] += 1
            field_cost["class"] += ntok(node.cls + " ")
        if node.box:
            has["box"] += 1
            x1, y1, x2, y2 = node.box
            field_cost["box"] += ntok(f"({int(x1)},{int(y1)},{int(x2 - x1)}x{int(y2 - y1)}) ")
        if node.flags:
            has["flags"] += 1
            field_cost["flags"] += ntok(node.flags + " ")

    total_tree = ntok(render_ui_hierarchy(elements))
    print(f"  树总 token {total_tree}")
    for key in ("index", "label", "id", "class", "box", "flags"):
        cost = field_cost[key]
        pct = cost / total_tree * 100 if total_tree else 0
        print(f"    {key:<7} {cost:6d} tok ({pct:5.1f}%)  出现在 {has.get(key, len(pruned)):3d}/{len(pruned)} 行")

    # 交互性分布：这决定了能不能只留「可操作 + 有 label」的行
    interactive = sum(1 for n in pruned if n.interactive)
    labelled = sum(1 for n in pruned if n.label)
    both = sum(1 for n in pruned if n.interactive or n.label)
    print(f"\n  可操作 {interactive} 行 / 有 label {labelled} 行 / 二者之一 {both} 行 / 总 {len(pruned)}")
    print(f"  只剩「可操作 or 有 label」的 token: ", end="")
    kept = [n for n in pruned if n.interactive or n.label]
    print(sum(ntok(f"[{i}] ") for i, _ in enumerate(kept, 1)) + ntok("\n".join(lines[:0]) or "") + ntok(
        "\n".join(
            " ".join(
                p
                for p in [
                    json.dumps(n.label, ensure_ascii=False) if n.label else "",
                    f"id={n.rid}" if n.rid else "",
                    n.cls if n.cls else "",
                    (
                        f"({int(n.box[0])},{int(n.box[1])},{int(n.box[2] - n.box[0])}x{int(n.box[3] - n.box[1])})"
                        if n.box
                        else ""
                    ),
                    n.flags if n.flags else "",
                ]
                if p
            )
            for n in kept
        )
    ))

    # ---------- 2. OCR 与树的重复度 ----------
    print(f"\n=== 2. OCR 与层级树的重复度（OCR 推理 {ocr_ms:.0f}ms）===")
    ocr_texts = [r.text.strip() for r in ocr_results if r.text and r.text.strip()]
    tree_texts = set()
    for e in elements:
        for key in ("text", "content-desc"):
            v = e.get(key)
            if isinstance(v, str) and v.strip():
                tree_texts.add(v.strip())

    def norm(s: str) -> str:
        return re.sub(r"\s+", "", s)

    tree_norm = {norm(t) for t in tree_texts}
    redundant, unique = [], []
    for t in ocr_texts:
        n = norm(t)
        if n in tree_norm:
            redundant.append(t)
        elif any(n in tn or tn in n for tn in tree_norm if n and tn):
            redundant.append(t)
        else:
            unique.append(t)

    full_ocr_render = render_ocr_results(ocr_results)
    print(f"  OCR 共 {len(ocr_texts)} 条 -> 树里已有 {len(redundant)} 条 (重复) / 树里没有 {len(unique)} 条 (新增)")
    print(f"  OCR 全量 {ntok(full_ocr_render)} tok")
    print(f"  重复部分约 {ntok(chr(10).join(redundant))} tok（含坐标/置信度开销会更高）")

    print("\n  树里没有、只有 OCR 能提供的文本：")
    for t in unique:
        print(f"    + {t[:60]}")
    print("\n  树里已经有的（OCR 纯重复）：")
    for t in redundant[:12]:
        print(f"    - {t[:60]}")

    print(f"\n  树中 text/content-desc 节点数: {len(tree_texts)}")
    out = {
        "rendered_lines": len(pruned),
        "raw_nodes": len(elements),
        "tree_tokens": total_tree,
        "field_cost": field_cost,
        "interactive": interactive,
        "labelled": labelled,
        "ocr_total": len(ocr_texts),
        "ocr_redundant": len(redundant),
        "ocr_unique": len(unique),
        "ocr_tokens": ntok(full_ocr_render),
        "ocr_ms": round(ocr_ms),
        "unique_texts": unique,
        "redundant_texts": redundant,
    }
    Path("probe_out/field_fat.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=> probe_out/field_fat.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
