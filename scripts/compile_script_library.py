"""把 manager-server 里已有的执行**批量**编译成脚本库（只读，不动任何东西）。

产出两样：
1. ``probe_out/script_library.json`` —— 可直接灌进 manager 的脚本库
2. 一份统计：多少步有稳定选择器、多少步只能靠 OCR（脆弱）、动作分布

统计的意义：它直接回答「这个方案值不值得做」——
如果 90% 的步都有稳定 resource_id，宏就有价值；
如果大部分步只能靠 OCR 文本（点赞数、广告标签），宏就救不了。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("cm", REPO / "scripts" / "compile_macro.py")
cm = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(cm)  # type: ignore[union-attr]

MANAGER = "http://127.0.0.1:9107"
OUT = REPO / "probe_out" / "script_library.json"


def list_executions(limit: int = 60) -> list[dict]:
    url = f"{MANAGER}/api/executions?size={limit}"
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)
    return data.get("items") or []


def main() -> int:
    execs = list_executions()
    print(f"manager 里有 {len(execs)} 次执行，开始编译……\n")

    library = []
    total_steps = 0
    total_with_id = 0
    action_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    app_counter: Counter[str] = Counter()
    per_exec = []

    for ex in execs:
        eid = ex.get("id")
        if eid is None:
            continue
        try:
            macro = cm.compile_macro(int(eid), MANAGER)
        except Exception as e:  # noqa: BLE001
            print(f"  执行 {eid}: 编译失败 {type(e).__name__}: {e}")
            continue
        steps = macro.get("steps") or []
        if not steps:
            continue
        with_id = sum(1 for s in steps if s.get("stable_selector"))
        total_steps += len(steps)
        total_with_id += with_id
        for s in steps:
            action_counter[s.get("action") or "?"] += 1
            source_counter[s.get("selector_source") or "?"] += 1
        pkg = macro.get("app_package") or "?"
        app_counter[pkg] += len(steps)

        name = f"exec{eid}"
        macro["name"] = name
        macro["description"] = f"从执行 {eid}（{ex.get('taskName') or '?'}）编译"
        macro["source_status"] = ex.get("status")
        macro["source_started_at"] = ex.get("startedAt")
        library.append(macro)
        per_exec.append((eid, ex.get("status"), len(steps), with_id, pkg))
        print(
            "  执行 %-4s %-9s 步数 %-3d 有稳定选择器 %-3d  app=%s"
            % (eid, ex.get("status"), len(steps), with_id, pkg)
        )

    OUT.write_text(json.dumps(library, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print("总计：%d 次执行编译出 %d 步，其中 %d 步有稳定 resource_id（%.0f%%）"
          % (len(library), total_steps, total_with_id, 100 * total_with_id / max(1, total_steps)))
    print()
    print("动作分布：", dict(action_counter.most_common()))
    print()
    print("选择器来源：")
    for k, v in source_counter.most_common():
        print("   %-26s %d" % (k, v))
    print()
    print("按 app：", dict(app_counter.most_common()))
    print()
    print("脚本库：", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
