"""把 DeepSeek 原始 API 探针的结果摊开，逐次打印 tool_call 的参数形态。"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "probe_out" / "deepseek_raw_probe.json").read_text(encoding="utf-8"))

print("\n=== 逐次 tool_call 形态 ===")
for s in d:
    print(
        f"  [{s['i']}] finish={s.get('finish_reason')}"
        f"  name={s.get('tc0_name')}"
        f"  arg_type={s.get('tc0_arg_type')}"
    )
    print(f"       args={str(s.get('tc0_args_preview'))[:150]}")

kinds = {}
for s in d:
    kinds[s.get("tc0_arg_type")] = kinds.get(s.get("tc0_arg_type"), 0) + 1
print(f"\n参数类型分布: {kinds}")
print(f"空 tool_calls: {sum(1 for s in d if not s.get('has_tool_calls'))}/{len(d)}")
print()
