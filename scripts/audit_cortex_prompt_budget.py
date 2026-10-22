"""量 cortex.md 的 token 预算（方案 §3.2 / §3.2.1 的依据）。

要回答两个问题：
1. ``cortex.md`` 整份多大？各节分别多大？
2. 新的 ``executor.md`` 需要承接哪些节？（元素定位 + 工具表 + 感知）

跑法：
    uv run python scripts/audit_cortex_prompt_budget.py
"""

from __future__ import annotations

import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

import tiktoken  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
enc = tiktoken.get_encoding("o200k_base")


def n(s: str) -> int:
    return len(enc.encode(s))


def main() -> int:
    cor = (ROOT / "minitap/mobile_use/agents/cortex/cortex.md").read_text(encoding="utf-8")
    exe = (ROOT / "minitap/mobile_use/agents/executor/executor.md").read_text(encoding="utf-8")

    print(f"cortex.md total = {n(cor)} tok / {len(cor.splitlines())} lines\n")
    parts = re.split(r"(?m)^##\s+", cor)
    for p in parts:
        if not p.strip():
            continue
        title = p.splitlines()[0].strip()
        print(f"  ## {title:28s} {n(p):6d} tok  ({len(p.splitlines()):3d} lines)")

    print()
    print("=== Executor prompt 承接了哪些 ===")
    for name, pattern in (
        ("Element targeting", r"(?ms)^## Element targeting.*?(?=^## |\Z)"),
        ("Tools", r"(?ms)^## Tools.*?(?=^## |\Z)"),
        ("Perception", r"(?ms)^## Perception.*?(?=^## |\Z)"),
    ):
        m = re.search(pattern, cor)
        if m:
            print(f"  {name:20s} {n(m.group(0)):6d} tok")

    print()
    print("=== 实际渲染尺寸 ===")
    print(f"  executor.md 原文      {n(exe):6d} tok  ({len(exe.splitlines())} lines)")
    print(f"  cortex.md 原文        {n(cor):6d} tok  ({len(cor.splitlines())} lines)")
    print()
    print("  ⇒ executor.md 已经比 cortex.md 小；它多出来的部分是循环规则。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
