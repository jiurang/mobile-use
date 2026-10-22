"""复现前缀缓存布局的影响（方案 §4.1.1 的依据）。

要证明的：**每轮变的内容放在中间，会把它之后的全部内容每轮重算**。

模拟两种布局的逐轮缓存命中：
* ``plan``        —— [静态][目标][元素表][历史]   ← 方案原版（元素表在历史前）
* ``history-first`` —— [静态][目标][历史][元素表]   ← 修正版（元素表在最后）

跑法：
    uv run python scripts/audit_executor_cache_layout.py
"""

from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")

# 实测尺寸（tiktoken o200k_base，见 audit_cortex_prompt_budget.py）
STATIC = 651  # system + 文本工具表
GOAL = 30  # 目标陈述
ELEMENTS = 96  # 屏幕元素表（每轮变）
HISTORY_PER_STEP = 90  # 动作 + 工具反馈


def lcp(a: list[str], b: list[str]) -> int:
    """最长公共前缀长度。"""
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def build(layout: str, rounds: int) -> list[list[str]]:
    static = [f"s{i}" for i in range(STATIC)]
    goal = [f"g{i}" for i in range(GOAL)]
    prompts: list[list[str]] = []
    for n in range(1, rounds + 1):
        elements = [f"E{n}-{i}" for i in range(ELEMENTS)]
        history: list[str] = []
        for i in range(1, n):
            history += [f"action{i}-{j}" for j in range(45)]
            history += [f"feedback{i}-{j}" for j in range(45)]
        if layout == "plan":
            prompt = static + goal + elements + history
        else:
            prompt = static + goal + history + elements
        prompts.append(prompt)
    return prompts


def main() -> int:
    for layout in ("plan", "history-first"):
        prompts = build(layout, 9)
        total_p = sum(len(p) for p in prompts[1:])
        total_h = sum(lcp(prompts[i - 1], prompts[i]) for i in range(1, len(prompts)))
        label = "原版（元素表在历史前）" if layout == "plan" else "修正版（元素表在最后）"
        print(f"--- {label} ---")
        for i in range(1, len(prompts)):
            cached = lcp(prompts[i - 1], prompts[i])
            print(
                f"  轮 {i + 1}: prompt={len(prompts[i]):5d}  cached={cached:5d}  "
                f"({cached / len(prompts[i]):.0%})"
            )
        print(f"  合计 prompt={total_p}  cached={total_h}  命中率={total_h / total_p:.1%}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
