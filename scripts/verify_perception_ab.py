"""**感知段优化的真机 A/B 验收**（2026-10-22）。

改了三件事，每一件都要用真机对一遍：

1. **沉降等待**：下限 3000ms → 200ms + 树稳定确认。
2. **前台包名**：从树里推，免掉一次串行 adb 往返（实测 127ms）。
3. **Executor 截图通道**：整条关掉（少了「同轮二问」那一次 LLM 调用）。

A/B 怎么排
----------
* **A 组（基线）** = ``git worktree`` 拉的一份 **HEAD 干净副本**
  （``.ab-baseline/``）—— 工作区一��不碰，所以**不会**把用户正在并行
  改的那些文件卷进来（``git stash`` 会，那很危险）。
* **B 组（新代码）** = 当前工作区。

同一个任务各跑 N 轮，比三件事：

* **墙钟** —— 提速的直接证据
* **子目标步数 / 失败目标数** —— 沉降提前会不会让模型拿到「残缺的表」，
  从而绕路多走几步。**这是沉降改动的风险面**（表和屏是同一次抓的，
  索引契约自洽，所以不会「点错」，只会「表里缺控件 ⇒ 模型以为按钮不存在」）
* **LLM 调用次数** —— 少了 Executor 的二问就该降

跑法（要真机在线）::

    uv run python scripts/verify_perception_ab.py --runs 2
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MOBILE_USE_TELEMETRY_ENABLED", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

BASELINE = ROOT / ".ab-baseline"
TASK = "打开设置，进入 WLAN 页面，然后返回设置主页，再进入蓝牙页面"


def _run_one(workdir: Path, tag: str) -> dict:
    """跑一轮真机任务，返回那一轮的 json。"""
    out = workdir / "probe_out" / "executor_e2e.json"
    if out.exists():
        out.unlink()

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"\n>>> [{tag}] 开始：{TASK}", flush=True)
    started = time.perf_counter()
    proc = subprocess.run(
        [
            sys.executable,
            str(workdir / "scripts" / "probe_executor_e2e.py"),
            "--goal",
            TASK,
        ],
        cwd=str(workdir),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )
    wall = time.perf_counter() - started
    print(f">>> [{tag}] 结束，墙钟 {wall:.1f}s，退出码 {proc.returncode}", flush=True)

    if not out.exists():
        print(f"!!! [{tag}] 没有产出 json。stderr 尾部:\n{proc.stderr[-800:]}", flush=True)
        return {"tag": tag, "wall_s": round(wall, 1), "error": "no output json"}

    data = json.loads(out.read_text(encoding="utf-8"))
    data["tag"] = tag
    data["outer_wall_s"] = round(wall, 1)
    return data


def _summarize(rows: list[dict], label: str) -> None:
    print()
    print("=" * 96)
    print(f"{label} 汇总")
    print("=" * 96)
    print(
        f"  {'轮次':<10}{'墙钟s':>9}{'LLM调用':>10}{'executor':>10}"
        f"{'子目标':>8}{'总步数':>9}{'重试':>7}{'失败目标':>10}"
    )
    for r in rows:
        by_agent = r.get("by_agent") or {}
        prog = r.get("subgoal_progress") or []
        steps = sum(int(p.get("steps", 0)) for p in prog)
        retries = sum(int(p.get("retry_count", 0)) for p in prog)
        failed = sum(1 for p in prog if p.get("outcome") in ("failed", "blocked"))
        print(
            f"  {r.get('tag', '?'):<10}{r.get('outer_wall_s', 0):>9.1f}"
            f"{r.get('llm_calls', 0):>10}{by_agent.get('executor', {}).get('n', 0):>10}"
            f"{len(prog):>8}{steps:>9}{retries:>7}{failed:>10}"
        )

    walls = [r.get("outer_wall_s", 0) for r in rows if r.get("outer_wall_s")]
    calls = [r.get("llm_calls", 0) for r in rows if r.get("llm_calls")]
    steps = [
        sum(int(p.get("steps", 0)) for p in (r.get("subgoal_progress") or [])) for r in rows
    ]
    if walls:
        print(
            f"\n  墙钟中位 {_med(walls):.1f}s   LLM 调用中位 {_med(calls):.0f}   "
            f"总步数中位 {_med(steps):.0f}"
        )


def _med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()

    if not BASELINE.exists():
        print(
            "基线 worktree 不存在。先跑：\n"
            "  git -C <repo> worktree add <repo>/.ab-baseline HEAD\n"
            "  Copy-Item <repo>/.env <repo>/.ab-baseline/.env"
        )
        return 1

    results: dict[str, list[dict]] = {"A_baseline": [], "B_new": []}

    # 先跑基线，再跑新代码 —— 顺序固定，避免设备状态漂移影响对比。
    for i in range(args.runs):
        results["A_baseline"].append(_run_one(BASELINE, f"A#{i + 1}"))
    for i in range(args.runs):
        results["B_new"].append(_run_one(ROOT, f"B#{i + 1}"))

    _summarize(results["A_baseline"], "A 组（基线 = HEAD 干净副本）")
    _summarize(results["B_new"], "B 组（新代码 = 当前工作区）")

    a_walls = [r.get("outer_wall_s", 0) for r in results["A_baseline"]]
    b_walls = [r.get("outer_wall_s", 0) for r in results["B_new"]]
    a_calls = [r.get("llm_calls", 0) for r in results["A_baseline"]]
    b_calls = [r.get("llm_calls", 0) for r in results["B_new"]]
    a_steps = [
        sum(int(p.get("steps", 0)) for p in (r.get("subgoal_progress") or []))
        for r in results["A_baseline"]
    ]
    b_steps = [
        sum(int(p.get("steps", 0)) for p in (r.get("subgoal_progress") or []))
        for r in results["B_new"]
    ]

    print()
    print("=" * 96)
    print("对比")
    print("=" * 96)
    if a_walls and b_walls:
        print(
            f"  墙钟     {(_med(a_walls) - _med(b_walls)):+.1f}s "
            f"（{(_med(b_walls) / _med(a_walls) - 1) * 100:+.1f}%）"
        )
        print(f"  LLM 调用 {_med(b_calls) - _med(a_calls):+.0f} 次")
        print(f"  总步数   {_med(b_steps) - _med(a_steps):+.0f} 步")
        print()
        print("  ⚠️ 步数**涨**就是沉降提前的代价（表里缺控件 ⇒ 模型绕路）。")
        print("     单轮对比不可靠 —— 实测同一任务的墙钟方差能到 5.6 倍。")

    out = ROOT / "probe_out" / "perception_ab.json"
    out.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n=> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
