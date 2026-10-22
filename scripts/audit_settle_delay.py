"""**沉降延迟（`settle_delay_ms`）对照实验** —— 提速最快的一刀，值不值得砍。

问题
----
真机第七轮：254.4s / 27 个动作步。**非 LLM 时间 87s（34%）**，落在每步
3.2s 上。按时序拆：

    executor 决策 4.1s → 工具执行 0.5s（动了屏）→ summarizer 0.05s
    → contextor：距上次动屏才 0.55s < 3s 下限 ⇒ **补等 2.45s** → 抓屏 0.8s

⇒ 每步 3.8s 的非 LLM 时间里，**2.45s 是纯等待**。27 步 ≈ **66 秒 = 总时间的 26%**。

实测佐证：`probe_out/wake_e2e.log` 里 "Waiting Xs for the screen to settle"
命中 3 次、**总计 6.8s、中位 3.0s**。

`settle_delay_ms` 的语义是**下限**（不是固定延迟）：距上次动屏不足这个数就补等。
它存在是为了别抓到「控件还没挂上」的中间态（MIUI 动画）。

所以要回答两个问题：
1. **能省多少？** （预期每步省 1.5~2.5s）
2. **会不会付出代价？** —— 抓早了的症状不是「点错」（元素表和屏是同一次抓的，
   索引契约自洽），而是「**表里缺控件**」⇒ 模型以为那个按钮不存在 ⇒ 绕路。
   所以要看的是**步数**和**失败目标数**有没有涨，不是「点错率」。

跑法（要真机在线，每个档位各跑 N 轮）：
    uv run python scripts/audit_settle_delay.py --levels 3000,1000,0 --runs 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("MOBILE_USE_TELEMETRY_ENABLED", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

RESULTS: list[dict] = []


async def one_run(settle_ms: int, goal: str) -> dict:
    from minitap.mobile_use.config import initialize_llm_config
    from minitap.mobile_use.context import PerceptionConfig
    from minitap.mobile_use.sdk import Agent
    from minitap.mobile_use.sdk.builders import Builders
    from minitap.mobile_use.sdk.types.agent import AgentProfile

    llm_config = initialize_llm_config()
    profile = AgentProfile(name="default", llm_config=llm_config)
    config = (
        Builders.AgentConfig.with_default_profile(profile=profile)
        .with_perception(PerceptionConfig(settle_delay_ms=settle_ms))
        .build()
    )
    agent = Agent(config=config)
    await agent.init(retry_count=5, retry_wait_seconds=2)
    started = time.perf_counter()
    error = None
    try:
        # 步数封顶：默认 80 个动作步（≈752 super-step）够一个跑飞的任务烧
        # 20+ 分钟（实测踩过：同一任务 4 分钟 vs >23 分钟，全看有没有进重发循环）。
        # 对照实验要的是**同一形状**的运行，不是跑满预算的那种。
        await agent.run_task(request=agent.new_task(goal).with_max_steps(30).build())
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        await agent.clean()
    wall = time.perf_counter() - started
    return {"settle_delay_ms": settle_ms, "wall_s": round(wall, 1), "error": error}


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--levels", default="3000,1000,0", help="逗号分隔的 settle_delay_ms 档位")
    p.add_argument("--runs", type=int, default=2, help="每档位跑几轮")
    p.add_argument("--goal", default="打开设置，进入 WLAN 页面")
    args = p.parse_args()

    levels = [int(x) for x in args.levels.split(",")]

    print("=" * 84)
    print(f"沉降延迟对照实验 —— 每个档位各跑 {args.runs} 轮")
    print("=" * 84)
    print(f"任务: {args.goal}\n")

    for level in levels:
        for i in range(args.runs):
            print(f"--- settle_delay_ms={level} 第 {i + 1}/{args.runs} 轮 ---")
            r = await one_run(level, args.goal)
            r["level_run"] = i + 1
            RESULTS.append(r)
            print(f"    墙钟 {r['wall_s']}s   错误: {r['error']}\n")

    print("=" * 84)
    print("汇总")
    print("=" * 84)
    print(f"  {'settle_delay_ms':>16} {'轮数':>5} {'中位墙钟':>10} {'最小':>8} {'最大':>8}")
    for level in levels:
        vals = [r["wall_s"] for r in RESULTS if r["settle_delay_ms"] == level]
        if not vals:
            continue
        print(
            f"  {level:>16} {len(vals):>5} "
            f"{statistics.median(vals):>9.1f}s {min(vals):>7.1f}s {max(vals):>7.1f}s"
        )

    base = [r["wall_s"] for r in RESULTS if r["settle_delay_ms"] == levels[0]]
    if base:
        b = statistics.median(base)
        print()
        for level in levels[1:]:
            vals = [r["wall_s"] for r in RESULTS if r["settle_delay_ms"] == level]
            if not vals:
                continue
            m = statistics.median(vals)
            print(
                f"  {level}ms 相对 {levels[0]}ms：**{(b - m) / b * 100:+.1f}%** "
                f"（省 {b - m:+.1f}s）"
            )

    out = ROOT / "probe_out" / "settle_delay_ab.json"
    out.write_text(json.dumps(RESULTS, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
