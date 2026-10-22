"""**每个动作步的开销分解** —— 墙钟时间到底花在哪个节点上。

为什么单独测
------------
真机第七轮：254.4s / 46 次 LLM 调用 ⇒ LLM 占了 167s（66%），
剩下 87s（34%）在**非 LLM** 部分。87s 落在 27 个动作步上 = **3.2s/步**。

这 3.2s 里有：
* **抓屏 + 建元素表**（contextor）—— 实测中位 ~0.8s
* **等屏幕沉降**（contextor 里的 ``settle_delay_ms`` 下限）—— ⚠️ 可疑
* **工具执行**（executor_tools：adb tap/swipe + 等 UI 响应）

``settle_delay_ms`` 默认 3000ms，它的语义是**下限**：距上次动屏不足 3 秒就
**补等**。而决策本身要 4 秒多，于是「工具执行完 → 下次抓屏」这一小段
（通常 1 秒左右）就会被**补到 3 秒** ⇒ **每步白等约 2 秒**。
27 步 × 2s = 54s ≈ 总时间的 21% —— **而这只是一个配置项**。

本探针把每个节点的墙钟耗时逐次记下来，用实测数据确认这个账。

跑法（要真机在线）：
    uv run python scripts/audit_step_overhead.py
"""

from __future__ import annotations

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

TIMES: dict[str, list[float]] = {}
SETTLE_WAITS: list[float] = []


def _install_timers() -> None:
    """给**所有**节点包一层计时。

    ⚠️ `ExecutorToolNode` 继承自 LangGraph 的 `ToolNode`，它走的是
    ``invoke`` / ``ainvoke``，**没有** ``__call__`` —— 包错方法会静默
    漏掉整个节点（踩过：上一版只包 ``__call__``，工具执行的时间全丢了）。
    """
    from minitap.mobile_use.agents.contextor.contextor import ContextorNode
    from minitap.mobile_use.agents.cortex.cortex import CortexNode
    from minitap.mobile_use.agents.executor.executor import ExecutorNode
    from minitap.mobile_use.agents.executor.tool_node import ExecutorToolNode
    from minitap.mobile_use.agents.orchestrator.orchestrator import OrchestratorNode
    from minitap.mobile_use.agents.planner.planner import PlannerNode
    from minitap.mobile_use.agents.reporter.reporter import ReporterNode
    from minitap.mobile_use.agents.summarizer.summarizer import SummarizerNode

    def wrap_call(cls, label):
        orig = cls.__call__

        async def timed(self, *args, **kwargs):
            t0 = time.perf_counter()
            try:
                return await orig(self, *args, **kwargs)
            finally:
                TIMES.setdefault(label, []).append((time.perf_counter() - t0) * 1000)

        cls.__call__ = timed  # type: ignore[method-assign]

    def wrap_invoke(cls, label):
        """ToolNode 走 invoke/ainvoke。"""
        orig_invoke = cls.invoke
        orig_ainvoke = cls.ainvoke

        def timed_invoke(self, *args, **kwargs):
            t0 = time.perf_counter()
            try:
                return orig_invoke(self, *args, **kwargs)
            finally:
                TIMES.setdefault(label, []).append((time.perf_counter() - t0) * 1000)

        async def timed_ainvoke(self, *args, **kwargs):
            t0 = time.perf_counter()
            try:
                return await orig_ainvoke(self, *args, **kwargs)
            finally:
                TIMES.setdefault(label, []).append((time.perf_counter() - t0) * 1000)

        cls.invoke = timed_invoke  # type: ignore[method-assign]
        cls.ainvoke = timed_ainvoke  # type: ignore[method-assign]

    wrap_call(PlannerNode, "planner（拆计划，LLM）")
    wrap_call(OrchestratorNode, "orchestrator（验收，LLM）")
    wrap_call(ContextorNode, "contextor（抓屏+建表+沉降等待）")
    wrap_call(CortexNode, "cortex（判汇报/下发目标，LLM）")
    wrap_call(ExecutorNode, "executor（LLM 决策）")
    wrap_invoke(ExecutorToolNode, "executor_tools（工具执行）")
    wrap_call(ReporterNode, "reporter（汇报，默认免 LLM）")
    wrap_call(SummarizerNode, "summarizer（裁剪历史）")

    # 沉降等待：`_wait_for_settle` **返回实际等待的秒数**（0 = 没等），
    # 直接记录它，就得到「把 settle_delay_ms 调成 0 能省多少」的上界 ——
    # 不需要跑多轮 A/B（实测方差 >5.6 倍，单次 A/B 根本测不出这个量级的效应）。
    orig_settle = ContextorNode._wait_for_settle

    async def settle_timed(self, state):
        waited = await orig_settle(self, state)
        SETTLE_WAITS.append(float(waited or 0.0))
        return waited

    ContextorNode._wait_for_settle = settle_timed  # type: ignore[method-assign]


async def main() -> int:
    _install_timers()

    from minitap.mobile_use.config import initialize_llm_config
    from minitap.mobile_use.sdk import Agent
    from minitap.mobile_use.sdk.builders import Builders
    from minitap.mobile_use.sdk.types.agent import AgentProfile

    llm_config = initialize_llm_config()
    profile = AgentProfile(name="default", llm_config=llm_config)
    config = Builders.AgentConfig.with_default_profile(profile=profile)
    agent = Agent(config=config.build())
    await agent.init(retry_count=5, retry_wait_seconds=2)

    goal = "打开设置，进入 WLAN 页面，然后返回设置主页，再进入蓝牙页面"
    print(f"[probe] 真机跑一轮，测每步开销: {goal}")
    started = time.perf_counter()
    error = None
    try:
        await agent.run_task(request=agent.new_task(goal).build())
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        await agent.clean()
    wall = time.perf_counter() - started

    print()
    print("=" * 84)
    print(f"墙钟 {wall:.1f}s   错误: {error}")
    print("=" * 84)
    total_ms = 0.0
    for label, samples in TIMES.items():
        if not samples:
            continue
        med = statistics.median(samples)
        total_ms += sum(samples)
        print(
            f"  {label:<36} n={len(samples):>3}  中位 {med:>7.0f} ms  "
            f"合计 {sum(samples) / 1000:>6.1f}s  ({sum(samples) / wall * 100:>4.1f}%)"
        )
    print(f"  {'（节点耗时合计）':<36} {'':>3}  {'':>11}  {total_ms / 1000:>6.1f}s")

    # 沉降等待的账：这是「把 settle_delay_ms 调成 0」能省掉的**上界**
    if SETTLE_WAITS:
        waited_total = sum(SETTLE_WAITS)
        waited_n = sum(1 for w in SETTLE_WAITS if w > 0)
        print()
        print("  🟡 沉降等待（settle_delay_ms 的补等）:")
        print(
            f"     共 {len(SETTLE_WAITS)} 次抓屏，其中 {waited_n} 次真等了，"
            f"合计 **{waited_total:.1f}s**"
        )
        print(
            f"     ⇒ 把 settle_delay_ms 调成 0，最多省 **{waited_total:.1f}s**"
            f"（占墙钟 {waited_total / wall * 100:.1f}%）"
        )
        if waited_n:
            print(
                f"     每次平均等 {waited_total / waited_n:.2f}s"
                f"（上限是 3.0s = 默认 settle_delay_ms）"
            )

    out = ROOT / "probe_out" / "step_overhead.json"
    out.write_text(
        json.dumps(
            {
                "wall_s": round(wall, 1),
                "error": error,
                "nodes": {
                    k: {
                        "n": len(v),
                        "median_ms": round(statistics.median(v)),
                        "total_s": round(sum(v) / 1000, 1),
                    }
                    for k, v in TIMES.items()
                },
                "settle_waits": {
                    "n": len(SETTLE_WAITS),
                    "waited_n": sum(1 for w in SETTLE_WAITS if w > 0),
                    "total_s": round(sum(SETTLE_WAITS), 1),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n=> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
