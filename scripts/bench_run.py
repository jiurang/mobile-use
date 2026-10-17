"""mobile-use 端到端 A/B 基准：量一次真实 goal 的每节点耗时与 LLM 调用构成。

为什么不用 TUI 的 node_start
---------------------------
TUI 的 NODE_START/NODE_END 是交互式的，headless 跑拿不到结构化数据。这里改成
在脚本里直接给每个 agent 节点套一层计时包装（**只在本脚本内 monkeypatch，
不动生产代码**），再挂一个 LangChain callback 数 LLM 调用次数。

量到什么
--------
1. 每个节点的墙上耗时（cortex / contextor / executor / planner / orchestrator）
2. LLM 调用次数，按 agent 名分组 —— 优化前后 executor 的调用数应该直接对半砍
3. `executor_dispatch` 命中/退回次数 —— 快路径的真实命中率

用法：
    uv run python scripts/bench_run.py --goal "..." --tag optimized
    # 对照组：先 git stash 掉改动，再
    uv run python scripts/bench_run.py --goal "..." --tag baseline

结果写到 probe_out/bench_<tag>.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MOBILE_USE_TELEMETRY_ENABLED", "false")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "minitap"))

from langchain_core.callbacks import AsyncCallbackHandler  # noqa: E402

from minitap.mobile_use.agents.contextor.contextor import ContextorNode  # noqa: E402
from minitap.mobile_use.agents.cortex.cortex import CortexNode  # noqa: E402
from minitap.mobile_use.agents.executor.executor import ExecutorNode  # noqa: E402
from minitap.mobile_use.agents.orchestrator.orchestrator import OrchestratorNode  # noqa: E402
from minitap.mobile_use.agents.planner.planner import PlannerNode  # noqa: E402
from minitap.mobile_use.config import initialize_llm_config  # noqa: E402
from minitap.mobile_use.sdk import Agent  # noqa: E402
from minitap.mobile_use.sdk.builders import Builders  # noqa: E402
from minitap.mobile_use.sdk.types.agent import AgentProfile  # noqa: E402
from minitap.mobile_use.services import telemetry as telemetry_mod  # noqa: E402

NODE_CLASSES = {
    "planner": PlannerNode,
    "orchestrator": OrchestratorNode,
    "contextor": ContextorNode,
    "cortex": CortexNode,
    "executor": ExecutorNode,
}

STATS: dict = {"nodes": {}, "llm_calls": [], "dispatch": {"hit": 0, "miss": 0}}


def install_node_timers() -> None:
    """给每个节点套计时。只在本进程内生效。"""

    def wrap(cls, label):
        original = cls.__call__

        async def timed(self, state):
            started = time.perf_counter()
            try:
                return await original(self, state)
            finally:
                ms = (time.perf_counter() - started) * 1000
                entry = STATS["nodes"].setdefault(label, [])
                entry.append(ms)

        cls.__call__ = timed

    for label, cls in NODE_CLASSES.items():
        wrap(cls, label)


class LlmCounter(AsyncCallbackHandler):
    """数 LLM 调用。不碰内容，只记次数和所属 agent。"""

    async def on_chat_model_start(self, serialized, messages, **kwargs):
        STATS["llm_calls"].append(
            {
                "model": (serialized or {}).get("name") or (serialized or {}).get("id"),
                "run_id": kwargs.get("run_id"),
                "tags": [t for t in (kwargs.get("tags") or []) if t],
            }
        )


def install_dispatch_counter() -> None:
    """统计确定性直连的命中/退回。

    baseline（改动 stash 掉）里没有这个方法，那时统计恒为 0 —— 对照组本来
    就每次都走 LLM，0/0 是正确的语义，不要为了跑通而硬造一个。
    """
    original = getattr(telemetry_mod.telemetry, "capture_executor_dispatch", None)
    if original is None:
        STATS["dispatch"]["note"] = "baseline: 没有直连事件，全部走 LLM"
        return

    def counting(task_id, dispatched):
        STATS["dispatch"]["hit" if dispatched else "miss"] += 1
        return original(task_id=task_id, dispatched=dispatched)

    telemetry_mod.telemetry.capture_executor_dispatch = counting


async def run(goal: str, tag: str) -> dict:
    install_node_timers()
    install_dispatch_counter()

    llm_config = initialize_llm_config()
    profile = AgentProfile(name="default", llm_config=llm_config)
    config = Builders.AgentConfig.with_default_profile(profile=profile)
    config.with_graph_config_callbacks([LlmCounter()])

    started = time.perf_counter()
    error = None
    agent = None
    try:
        agent = Agent(config=config.build())
        await agent.init(retry_count=5, retry_wait_seconds=2)
        task = agent.new_task(goal)
        await agent.run_task(request=task.build())
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    finally:
        if agent is not None:
            await agent.clean()
    total_ms = (time.perf_counter() - started) * 1000

    nodes = {}
    for label, samples in STATS["nodes"].items():
        samples_sorted = sorted(samples)
        nodes[label] = {
            "calls": len(samples),
            "total_ms": round(sum(samples)),
            "mean_ms": round(sum(samples) / len(samples)),
            "median_ms": round(samples_sorted[len(samples_sorted) // 2]),
            "max_ms": round(samples_sorted[-1]),
        }

    llm_total = len(STATS["llm_calls"])
    result = {
        "tag": tag,
        "goal": goal,
        "total_ms": round(total_ms),
        "error": error,
        "nodes": nodes,
        "llm_calls_total": llm_total,
        "llm_calls_per_iteration": (
            round(llm_total / nodes["cortex"]["calls"], 2) if nodes.get("cortex", {}).get("calls") else None
        ),
        "dispatch": STATS["dispatch"],
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", required=True)
    parser.add_argument("--tag", default="run")
    args = parser.parse_args()

    result = asyncio.run(run(args.goal, args.tag))

    out = ROOT / "probe_out" / f"bench_{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"[{result['tag']}] {result['goal']}")
    print(f"  总耗时      {result['total_ms'] / 1000:.1f} s")
    if result["error"]:
        print(f"  ⚠ 错误: {result['error']}")
    print(f"{'节点':<14}{'次数':>6}{'总耗时(s)':>12}{'单次均值(ms)':>16}{'中位(ms)':>10}")
    print("-" * 62)
    for label, stat in result["nodes"].items():
        print(
            f"{label:<14}{stat['calls']:>6}{stat['total_ms'] / 1000:>12.1f}"
            f"{stat['mean_ms']:>16}{stat['median_ms']:>10}"
        )
    print("-" * 62)
    print(f"  LLM 调用总数     {result['llm_calls_total']}")
    print(f"  每轮 LLM 调用数   {result['llm_calls_per_iteration']}")
    print(f"  直连 命中/退回    {result['dispatch']['hit']} / {result['dispatch']['miss']}")
    print(f"  => {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
