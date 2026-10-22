"""改造前基线：同任务、同设备、同模型，量 token 与耗时。

改造后的对应脚本是 `scripts/probe_executor_e2e.py`。
两边必须**同任务同模型**跑，否则比出来的收益是假的
（审计发现过原方案把「27 分钟的病态跑」和「109 秒的成功跑」拼在一起比）。

跑法：
    uv run python scripts/probe_baseline_before.py --goal "打开设置，进入 WLAN 页面"
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

from minitap.mobile_use.config import initialize_llm_config  # noqa: E402
from minitap.mobile_use.sdk import Agent  # noqa: E402
from minitap.mobile_use.sdk.builders import Builders  # noqa: E402
from minitap.mobile_use.sdk.types.agent import AgentProfile  # noqa: E402
from minitap.mobile_use.services import telemetry as telemetry_mod  # noqa: E402

CALLS: list[dict] = []


def _install_usage_probe() -> None:
    """拦截 telemetry 的 usage 埋点（成本核算的唯一数据源）。

    ⚠️ **不要**自己开捕获槽（`open_capture`）——
    `invoke_llm_with_timeout_message` 内部已经开了一个并在 finally 里
    `close_capture()`（services/llm.py:141），外层再开的会被抢先关掉。
    而且在 LangGraph 的图里那个槽**根本收不到东西**
    （ContextVar 跨不过图的任务上下文，services/llm.py:170-174）。
    """
    original = telemetry_mod.telemetry.capture_llm_usage

    def probe(
        task_id: str,
        agent: str,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        duration_seconds: float | None = None,
        used_fallback: bool = False,
    ) -> None:
        CALLS.append(
            {
                "agent": agent,
                "ms": round((duration_seconds or 0) * 1000),
                "input": input_tokens,
                "cached": cached_tokens,
                "output": output_tokens,
            }
        )
        return original(
            task_id=task_id,
            agent=agent,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            duration_seconds=duration_seconds,
            used_fallback=used_fallback,
        )

    telemetry_mod.telemetry.capture_llm_usage = probe


async def run(goal: str) -> int:
    _install_usage_probe()
    llm_config = initialize_llm_config()
    profile = AgentProfile(name="default", llm_config=llm_config)
    config = Builders.AgentConfig.with_default_profile(profile=profile)

    print("=" * 70)
    print(f"改造前基线: {goal}")
    print("=" * 70)

    started = time.perf_counter()
    error = None
    agent = None
    try:
        agent = Agent(config=config.build())
        await agent.init(retry_count=5, retry_wait_seconds=2)
        await agent.run_task(request=agent.new_task(goal).build())
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {str(e)[:400]}"
    finally:
        if agent is not None:
            await agent.clean()
    wall = time.perf_counter() - started

    by_agent: dict[str, dict] = {}
    for c in CALLS:
        a = by_agent.setdefault(c["agent"], {"n": 0, "in": 0, "ca": 0, "out": 0, "ms": []})
        a["n"] += 1
        a["in"] += c["input"]
        a["ca"] += c["cached"]
        a["out"] += c["output"]
        a["ms"].append(c["ms"])

    total_in = sum(a["in"] for a in by_agent.values())
    total_ca = sum(a["ca"] for a in by_agent.values())
    total_out = sum(a["out"] for a in by_agent.values())

    print()
    print("=" * 70)
    print(f"耗时 {wall:.1f}s   LLM 调用 {len(CALLS)} 次")
    if error:
        print(f"⚠ {error}")
    print("=" * 70)
    print(f"  input={total_in}  cached={total_ca}  output={total_out}  "
          f"缓存命中率={(total_ca/total_in*100 if total_in else 0):.1f}%")
    print()
    print(f"  {'agent':<14}{'次数':>5}{'input':>9}{'cached':>9}{'output':>9}{'中位ms':>9}")
    for k, v in sorted(by_agent.items(), key=lambda x: -x[1]["in"]):
        med = round(statistics.median(v["ms"])) if v["ms"] else 0
        print(f"  {k:<14}{v['n']:>5}{v['in']:>9}{v['ca']:>9}{v['out']:>9}{med:>9}")

    out = ROOT / "probe_out" / "executor_ab_before.json"
    out.write_text(
        json.dumps(
            {
                "tag": "before",
                "goal": goal,
                "wall_s": round(wall, 1),
                "error": error,
                "llm_calls": len(CALLS),
                "input": total_in,
                "cached": total_ca,
                "output": total_out,
                "by_agent": {
                    k: {kk: vv for kk, vv in v.items() if kk != "ms"}
                    for k, v in by_agent.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print()
    print(f"  => {out}")
    return 1 if error else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--goal", default="打开设置，进入 WLAN 页面")
    a = p.parse_args()
    return asyncio.run(run(a.goal))


if __name__ == "__main__":
    raise SystemExit(main())
