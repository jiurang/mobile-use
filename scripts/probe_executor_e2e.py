"""真机端到端：跑**完整的新架构**（走 graph），验证改造后的系统能真的完成任务。

这是最终验收 —— 前面所有探针都只驱动了部分节点。

跑法：
    uv run python scripts/probe_executor_e2e.py --goal "打开设置，进入 WLAN 页面"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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

CALLS: list[dict] = []
# 子目标级进度 + 重规划计数（方案 §11.6 的两个指标）。
#
# 这两个埋点平时发到 PostHog，本机没配后端，所以这里**拦截**下来落成 JSONL ——
# 步骤上限的校准（§12.3 C3）要靠 `steps_per_subgoal` 的 p90，数据得先能拿到。
SUBGOAL_PROGRESS: list[dict] = []
REPLANS: list[dict] = []


def _install_usage_probe() -> None:
    """记录每次 LLM 调用的 usage。

    **不要**自己开捕获槽（`open_capture`）—— `invoke_llm_with_timeout_message`
    内部已经开了一个槽并在 finally 里 `close_capture()`，外层再开的槽会被
    抢先关掉，永远是空（实测踩过）。

    正确做法是**拦截 telemetry 的 capture_llm_usage** —— 它是成本核算的唯一
    数据源（`services/llm.py` 的 `_record_usage`）。

    ⚠️ 2026-10-21 更正：本段原来写着「在 LangGraph 的图里那个捕获槽根本收不到
    东西 ⇒ usage 仍然是 0」，**那是错的**（`services/usage_bridge.py` 从未存在，
    图里的 usage 并不是靠它上报的）。真机 graph 实测证明这条路是通的：
    `probe_out/executor_e2e.json` 里 executor 21 次调用全是真实非零 token。
    """
    from minitap.mobile_use.services import telemetry as telemetry_mod

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

    # 子目标级进度 + replan 计数（§11.6 的两个指标）
    original_progress = telemetry_mod.telemetry.capture_subgoal_progress
    original_replan = telemetry_mod.telemetry.capture_replan

    def progress_probe(**kwargs) -> None:
        SUBGOAL_PROGRESS.append(dict(kwargs))
        return original_progress(**kwargs)

    def replan_probe(**kwargs) -> None:
        REPLANS.append(dict(kwargs))
        return original_replan(**kwargs)

    telemetry_mod.telemetry.capture_subgoal_progress = progress_probe  # type: ignore[method-assign]
    telemetry_mod.telemetry.capture_replan = replan_probe  # type: ignore[method-assign]


async def run(goal: str) -> int:
    _install_usage_probe()
    llm_config = initialize_llm_config()
    profile = AgentProfile(name="default", llm_config=llm_config)
    config = Builders.AgentConfig.with_default_profile(profile=profile)

    print("=" * 70)
    print(f"真机端到端（新架构）: {goal}")
    print("=" * 70)

    started = time.perf_counter()
    error = None
    agent = None
    try:
        agent = Agent(config=config.build())
        await agent.init(retry_count=5, retry_wait_seconds=2)
        # 把**完整决策序列**落盘：`agents_thoughts` 里有 cortex 每次
        # 「Issued goal for subgoal X」和「judged COMPLETE」的记录 ——
        # 这是诊断「子目标为什么被反复重发」的唯一直接证据
        # （真机第六轮：3 个子目标被重发了 5 次，光看汇总数字看不出原因）。
        thoughts_path = ROOT / "probe_out" / "run_thoughts.txt"
        await agent.run_task(
            request=agent.new_task(goal)
            .with_thoughts_output_saving(str(thoughts_path))
            .build()
        )
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
    import statistics

    for k, v in sorted(by_agent.items(), key=lambda x: -x[1]["in"]):
        med = round(statistics.median(v["ms"])) if v["ms"] else 0
        print(f"  {k:<14}{v['n']:>5}{v['in']:>9}{v['ca']:>9}{v['out']:>9}{med:>9}")

    out = ROOT / "probe_out" / "executor_e2e.json"
    out.write_text(
        json.dumps(
            {
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
                # §11.6 的两个指标，随这次运行一起存下来，供
                # `scripts/audit_steps_per_subgoal.py` 跨多次运行做分布统计。
                "subgoal_progress": SUBGOAL_PROGRESS,
                "replans": REPLANS,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print()
    print(f"  => {out}")

    # 子目标步数分布（步数上限校准的原始数据）
    if SUBGOAL_PROGRESS:
        steps = [p.get("steps", 0) for p in SUBGOAL_PROGRESS]
        print()
        print("  子目标步数分布（§12.3 C3 校准步数上限的原始数据）:")
        print(f"    n={len(steps)}  steps={steps}  max={max(steps)}")
    if REPLANS:
        print(f"  replan 次数: {len(REPLANS)}  （>0 说明看门狗真的被触发过）")
    return 1 if error else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--goal", default="打开设置，进入 WLAN 页面")
    a = p.parse_args()
    return asyncio.run(run(a.goal))


if __name__ == "__main__":
    raise SystemExit(main())
