"""拆开量 Cortex 一轮的耗时构成，回答「cortex 到底慢在哪」。

为什么必须拆
------------
实测两轮真机数据：
    baseline   cortex 均值  8293ms  中位 7012ms（只跑了 5 轮就崩）
    optimized  cortex 均值 30743ms  中位 12567ms（跑满 10 轮）
均值是中位的 2.4 倍 —— 说明**方差**才是问题，不是中位数。最可能的原因是
`with_fallback`：主模型解析失败时会再打一次**完整**的 LLM 调用，那一轮直接翻倍。
但这只是猜测，本探针用实测把它钉死。

插桩点（全部是 cortex 模块里的引用，只在本进程内替换）
----------------------------------------------------
  render        render_hierarchy_and_labels + render_ocr_results（裁剪与去重）
  screenshot    controller.get_compressed_b64_screenshot（JPEG 重编码）
  llm_main      with_fallback 的 main_call
  llm_fallback  with_fallback 的 fallback_call
  llm_build     get_llm + with_structured_output（构造客户端和 schema）
  other         cortex 总时间减去上面全部

跑法：
    uv run python scripts/probe_cortex_profile.py --goal "..." --tag profile
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MOBILE_USE_TELEMETRY_ENABLED", "false")

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "minitap"))

from minitap.mobile_use.agents.cortex import cortex as cortex_mod  # noqa: E402
from minitap.mobile_use.agents.contextor.contextor import ContextorNode  # noqa: E402
from minitap.mobile_use.agents.executor.executor import ExecutorNode  # noqa: E402
from minitap.mobile_use.config import initialize_llm_config  # noqa: E402
from minitap.mobile_use.sdk import Agent  # noqa: E402
from minitap.mobile_use.sdk.builders import Builders  # noqa: E402
from minitap.mobile_use.sdk.types.agent import AgentProfile  # noqa: E402

BUCKETS: dict[str, list[float]] = defaultdict(list)
CORTEX_TOTAL: list[float] = []
CONTEXTOR_TOTAL: list[float] = []
EXEC_CALLS: list[float] = []


def _timed(bucket: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        BUCKETS[bucket].append((time.perf_counter() - t0) * 1000)


def install() -> None:
    # ---- 渲染（裁剪 + OCR 去重）----
    orig_render_tree = cortex_mod.render_hierarchy_and_labels
    cortex_mod.render_hierarchy_and_labels = lambda els: _timed(
        "render", orig_render_tree, els
    )

    orig_render_ocr = cortex_mod.render_ocr_results
    cortex_mod.render_ocr_results = lambda *a, **k: _timed("render", orig_render_ocr, *a, **k)

    # ---- LLM 客户端构造 ----
    orig_get_llm = cortex_mod.get_llm
    cortex_mod.get_llm = lambda **k: _timed("llm_build", orig_get_llm, **k)

    # ---- 主 / 兜底两次调用的耗时分开记 ----
    orig_with_fallback = cortex_mod.with_fallback

    async def timed_with_fallback(main_call, fallback_call, **kwargs):
        async def timed_main():
            t0 = time.perf_counter()
            try:
                return await main_call()
            finally:
                BUCKETS["llm_main"].append((time.perf_counter() - t0) * 1000)

        async def timed_fallback():
            t0 = time.perf_counter()
            try:
                return await fallback_call()
            finally:
                BUCKETS["llm_fallback"].append((time.perf_counter() - t0) * 1000)

        return await orig_with_fallback(timed_main, timed_fallback, **kwargs)

    cortex_mod.with_fallback = timed_with_fallback

    # ---- 截图压缩：在 controller 类上包，cortex 里是动态调用 ----
    from minitap.mobile_use.controllers.android_controller import AndroidDeviceController

    orig_compress = AndroidDeviceController.get_compressed_b64_screenshot

    def timed_compress(self, *a, **k):
        t0 = time.perf_counter()
        try:
            return orig_compress(self, *a, **k)
        finally:
            BUCKETS["screenshot"].append((time.perf_counter() - t0) * 1000)

    AndroidDeviceController.get_compressed_b64_screenshot = timed_compress

    # ---- 节点总时间 ----
    def wrap(cls, sink):
        original = cls.__call__

        async def timed(self, state):
            t0 = time.perf_counter()
            try:
                return await original(self, state)
            finally:
                sink.append((time.perf_counter() - t0) * 1000)

        cls.__call__ = timed

    wrap(cortex_mod.CortexNode, CORTEX_TOTAL)
    wrap(ContextorNode, CONTEXTOR_TOTAL)
    wrap(ExecutorNode, EXEC_CALLS)


def stats(samples: list[float]) -> dict:
    if not samples:
        return {"n": 0}
    s = sorted(samples)
    return {
        "n": len(s),
        "total_s": round(sum(s) / 1000, 1),
        "mean_ms": round(sum(s) / len(s)),
        "median_ms": round(statistics.median(s)),
        "min_ms": round(s[0]),
        "max_ms": round(s[-1]),
    }


async def run(goal: str, tag: str, ocr: bool) -> dict:
    install()
    llm_config = initialize_llm_config()
    profile = AgentProfile(name="default", llm_config=llm_config)
    config = Builders.AgentConfig.with_default_profile(profile=profile)

    if not ocr:
        from minitap.mobile_use.context import PerceptionConfig

        config = config.with_perception(PerceptionConfig(ocr=False))

    started = time.perf_counter()
    error = None
    agent = None
    try:
        agent = Agent(config=config.build())
        await agent.init(retry_count=5, retry_wait_seconds=2)
        await agent.run_task(request=agent.new_task(goal).build())
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        if agent is not None:
            await agent.clean()
    wall_ms = (time.perf_counter() - started) * 1000

    result = {
        "tag": tag,
        "ocr_enabled": ocr,
        "goal": goal,
        "wall_s": round(wall_ms / 1000, 1),
        "error": error,
        "cortex_total": stats(CORTEX_TOTAL),
        "contextor_total": stats(CONTEXTOR_TOTAL),
        "executor_total": stats(EXEC_CALLS),
        "buckets": {k: stats(v) for k, v in sorted(BUCKETS.items())},
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # other = cortex 总时间 - 各项之和（逐轮对齐后求和，只用于量级判断）
    accounted = sum(sum(v) for k, v in BUCKETS.items())
    result["other_ms_total"] = round(sum(CORTEX_TOTAL) - accounted)

    out = ROOT / "probe_out" / f"cortex_profile_{tag}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def report(r: dict) -> None:
    print("\n" + "=" * 70)
    print(f"[{r['tag']}] OCR={'开' if r['ocr_enabled'] else '关'}  {r['goal']}")
    print(f"  全程 {r['wall_s']}s   cortex 轮次={r['cortex_total'].get('n', 0)}"
          f"   {'⚠ ' + r['error'] if r['error'] else '成功'}")
    print("=" * 70)

    ct = r["cortex_total"]
    print(f"\n【cortex 总耗时】 均值 {ct.get('mean_ms', 0)}ms  中位 {ct.get('median_ms', 0)}ms  "
          f"最大 {ct.get('max_ms', 0)}ms")
    print(f"\n{'组成部分':<16}{'次数':>5}{'总(s)':>9}{'均值ms':>10}{'中位ms':>9}{'最大ms':>9}{'占cortex':>10}")
    print("-" * 70)
    acct = 0.0
    for key, label in [
        ("llm_main", "LLM 主调用"),
        ("llm_fallback", "LLM 兜底"),
        ("screenshot", "截图压缩"),
        ("render", "树/OCR 渲染"),
        ("llm_build", "LLM 客户端构造"),
    ]:
        s = r["buckets"].get(key, {"n": 0})
        if not s.get("n"):
            print(f"{label:<16}{0:>5}{'-':>9}{'-':>10}{'-':>9}{'-':>9}{'-':>10}")
            continue
        acct += s["total_s"]
        pct = s["total_s"] / ct["total_s"] * 100 if ct.get("total_s") else 0
        print(f"{label:<16}{s['n']:>5}{s['total_s']:>9.1f}{s['mean_ms']:>10}{s['median_ms']:>9}"
              f"{s['max_ms']:>9}{pct:>9.1f}%")
    other_s = r["other_ms_total"] / 1000
    pct = other_s / ct["total_s"] * 100 if ct.get("total_s") else 0
    print(f"{'其它(Jinja等)':<16}{'-':>5}{other_s:>9.1f}{'-':>10}{'-':>9}{'-':>9}{pct:>9.1f}%")

    fb = r["buckets"].get("llm_fallback", {"n": 0})
    main_n = r["buckets"].get("llm_main", {"n": 0}).get("n", 0)
    if fb.get("n"):
        print(f"\n⚠ 触发了 {fb['n']}/{main_n} 次 fallback —— 这些轮次 cortex 耗时直接翻倍")
    else:
        print(f"\n✓ fallback 一次都没触发（{main_n} 次主调用全部成功）")

    print(f"\n对照：contextor 均值 {r['contextor_total'].get('mean_ms', 0)}ms"
          f"   executor 均值 {r['executor_total'].get('mean_ms', 0)}ms")
    print(f"=> probe_out/cortex_profile_{r['tag']}.json")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--goal", required=True)
    p.add_argument("--tag", default="profile")
    p.add_argument("--ocr", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    report(asyncio.run(run(a.goal, a.tag, a.ocr)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
