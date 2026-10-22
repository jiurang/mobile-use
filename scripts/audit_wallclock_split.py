"""**墙钟时间去哪了** —— 把「图在干活」和「SDK 在消费事件」分开计时。

问题
----
`scripts/audit_step_overhead.py` 实测：一次 254.9s 的运行里，
**所有节点耗时合计只有 56.3s** —— 有 **198.6s（78%）不在任何节点里**。

嫌疑（SDK 侧，`sdk/agent.py:756-782`）::

    async for chunk in (await get_graph(context)).astream(
        ...,
        stream_mode=["messages", "custom", "updates", "values"],
    ):
        stream_mode, payload = chunk
        if stream_mode == "values":
            last_state_snapshot = payload
            last_state = State(**last_state_snapshot)   # ← 每个 super-step 重建整个 state

两个可疑点：
1. **`values` 每个 super-step 吐整个 state**，SDK 每次都
   ``State(**payload)`` 重建 pydantic 模型 —— 而 state 里有**越来越大的
   `messages`**、还有 `latest_screenshot`（PNG bytes）。⇒ O(super-steps × state 大小)。
2. **`messages` 模式吐每一个 LLM token chunk** —— 一条 372 tok 的输出就是
   几百个事件，每个都要过一遍 SDK 的循环。

本探针**复刻 SDK 的 astream 循环**（同一份图、同一批参数），但把两段分开计时：
* **等图**：``await`` 下一个 chunk 的时间（图自己的活）
* **消费**：处理这个 chunk 的时间（SDK 侧的活），并按 stream_mode 分开

跑法（要真机在线）：
    uv run python scripts/audit_wallclock_split.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MOBILE_USE_TELEMETRY_ENABLED", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

WAIT: dict[str, float] = defaultdict(float)  # 等图（按 stream_mode 记）
HANDLE: dict[str, float] = defaultdict(float)  # 消费（按 stream_mode 记）
COUNT: dict[str, int] = defaultdict(int)
STATE_REBUILD: list[float] = []


async def main() -> int:
    from minitap.mobile_use.config import initialize_llm_config
    from minitap.mobile_use.graph.graph import get_graph
    from minitap.mobile_use.graph.state import State
    from minitap.mobile_use.sdk.builders import Builders
    from minitap.mobile_use.sdk.types.agent import AgentProfile

    llm_config = initialize_llm_config()
    profile = AgentProfile(name="default", llm_config=llm_config)
    config = Builders.AgentConfig.with_default_profile(profile=profile).build()

    from minitap.mobile_use.context import MobileUseContext
    from minitap.mobile_use.sdk.agent import Agent

    agent = Agent(config=config)
    await agent.init(retry_count=5, retry_wait_seconds=2)

    # 复刻 SDK 的 context 构造（见 sdk/agent.py:710）
    ctx = MobileUseContext(
        trace_id="audit",
        device=agent._device_context,
        adb_client=agent._adb_client,
        ui_adb_client=agent._ui_adb_client,
        ios_client=agent._ios_client,
        llm_config=profile.llm_config,
        video_recording_enabled=False,
    )
    graph = await get_graph(ctx)

    goal = "打开设置，进入 WLAN 页面"
    # 复刻 SDK 的 graph_input（`sdk/agent.py:_get_graph_state`）——
    # State 的必填字段一个都不能少，否则 `State(**payload)` 直接抛。
    graph_input = State(
        messages=[],
        initial_goal=goal,
        subgoal_plan=[],
        latest_ui_hierarchy=None,
        latest_ocr_results=None,
        latest_screen_elements=None,
        latest_screenshot=None,
        focused_app_info=None,
        device_date=None,
        structured_decisions=None,
        complete_subgoals_by_ids=[],
        agents_thoughts=[],
        remaining_steps=30,
        executor_messages=[],
        cortex_last_thought=None,
        scratchpad={},
    ).model_dump()
    print(f"[probe] 复刻 SDK 的 astream 循环，分开计时: {goal}")

    started = time.perf_counter()
    it = graph.astream(
        input=graph_input,
        config={"recursion_limit": 400},
        stream_mode=["messages", "custom", "updates", "values"],
    )
    while True:
        t0 = time.perf_counter()
        try:
            chunk = await it.__anext__()
        except StopAsyncIteration:
            break
        WAIT["total"] += time.perf_counter() - t0

        t1 = time.perf_counter()
        stream_mode, payload = chunk
        COUNT[str(stream_mode)] += 1
        if stream_mode == "values":
            t2 = time.perf_counter()
            State(**payload)  # type: ignore[arg-type]
            STATE_REBUILD.append(time.perf_counter() - t2)
        HANDLE[str(stream_mode)] += time.perf_counter() - t1

    wall = time.perf_counter() - started
    await agent.clean()

    print()
    print("=" * 84)
    print(f"墙钟 {wall:.1f}s")
    print("=" * 84)
    wait_s = WAIT["total"]
    handle_s = sum(HANDLE.values())
    print(f"  {'等图（图自己干活）':<28} {wait_s:>9.1f}s  ({wait_s / wall * 100:>5.1f}%)")
    print(f"  {'SDK 消费（处理事件）':<28} {handle_s:>9.1f}s  ({handle_s / wall * 100:>5.1f}%)")
    print()
    print(f"  {'stream_mode':<22}{'事件数':>8}{'消费耗时':>11}{'占比':>9}")
    for mode in sorted(COUNT, key=lambda m: -HANDLE[m]):
        print(
            f"  {mode:<22}{COUNT[mode]:>8}{HANDLE[mode]:>10.2f}s"
            f"{HANDLE[mode] / wall * 100:>8.1f}%"
        )
    if STATE_REBUILD:
        total_rebuild = sum(STATE_REBUILD)
        print()
        print(
            f"  🔴 `State(**payload)` 重建 state：{len(STATE_REBUILD)} 次，"
            f"合计 **{total_rebuild:.1f}s**（占墙钟 {total_rebuild / wall * 100:.1f}%），"
            f"单次中位 {sorted(STATE_REBUILD)[len(STATE_REBUILD) // 2] * 1000:.0f}ms"
        )

    out = ROOT / "probe_out" / "wallclock_split.json"
    out.write_text(
        json.dumps(
            {
                "wall_s": round(wall, 1),
                "wait_s": round(WAIT["total"], 1),
                "handle_s": round(sum(HANDLE.values()), 1),
                "by_mode": {
                    m: {"count": COUNT[m], "handle_s": round(HANDLE[m], 2)} for m in COUNT
                },
                "state_rebuild_s": round(sum(STATE_REBUILD), 1),
                "state_rebuild_n": len(STATE_REBUILD),
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
