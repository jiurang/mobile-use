"""新拓扑的**结构**冒烟测试：不接真机、不调 LLM，只验证图能编译、边能走通。

要锁的东西（都是审计里判过致命的）：

1. 图能编译（`get_graph(ctx)` 不抛）。
2. **`reporter → orchestrator` 这条无条件边存在** —— 少了它，计划级失败就没有
   处理者，Cortex 一直重发到撞 `recursion_limit`（方案 §6.5.1 / R16）。
3. **`executor_tools → contextor`** —— 工具执行完必须回 contextor 重新抓屏，
   直接回 executor 会让工具拿到上一屏的 `Target.index`（方案 §9.1 / R19）。
4. `contextor → executor`（抓屏 → 决策）在环上。

跑法：
    uv run python scripts/probe_executor_graph_structure.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "minitap"))

from minitap.mobile_use.config import LLM, LLMConfig, LLMConfigUtils, LLMWithFallback  # noqa: E402
from minitap.mobile_use.context import (  # noqa: E402
    DeviceContext,
    DevicePlatform,
    MobileUseContext,
)
from minitap.mobile_use.graph.graph import get_graph  # noqa: E402


def _ctx() -> MobileUseContext:
    def llm() -> LLMWithFallback:
        return LLMWithFallback(
            provider="openai", model="gpt-5", fallback=LLM(provider="openai", model="o4-mini")
        )

    return MobileUseContext(
        trace_id="probe",
        device=DeviceContext(
            host_platform="WINDOWS",
            mobile_platform=DevicePlatform.ANDROID,
            device_id="probe",
            device_width=1220,
            device_height=2712,
        ),
        llm_config=LLMConfig(
            planner=llm(),
            orchestrator=llm(),
            contextor=llm(),
            cortex=llm(),
            executor=llm(),
            reporter=None,  # 走回落
            utils=LLMConfigUtils(outputter=llm(), hopper=llm()),
        ),
    )


def main() -> int:
    ctx = _ctx()
    graph = asyncio.run(get_graph(ctx))

    # 拿到编译后的边
    drawable = graph.get_graph()
    edges: list[tuple[str, str]] = []
    for e in drawable.edges:
        edges.append((e.source, e.target))

    print("=" * 70)
    print("编译后的边")
    print("=" * 70)
    for src, dst in sorted(set(edges)):
        print(f"  {src:<16} → {dst}")

    print()
    print("=" * 70)
    print("关键边检查")
    print("=" * 70)

    checks = [
        (
            ("reporter", "orchestrator"),
            "汇报 → Orchestrator 看门狗（无条件）。\n"
            "     少了它，计划级失败没有处理者 → Cortex 一直重发到撞 recursion_limit",
        ),
        (
            ("executor_tools", "summarizer"),
            "工具执行完 → 裁剪 messages",
        ),
        (
            ("summarizer", "contextor"),
            "→ 重新抓屏。\n"
            "     不能直接回 executor：那样工具解析 Target.index 时拿到的是上一屏的编号",
        ),
        (
            ("contextor", "executor"),
            "抓屏 → 决策（主循环入口）",
        ),
    ]

    ok = True
    for pair, why in checks:
        present = pair in edges
        if not present:
            ok = False
        print(f"  {'✓' if present else '✗'} {pair[0]} → {pair[1]}")
        print(f"      {why}")

    print()
    print("=" * 70)
    print("节点")
    print("=" * 70)
    nodes = sorted(str(n) for n in drawable.nodes)
    print(f"  {nodes}")
    for required in ("planner", "orchestrator", "contextor", "cortex", "executor", "reporter"):
        present = required in nodes
        if not present:
            ok = False
        print(f"  {'✓' if present else '✗'} 节点 {required}")

    print()
    print("=" * 70)
    print("结果：" + ("全部通过" if ok else "**有缺失**"))
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
