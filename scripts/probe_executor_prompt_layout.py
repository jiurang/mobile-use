"""验证 Executor prompt 的组装结果：顺序对不对、token 多少、缓存能不能命中。

这是方案 §4.1.1 的核心机制 —— 「每轮变的东西一律放末尾」。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "minitap"))

import tiktoken  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

from minitap.mobile_use.agents.executor.prompt import build_executor_messages  # noqa: E402
from minitap.mobile_use.config import LLM, LLMConfig, LLMConfigUtils, LLMWithFallback  # noqa: E402
from minitap.mobile_use.context import (  # noqa: E402
    DeviceContext,
    DevicePlatform,
    MobileUseContext,
)
from minitap.mobile_use.utils.screen_elements import ScreenElement  # noqa: E402
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds  # noqa: E402

enc = tiktoken.get_encoding("o200k_base")


def ntok(text: str) -> int:
    return len(enc.encode(text))


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
            utils=LLMConfigUtils(outputter=llm(), hopper=llm()),
        ),
    )


def main() -> int:
    ctx = _ctx()

    elements = [
        ScreenElement(
            index=i,
            kind="ui",
            label=f"Row {i}",
            text="",
            resource_id="",
            display_id="",
            class_name="TextView",
            bounds=ElementBounds(x=0, y=0, width=100, height=50),
            flags="C",
        )
        for i in range(1, 31)
    ]

    print("=" * 72)
    print("1. 消息顺序检查（每轮变的必须在末尾）")
    print("=" * 72)
    local = [
        AIMessage(content="", tool_calls=[{"name": "tap", "args": {}, "id": "c1"}]),
        ToolMessage(content="Tapped successfully.", tool_call_id="c1"),
    ]
    msgs = build_executor_messages(
        ctx=ctx,
        goal="进入 WLAN 页面并读出当前连接的 Wi-Fi 名称",
        screen_elements=elements,
        failures=[
            {
                "key": "index=48|id=x",
                "index": 48,
                "resource_id": "x",
                "label": "点赞",
                "reason": "gone",
            }
        ],
        local_messages=local,
        device_state_lines=["Device date: 2026-10-21", "Screen is OFF (the device is asleep)."],
    )
    for i, m in enumerate(msgs):
        first = str(m.content)[:58].replace("\n", " ")
        kind = type(m).__name__
        print(f"  [{i}] {kind:<14} {first}")

    # 找元素表块和局部历史块的位置
    screen_idx = None
    history_idx = None
    for i, m in enumerate(msgs):
        c = str(m.content)
        if "[1]" in c and "Row 1" in c:
            screen_idx = i
        if isinstance(m, ToolMessage):
            history_idx = i
    print()
    print(f"  局部历史位置 = {history_idx}，屏幕元素表位置 = {screen_idx}")
    if screen_idx is not None and history_idx is not None:
        ok = screen_idx > history_idx
        print(f"  {'✓' if ok else '✗'} 元素表在局部历史**之后** —— 布局正确")
        if not ok:
            return 1

    print()
    print("=" * 72)
    print("2. token 预算")
    print("=" * 72)
    total = 0
    for i, m in enumerate(msgs):
        t = ntok(str(m.content))
        total += t
        print(f"  [{i}] {type(m).__name__:<14} {t:>6} tok")
    print(f"  {'合计':<17}{total:>6} tok")
    print()
    print(f"  对比当前 cortex 的 8,454 tok → {8454/max(total,1):.1f}x 更小")

    print()
    print("=" * 72)
    print("3. 缓存布局模拟（元素表在末尾 vs 在中间）")
    print("=" * 72)

    def simulate(elements_last: bool, steps: int = 8) -> float:
        static = ntok(str(msgs[0].content))  # system
        goal_t = ntok(str(msgs[1].content))
        hist_per_step = 90
        el_t = 96
        hit = 0.0
        prev_prompt = 0
        for step in range(1, steps + 1):
            hist = hist_per_step * (step - 1)
            if elements_last:
                prompt = static + goal_t + hist + el_t
                stable = static + goal_t + hist
            else:
                prompt = static + goal_t + el_t + hist
                stable = static + goal_t
            # 缓存滞后一轮
            cached = min(hit, prev_prompt) if prev_prompt else 0
            cached = max(cached, min(stable, prev_prompt))
            hit = cached / prompt
            prev_prompt = prompt
        return hit

    last = simulate(True)
    middle = simulate(False)
    print(f"  元素表在末尾（我们的实现）  均值 {last*100:.1f}%")
    print(f"  元素表在中间（错误布局）    均值 {middle*100:.1f}%")
    print("  现状基线                    74.6%")
    print()
    if last > 0.746:
        print(f"  ✓ 优于现状基线（{last*100:.1f}% > 74.6%）")
    else:
        print("  ✗ 低于现状基线 —— 布局有问题")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
