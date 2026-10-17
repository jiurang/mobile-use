"""用**真实工具**验证 cortex.md 里写的那套动作形状到底能不能直连。

为什么要单独验
--------------
`tests/test_cortex_dispatcher.py` 里用的是几个玩具工具，它只能证明调度器
逻辑对，证明不了「Cortex 按 prompt 里写的格式产出的 decisions，调度器认得」。

而这恰恰是整件事成败的地方：认不得就会静默退回 LLM —— 功能完全正常，
但一整轮往返白省了，你在日志里只能看到 "Dispatch: unknown arg"，
很容易误以为「优化没生效」。

所以这里拿 `tools/index.py` 里**真实的** EXECUTOR_WRAPPERS_TOOLS，逐个用
cortex.md 里写的形状喂一遍 `dispatch_decisions`，报告命中/退回。

跑法：
    uv run python scripts/probe_dispatch_hit_rate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "minitap"))

from minitap.mobile_use.agents.cortex.dispatcher import dispatch_decisions  # noqa: E402
from minitap.mobile_use.config import LLM, LLMConfig, LLMConfigUtils, LLMWithFallback  # noqa: E402
from minitap.mobile_use.context import DeviceContext, DevicePlatform, MobileUseContext  # noqa: E402
from minitap.mobile_use.tools.index import EXECUTOR_WRAPPERS_TOOLS, get_tools_from_wrappers  # noqa: E402

THOUGHT = "Tapping the search field to focus it before typing the query."


def _ctx() -> MobileUseContext:
    def llm() -> LLMWithFallback:
        return LLMWithFallback(provider="openai", model="gpt-5", fallback=LLM(provider="openai", model="o4-mini"))

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


TARGET = {
    "resource_id": "com.app:id/search_input",
    "resource_id_index": 0,
    "bounds": {"x": 91, "y": 2446, "width": 1038, "height": 137},
    "text": "Search",
    "text_index": 0,
}

# 完全照抄 cortex.md 的 "Action Arguments" 表
CASES: list[tuple[str, dict]] = [
    ("tap", {"action": "tap", "target": TARGET}),
    ("long_press_on", {"action": "long_press_on", "target": TARGET}),
    ("focus_and_clear_text", {"action": "focus_and_clear_text", "target": TARGET}),
    (
        "focus_and_input_text",
        {"action": "focus_and_input_text", "target": TARGET, "text": "hello world"},
    ),
    (
        "swipe_percentages",
        {
            "action": "swipe_percentages",
            "start_x_percent": 50,
            "start_y_percent": 80,
            "end_x_percent": 50,
            "end_y_percent": 20,
        },
    ),
    (
        "swipe_coordinates",
        {
            "action": "swipe_coordinates",
            "start_x": 600,
            "start_y": 2000,
            "end_x": 600,
            "end_y": 800,
        },
    ),
    ("launch_app", {"action": "launch_app", "app_name": "WhatsApp"}),
    ("stop_app", {"action": "stop_app", "package_name": "com.app"}),
    ("open_link", {"action": "open_link", "url": "https://example.com"}),
    ("press_key", {"action": "press_key", "key": "enter"}),
    ("wait_for_delay", {"action": "wait_for_delay", "time_in_ms": 800}),
    ("back", {"action": "back"}),
    ("erase_one_char", {"action": "erase_one_char"}),
    ("save_note", {"action": "save_note", "key": "order", "content": "12345"}),
    ("read_note", {"action": "read_note", "key": "order"}),
    ("list_notes", {"action": "list_notes"}),
]


def main() -> int:
    ctx = _ctx()
    tools = get_tools_from_wrappers(ctx, EXECUTOR_WRAPPERS_TOOLS)
    print(f"\n真实工具 {len(tools)} 个: {', '.join(t.name for t in tools)}")

    print(f"\n=== cortex.md 文档形状的直连命中率 ===")
    hits, misses = [], []
    for label, action in CASES:
        decisions = json.dumps([action], ensure_ascii=False)
        calls = dispatch_decisions(decisions, THOUGHT, tools)
        if calls is None:
            misses.append(label)
            print(f"  ✗ {label:<26} 退回 LLM")
        else:
            hits.append(label)
            print(f"  ✓ {label:<26} -> {calls[0]['name']}")

    print(f"\n  命中 {len(hits)}/{len(CASES)}")
    if misses:
        print(f"  退回: {', '.join(misses)}")
        print("  => cortex.md 的表格和真实工具签名对不上，需要修文档")
        return 1

    # 顺带确认：未知动作名 / 缺必填参数 依然会退回（不能因为文档补全就放松校验）
    for label, bad in [
        ("unknown tool", {"action": "teleport"}),
        ("missing target", {"action": "tap"}),
        ("unknown arg", {"action": "back", "coord_x": 3}),
    ]:
        assert dispatch_decisions(json.dumps([bad]), THOUGHT, tools) is None, label
    print("  ✓ 坏数据依然全部退回 LLM（校验没被放松）")

    # ---- 这次优化到底省了多少 prefill ----
    import tiktoken

    from langchain_core.utils.function_calling import convert_to_openai_tool

    enc = tiktoken.get_encoding("o200k_base")

    def ntok(text: str) -> int:
        return len(enc.encode(text))

    executor_md = (
        Path("minitap/mobile_use/agents/executor/executor.md")
        .read_text(encoding="utf-8")
        .replace("{{ platform }}", "android")
    )
    schemas = json.dumps(
        [convert_to_openai_tool(t) for t in tools], ensure_ascii=False
    )
    sample_decisions = json.dumps([CASES[0][1]], ensure_ascii=False)

    prompt_tok = ntok(executor_md) + ntok(THOUGHT) + ntok(sample_decisions)
    schema_tok = ntok(schemas)

    print(f"\n=== 每轮省掉的 Executor prefill ===")
    print(f"  executor.md          {ntok(executor_md):6d} tok")
    print(f"  16 个工具的 schema    {schema_tok:6d} tok  ({(schema_tok / (prompt_tok + schema_tok) * 100):.0f}%)")
    print(f"  cortex thought       {ntok(THOUGHT):6d} tok")
    print(f"  decisions            {ntok(sample_decisions):6d} tok")
    total = prompt_tok + schema_tok
    print(f"  --- 合计             {total:6d} tok / 轮")
    print(f"  外加一次完整的生成 + 网络往返（这部分是净省，不是 prefill）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
