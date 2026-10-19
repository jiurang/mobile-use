"""用**真实 cortex prompt + 真机截图**复现 MiniMax-M3 的解析失败。

上一版探针（scripts/probe_minimax_parse_fail.py）用的是短小的合成 prompt，
结果 10/10 全过 —— 说明失败不是模型的固有随机性，而是**和真实 prompt 有关**。
这一版把真实条件补齐：cortex.md 全文 + 真机截图 + 真机层级树渲染。

关注点：失败响应到底长什么样。判据只有三条
    1. tool_calls 是不是空
    2. <think> 有没有闭合（_strip_markdown_json 靠闭合标签才剥得掉）
    3. finish_reason 是不是 length（是的话就是被 max_tokens 截断）

跑法：
    uv run python scripts/probe_cortex_real_prompt.py --n 10
    uv run python scripts/probe_cortex_real_prompt.py --n 6 --max-tokens 32768
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

from jinja2 import Template  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from minitap.mobile_use.agents.cortex.types import CortexOutput  # noqa: E402
from minitap.mobile_use.services.llm import MiniMaxChatOpenAI  # noqa: E402
from minitap.mobile_use.utils.conversations import get_screenshot_message_for_llm  # noqa: E402
from minitap.mobile_use.utils.image_compress import compress_screenshot_base64  # noqa: E402
from minitap.mobile_use.utils.ui_hierarchy_view import render_hierarchy_and_labels  # noqa: E402

SERIAL = "89R8ZXNBGQIRJRLF"


def build_messages() -> tuple[list, dict]:
    """用真机当前状态拼一份和 cortex.py 同构的消息。"""
    from minitap.mobile_use.clients.ui_automator_client import (
        UIAutomatorClient,
        _parse_hierarchy_xml_to_elements,
        _pil_to_base64,
    )

    client = UIAutomatorClient(SERIAL)
    device = client._ensure_connected()
    shot = device.screenshot()
    xml = device.dump_hierarchy(compressed=True)
    elements = _parse_hierarchy_xml_to_elements(xml)

    tree_text, _ = render_hierarchy_and_labels(elements)
    compressed = compress_screenshot_base64(_pil_to_base64(shot))

    md = (ROOT / "minitap/mobile_use/agents/cortex/cortex.md").read_text(encoding="utf-8")
    # cortex.md 自 v2 起整段静态：目标 / 计划 / feedback 全部走 messages，不再入模板。
    system = Template(md).render(
        platform="android",
        executor_tools_list="back, open_link, tap, long_press_on, swipe_percentages, "
                            "swipe_coordinates, focus_and_input_text, erase_one_char, "
                            "launch_app, stop_app, focus_and_clear_text, press_key, "
                            "wait_for_delay, save_note, read_note, list_notes",
        locked_app_package=None,
        perception={"ui_hierarchy": True, "screenshot": True, "ocr": True},
    )

    # 与 cortex.py 的分层一致：恒定 → 只追加 → 每轮变。
    messages = [
        SystemMessage(content=system),
        HumanMessage(
            content="\n".join(
                [
                    "Here are my device info:",
                    "Device ID: 89R8ZXNBGQIRJRLF",
                    "Initial goal: 打开设置，点进 WLAN，告诉我当前连接的 Wi-Fi 名称，然后返回桌面",
                    "Subgoal plan (append-only history; the last PLAN section is the active one):",
                    "PLAN v1:\n- [ID:s1] 打开设置\n- [ID:s2] 进入 WLAN\n"
                    "- [ID:s3] 读出 Wi-Fi 名称\n- [ID:s4] 返回桌面",
                ]
            )
        ),
        HumanMessage(
            content="Agent thoughts history (newest last):\n"
            "Decisions reason: 已启动设置，正在进入 WLAN 页面。"
        ),
        HumanMessage(
            content="Subgoal status: [s1] done | [s2] active | [s3] pending | [s4] pending"
        ),
        HumanMessage(content="Executor feedback:\n[ToolMessage(content='Tapped WLAN successfully.')]"),
        HumanMessage(content=tree_text),
        get_screenshot_message_for_llm(compressed),
    ]

    meta = {
        "system_chars": len(system),
        "tree_chars": len(tree_text),
        "tree_lines": tree_text.count("\n"),
        "raw_nodes": len(elements),
        "screenshot_b64_kb": len(compressed) // 1024,
    }
    return messages, meta


async def run(n: int, max_tokens: int | None, provider: str, model: str, effort: str | None) -> int:
    from minitap.mobile_use.services.llm import _ToolCallStructuredRunnable

    messages, meta = build_messages()
    print(f"\n=== 真实 cortex prompt ===")
    for k, v in meta.items():
        print(f"    {k:<20} {v}")

    if provider == "deepseek":
        from minitap.mobile_use.services.llm import get_deepseek_llm

        llm = get_deepseek_llm(model, temperature=1)
        if max_tokens:
            llm.max_tokens = max_tokens
    else:
        from minitap.mobile_use.services.llm import MiniMaxChatOpenAI

        kwargs = {"max_tokens": max_tokens} if max_tokens else {}
        if effort:
            kwargs["extra_body"] = {"output_config": {"effort": effort}}
        llm = MiniMaxChatOpenAI(
            model=model,
            temperature=1,
            base_url=os.environ.get("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1"),
            api_key=os.environ["MINIMAX_API_KEY"],
            **kwargs,
        )
    runner = _ToolCallStructuredRunnable(llm, CortexOutput)

    label = f"provider={provider} model={model} max_tokens={max_tokens or '未设'} effort={effort or '未设'}"
    print(f"\n=== N={n}, {label} ===\n")

    results = []
    for i in range(1, n + 1):
        t0 = time.perf_counter()
        try:
            await runner.ainvoke(messages)
            ok, info = True, {}
        except Exception as e:  # noqa: BLE001
            ok, info = False, {"error": type(e).__name__, "detail": str(e)[:400]}
        ms = round((time.perf_counter() - t0) * 1000)
        results.append({"i": i, "ok": ok, "ms": ms, **info})
        print(f"  [{i:>2}/{n}] {'OK ' if ok else 'FAIL'} {ms:>7}ms  {info.get('error', '')}")
        if not ok:
            print(f"        {info.get('detail', '')[:300]}")
        await asyncio.sleep(0.4)

    ok_n = sum(1 for r in results if r["ok"])
    times = sorted(r["ms"] for r in results)
    print(f"\n  成功 {ok_n}/{n} ({ok_n / n * 100:.0f}%)")
    print(f"  耗时 中位 {statistics.median(times)}ms  最小 {times[0]}ms  最大 {times[-1]}ms")

    out = ROOT / "probe_out" / f"cortex_real_prompt_{model.replace('.', '_')}{'_effort_' + effort if effort else ''}.json"
    out.write_text(
        json.dumps({"meta": meta, "n": n, "provider": provider, "model": model,
                    "max_tokens": max_tokens, "effort": effort, "results": results},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  => {out}")
    return 0 if ok_n == n else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--provider", choices=["minimax", "deepseek"], default="minimax")
    p.add_argument("--model", default="MiniMax-M3")
    p.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"],
                   help="显式下发 output_config.effort；不传则吃服务端默认")
    a = p.parse_args()
    raise SystemExit(asyncio.run(run(a.n, a.max_tokens or None, a.provider, a.model, a.effort)))
