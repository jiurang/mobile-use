"""调查 MiniMax-M3 结构化输出解析失败的根因与发生率。

背景（2026-10-17 真机实测）
--------------------------
baseline 那一轮 goal 直接崩了：
    ❗ Main LLM inference failed: MiniMax returned no parseable CortexOutput
       (tool_calls=0, finish_reason='stop'). content[:1000]='<think>Let me analyze...'
    → Falling back...
    ❌ 同样失败 → 整个 goal 挂掉

且 `initialize_llm_config()` 实测：**5 个节点的主模型和 fallback 都是 MiniMax-M3**，
也就是「降级」重打的是同一个模型，成功率没有提升，只是多付一次完整调用的钱。

本探针要回答三件事：
1. 失败率到底多高（跑 N 次真实调用）
2. 失败响应的**完整**特征（是不是 thinking 没闭合、JSON 根本没写出来）
3. `max_tokens` 是不是关键变量

跑法：
    uv run python scripts/probe_minimax_parse_fail.py --n 10
    uv run python scripts/probe_minimax_parse_fail.py --n 6 --max-tokens 32768
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

from minitap.mobile_use.agents.cortex.types import CortexOutput  # noqa: E402
from minitap.mobile_use.services.llm import get_minimax_llm  # noqa: E402

PROMPT = """You are the **Cortex**.
Analyze this Android screen state and produce structured decisions.

Current UI elements:
[1] "Settings" id=com.android.settings:id/switch_bar TextView (0,84,1220x132)
[2] "Network & internet" id=com.android.settings:id/collapsing_toolbar_title TextView (64,300,1100x180)
[3] "WLAN" id=com.android.settings:id/collapsing_toolbar_title TextView (64,560,1100x180)
[4] "Bluetooth" id=com.android.settings:id/switch_widget TextView (1000,700,120x120)

The user goal is: "open Settings, go into WLAN, and report the connected network name".

Return your decisions using the required tool call. Always call the tool —
never answer in plain text."""


def classify(msg) -> dict:
    """把一条原始响应归类，落到「能不能解析」这个判据上。"""
    content = msg.content if isinstance(msg.content, str) else ""
    meta = getattr(msg, "response_metadata", {}) or {}
    tool_calls = msg.tool_calls or []

    has_open_think = content.lstrip().startswith("<think>")
    has_close_think = "</think>" in content
    after_think = content.split("</think>", 1)[1] if has_close_think else ""

    info = {
        "finish_reason": meta.get("finish_reason"),
        "n_tool_calls": len(tool_calls),
        "content_len": len(content),
        "content_tail": content[-160:],
        "starts_with_think": has_open_think,
        "think_closed": has_close_think,
        "has_brace": "{" in content,
        "after_think_len": len(after_think),
    }

    try:
        CortexOutput.model_validate(
            tool_calls[0]["args"] if tool_calls and isinstance(tool_calls[0]["args"], dict)
            else json.loads(content.split("```")[1].removeprefix("json") if "```" in content else after_think)
        )
        info["parsed"] = True
    except Exception as e:  # noqa: BLE001
        info["parsed"] = False
        info["parse_error"] = type(e).__name__

    return info


async def one_call(llm, structured, label: str) -> dict:
    runner = llm.with_structured_output(CortexOutput)
    t0 = time.perf_counter()
    try:
        out = await runner.ainvoke(PROMPT)
        elapsed = (time.perf_counter() - t0) * 1000
        return {"label": label, "ok": True, "ms": round(elapsed), "result": str(out)[:120]}
    except Exception as e:  # noqa: BLE001
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "label": label,
            "ok": False,
            "ms": round(elapsed),
            "error": type(e).__name__,
            "detail": str(e)[:300],
        }


async def main_async(n: int, max_tokens: int | None) -> int:
    from minitap.mobile_use.services.llm import MiniMaxChatOpenAI

    kwargs = {}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    llm = MiniMaxChatOpenAI(
        model="MiniMax-M3",
        temperature=1,
        base_url=os.environ.get("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1"),
        api_key=os.environ["MINIMAX_API_KEY"],
        **kwargs,
    )

    label = f"max_tokens={max_tokens or '未设'}"
    print(f"\n=== 直接打 MiniMax-M3，N={n}，{label} ===")
    print(f"    验证目标：主模型和 fallback 是同一个模型，"
          f"所以这里测的就是「重打一次」的真实成功率\n")

    results = []
    for i in range(1, n + 1):
        r = await one_call(llm, CortexOutput, label)
        results.append(r)
        mark = "OK " if r["ok"] else "FAIL"
        print(f"  [{i:>2}/{n}] {mark} {r['ms']:>6}ms  {r.get('error', '')}")
        if not r["ok"]:
            print(f"        {r.get('detail', '')[:200]}")
        await asyncio.sleep(0.4)

    ok = sum(1 for r in results if r["ok"])
    times = [r["ms"] for r in results]
    print(f"\n  成功 {ok}/{n}  ({ok / n * 100:.0f}%)")
    if times:
        srt = sorted(times)
        print(f"  耗时 中位 {statistics.median(srt):.0f}ms  最小 {srt[0]}ms  最大 {srt[-1]}ms")
    print(f"  与基线对比：同一个模型重打一次，成功率不会因为「兜底」提高")

    out = ROOT / "probe_out" / "minimax_parse_probe.json"
    out.write_text(
        json.dumps(
            {"n": n, "max_tokens": max_tokens, "ok": ok, "results": results},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"  => {out}")
    return 0 if ok == n else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=0)
    a = p.parse_args()
    raise SystemExit(asyncio.run(main_async(a.n, a.max_tokens or None)))
