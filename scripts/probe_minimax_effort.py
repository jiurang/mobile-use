"""MiniMax M3.1-Flash-Preview 的 `output_config.effort` 阶梯实测。

官方文档要点（2026-10-17 查）：
  - 「MiniMax-M3.1-Flash-Preview additionally **always thinks** and supports
     tunable thinking depth via **output_config.effort**, which accepts
     **low, medium, high, xhigh, max**; the default is **max**」
  - 「Thinking content is **returned separately in thinking content blocks**」

也就是说 M3.1 关不掉思考，但可以把思考**调浅**——默认 max 是最贵最慢的档。
这正是「低配置」在 MiniMax 上的对应物。

要回答三个问题：
  1. effort 调浅能省多少 token / 省多少时间（成本判据）
  2. 思考内容是不是真的不再落进 content（关系到解析失败）
  3. 各档的工具调用成功率（关思考会不会让 M3 重蹈 M3 的覆辙）

用原始 API 直发，因为只有它给得出 usage —— 那才是成本判据。

跑法：
    uv run python scripts/probe_minimax_effort.py --n 4
    uv run python scripts/probe_minimax_effort.py --n 4 --model MiniMax-M3
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))
sys.path.insert(0, str(ROOT / "scripts"))

from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402

from minitap.mobile_use.agents.cortex.types import CortexOutput  # noqa: E402
from probe_cortex_real_prompt import build_messages  # noqa: E402
from probe_deepseek_raw import to_openai_messages  # noqa: E402

EFFORTS = ["low", "medium", "high", "max"]


async def one(client, url, headers, base: dict, extra: dict) -> dict:
    t0 = time.perf_counter()
    r = await client.post(url, json={**base, **extra}, headers=headers)
    ms = round((time.perf_counter() - t0) * 1000)
    if r.status_code != 200:
        return {"ms": ms, "http": r.status_code, "err": r.text[:180]}

    body = r.json()
    ch = (body.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    content = msg.get("content") or ""

    return {
        "ms": ms,
        "finish_reason": ch.get("finish_reason"),
        "n_tool_calls": len(msg.get("tool_calls") or []),
        "content_len": len(content),
        "has_think_tag": "<think>" in content,
        "has_reasoning_content": bool(msg.get("reasoning_content")),
        "reasoning_len": len(msg.get("reasoning_content") or ""),
        "prompt_tokens": usage.get("prompt_tokens") or 0,
        "completion_tokens": usage.get("completion_tokens") or 0,
        "reasoning_tokens": details.get("reasoning_tokens") or 0,
    }


async def run(n: int, model: str) -> int:
    import httpx

    messages, meta = build_messages()
    tools = [convert_to_openai_tool(CortexOutput)]
    base = {
        "model": model,
        "messages": to_openai_messages(messages),
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": tools[0]["function"]["name"]}},
        "temperature": 1,
    }
    url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['MINIMAX_API_KEY']}",
        "Content-Type": "application/json",
    }

    print(f"\n=== {model} · output_config.effort 阶梯 ===")
    print(f"    prompt={meta['system_chars']}字 + {meta['screenshot_b64_kb']}KB 截图，每档 n={n}\n")
    print(f"{'effort':<10}{'成功':>7}{'中位ms':>9}{'最大ms':>9}{'完成tok':>9}{'推理tok':>9}"
          f"{'有思维':>8}{'content含<think>':>18}")
    print("-" * 80)

    summary = {}
    async with httpx.AsyncClient(timeout=240) as client:
        for effort in EFFORTS:
            extra = {"output_config": {"effort": effort}}
            rows = []
            for _ in range(n):
                try:
                    rows.append(await one(client, url, headers, base, extra))
                except Exception as e:  # noqa: BLE001
                    rows.append({"ms": 0, "err": f"{type(e).__name__}: {e}"})
                await asyncio.sleep(0.4)

            ok = [r for r in rows if not r.get("err") and not r.get("http")]
            times = sorted(r["ms"] for r in ok) or [0]
            ct = [r["completion_tokens"] for r in ok] or [0]
            rt = [r["reasoning_tokens"] for r in ok] or [0]
            think = sum(1 for r in ok if r["has_think_tag"] or r["has_reasoning_content"])
            tag = sum(1 for r in ok if r["has_think_tag"])
            bad = [r for r in rows if r.get("err") or r.get("http")]

            summary[effort] = {
                "ok": f"{len(ok)}/{n}",
                "median_ms": statistics.median(times),
                "max_ms": max(times),
                "completion_tokens": statistics.median(ct),
                "reasoning_tokens": statistics.median(rt),
                "responses_with_thinking": f"{think}/{len(ok)}",
                "think_tag_in_content": f"{tag}/{len(ok)}",
                "errors": [r.get("err") or r.get("http") for r in bad][:3],
            }
            print(
                f"{effort:<10}{len(ok)}/{n:<5}{statistics.median(times):>9.0f}{max(times):>9}"
                f"{statistics.median(ct):>9.0f}{statistics.median(rt):>9.0f}"
                f"{f'{think}/{len(ok)}':>8}{f'{tag}/{len(ok)}':>18}"
            )
            for r in bad[:2]:
                print(f"           失败: {r.get('err') or r.get('http')}")

    base_ct = summary.get("max", {}).get("completion_tokens", 0) or 1
    print(f"\n  相对 effort=max 的完成 token 降幅：")
    for effort in EFFORTS:
        ct = summary[effort]["completion_tokens"]
        print(f"    {effort:<8} {ct:>6} tok  ({(1 - ct / base_ct) * 100:5.1f}% 少)   "
              f"中位 {summary[effort]['median_ms']:.0f}ms")

    out = ROOT / "probe_out" / f"minimax_effort_{model.replace('.', '_')}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  => {out}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--model", default="MiniMax-M3.1-Flash-Preview")
    a = p.parse_args()
    raise SystemExit(asyncio.run(run(a.n, a.model)))
