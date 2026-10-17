"""直接打 MiniMax 原始 API，验证 `thinking: {"type":"disabled"}` 到底有没有生效。

为什么必须绕过 langchain
----------------------
langchain 路径上实测：
    thinking 开   -> 12/12 成功，中位 7160ms
    thinking 关   -> 11/12 成功，中位 7584ms  ← 没变快，响应里还有 <think>
这说明 `extra_body` 可能根本没进请求体，或者 M3 忽略了这个参数。
langchain 这一层还拿不到 token 用量，而**成本正是要判断的东西**。

所以这里用 httpx 直发，带上和 langchain 一样的参数，再加一组显式 thinking，
对比三件事：finish_reason / 响应里有没有 <think> / completion_tokens。

跑法：
    uv run python scripts/probe_minimax_raw.py --n 4
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

VARIANTS = [
    ("省略 thinking(服务端默认)", {}),
    ("thinking=adaptive", {"thinking": {"type": "adaptive"}}),
    ("thinking=disabled", {"thinking": {"type": "disabled"}}),
]


async def one(client, url, headers, base_payload, extra: dict) -> dict:
    payload = {**base_payload, **extra}
    t0 = time.perf_counter()
    r = await client.post(url, json=payload, headers=headers)
    ms = round((time.perf_counter() - t0) * 1000)
    if r.status_code != 200:
        return {"ms": ms, "http": r.status_code, "err": r.text[:200]}
    body = r.json()
    ch = (body.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    usage = body.get("usage") or {}
    content = msg.get("content") or ""
    return {
        "ms": ms,
        "finish_reason": ch.get("finish_reason"),
        "n_tool_calls": len(msg.get("tool_calls") or []),
        "content_len": len(content),
        "has_think_tag": "<think>" in content,
        "has_reasoning_content": bool(msg.get("reasoning_content")),
        "reasoning_len": len(msg.get("reasoning_content") or ""),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "completion_tokens_details": usage.get("completion_tokens_details"),
    }


async def run(n: int) -> int:
    import httpx

    messages, meta = build_messages()
    tools = [convert_to_openai_tool(CortexOutput)]
    base_payload = {
        "model": "MiniMax-M3",
        "messages": to_openai_messages(messages),
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": tools[0]["function"]["name"]}},
        "temperature": 1,
    }
    url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1") + "/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ['MINIMAX_API_KEY']}", "Content-Type": "application/json"}

    print(f"\n=== MiniMax-M3 原始 API，prompt={meta['system_chars']} 字 + {meta['screenshot_b64_kb']}KB 截图 ===")

    summary = {}
    async with httpx.AsyncClient(timeout=180) as client:
        for label, extra in VARIANTS:
            rows = []
            for _ in range(n):
                try:
                    rows.append(await one(client, url, headers, base_payload, extra))
                except Exception as e:  # noqa: BLE001
                    rows.append({"ms": 0, "err": f"{type(e).__name__}: {e}"})
                await asyncio.sleep(0.4)

            ok = [r for r in rows if not r.get("err") and not r.get("http")]
            times = sorted(r["ms"] for r in ok) or [0]
            bad = [r for r in rows if r.get("err") or r.get("http")]
            think = sum(1 for r in ok if r.get("has_think_tag") or r.get("has_reasoning_content"))
            ctoks = [r.get("completion_tokens") or 0 for r in ok]
            ptoks = [r.get("prompt_tokens") or 0 for r in ok]

            summary[label] = {
                "ok": f"{len(ok)}/{n}",
                "median_ms": statistics.median(times),
                "max_ms": max(times),
                "有思维内容": f"{think}/{len(ok)}",
                "completion_tokens中位": statistics.median(ctoks) if ctoks else 0,
                "prompt_tokens中位": statistics.median(ptoks) if ptoks else 0,
            }

            print(f"\n  【{label}】")
            print(f"    成功 {len(ok)}/{n}   耗时中位 {statistics.median(times):.0f}ms  最大 {max(times)}ms")
            print(f"    响应里带思维内容: {think}/{len(ok)}")
            print(f"    prompt_tokens 中位 {statistics.median(ptoks) if ptoks else 0}"
                  f"   completion_tokens 中位 {statistics.median(ctoks) if ctoks else 0}")
            for r in bad[:2]:
                print(f"    失败: {r.get('err') or r.get('http')}")
            finish = {}
            for r in ok:
                finish[r.get("finish_reason")] = finish.get(r.get("finish_reason"), 0) + 1
            print(f"    finish_reason 分布: {finish}")

    out = ROOT / "probe_out" / "minimax_raw_probe.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  => {out}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4)
    a = p.parse_args()
    raise SystemExit(asyncio.run(run(a.n)))
