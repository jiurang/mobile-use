"""绕过 langchain，直接打 DeepSeek 原始 API，看清工具调用到底返回了什么。

为什么必须绕过
--------------
用真实 cortex prompt 测 DeepSeek（DEEPSEEK_THINKING=false）得到 5/6，其中一次报：

    ValueError: MiniMax response had no tool_calls and no content.
    finish_reason='tool_calls'

`finish_reason='tool_calls'` 和「一个工具调用都没有」是**自相矛盾**的 —— 服务端说
它调用了工具，客户端却什么都没拿到。这说明内容在 langchain 的解析层被吞了，
而不是模型没返回。想知道是什么吞的，就得看原始 HTTP 响应。

（顺带：那句报错文案硬编码了 "MiniMax"，DeepSeek 走同一条路径也会这么显示 ——
文案 bug，会误导排查。）

跑法：
    uv run python scripts/probe_deepseek_raw.py --n 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("DEEPSEEK_THINKING", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))
sys.path.insert(0, str(ROOT / "scripts"))

from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402

from probe_cortex_real_prompt import build_messages  # noqa: E402


def to_openai_messages(messages) -> list[dict]:
    out = []
    for m in messages:
        content = m.content
        if isinstance(content, list):  # 图片块
            parts = []
            for item in content:
                if item.get("type") == "image_url":
                    parts.append({"type": "image_url", "image_url": {"url": item["image_url"]["url"]}})
                elif item.get("type") == "text":
                    parts.append({"type": "text", "text": item["text"]})
            out.append({"role": "user", "content": parts})
        else:
            role = "system" if m.type == "system" else "user"
            out.append({"role": role, "content": content})
    return out


async def run(n: int) -> int:
    import time

    import httpx

    from minitap.mobile_use.agents.cortex.types import CortexOutput

    messages, meta = build_messages()
    tools = [convert_to_openai_tool(CortexOutput)]

    payload = {
        "model": "deepseek-chat",
        "messages": to_openai_messages(messages),
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": tools[0]["function"]["name"]}},
        "thinking": {"type": "disabled"},
    }

    url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com") + "/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}", "Content-Type": "application/json"}

    print(f"\n=== DeepSeek 原始 API，N={n} ===")
    print(f"    {meta['system_chars']} 字 system + {meta['screenshot_b64_kb']}KB 截图\n")

    shapes = []
    times = []
    async with httpx.AsyncClient(timeout=120) as client:
        for i in range(1, n + 1):
            t0 = time.perf_counter()
            r = await client.post(url, json=payload, headers=headers)
            ms = round((time.perf_counter() - t0) * 1000)
            times.append(ms)
            try:
                body = r.json()
            except Exception:  # noqa: BLE001
                print(f"  [{i}] HTTP {r.status_code} 非 JSON 响应: {r.text[:200]}")
                continue

            if r.status_code != 200:
                print(f"  [{i}] HTTP {r.status_code}: {json.dumps(body, ensure_ascii=False)[:200]}")
                continue

            choices = body.get("choices") or []
            if not choices:
                print(f"  [{i}] choices 为空，完整响应: {json.dumps(body, ensure_ascii=False)[:400]}")
                shapes.append({"i": i, "shape": "no_choices"})
                continue

            msg = choices[0].get("message") or {}
            finish = choices[0].get("finish_reason")
            tcs = msg.get("tool_calls")
            content = msg.get("content")
            reasoning = msg.get("reasoning_content")

            shape = {
                "i": i,
                "ms": ms,
                "finish_reason": finish,
                "has_tool_calls": bool(tcs),
                "n_tool_calls": len(tcs or []),
                "content_is_none": content is None,
                "content_len": len(content) if isinstance(content, str) else -1,
                "has_reasoning": bool(reasoning),
                "msg_keys": sorted(msg.keys()),
            }
            if tcs:
                shape["tc0_keys"] = sorted(tcs[0].keys())
                shape["tc0_name"] = tcs[0].get("function", {}).get("name")
                shape["tc0_arg_type"] = type(
                    tcs[0].get("function", {}).get("arguments")
                ).__name__
                shape["tc0_args_preview"] = str(tcs[0].get("function", {}).get("arguments"))[:160]
            shapes.append(shape)

            flag = "OK " if tcs else "空!"
            print(f"  [{i:>2}] {flag} {ms:>6}ms  finish={finish}  tool_calls={len(tcs or [])}  "
                  f"content={shape['content_len']}  keys={shape['msg_keys']}")

    n_empty = sum(1 for s in shapes if not s.get("has_tool_calls"))
    print(f"\n  空 tool_calls: {n_empty}/{len(shapes)}")
    if times:
        print(f"  耗时 中位 {statistics.median(sorted(times))}ms")

    out = ROOT / "probe_out" / "deepseek_raw_probe.json"
    out.write_text(json.dumps(shapes, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  => {out}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8)
    a = p.parse_args()
    raise SystemExit(asyncio.run(run(a.n)))
