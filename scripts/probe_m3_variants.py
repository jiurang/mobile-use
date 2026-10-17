"""探测「MiniMax-M3.1 有没有非 Flash 的 max 档」。

为什么必须实测而不是查清单
--------------------------
`GET /v1/models` 返回的列表只是**这个账号当前可见**的模型，不是平台全量。
有些模型文档里有、清单里没列（内测/灰度/按 key 开通），直接打一次才知道。
反过来，清单里有的也可能因为没开通而 400。

用户要的是「max 版」，有两种完全不同的可能：
    A. 存在一个更贵的非 Flash 模型（比如 MiniMax-M3.1）——换过去
    B. 指 output_config.effort = max（M3.1 的思考深度档）——但官方说默认就是 max
两种含义要靠实测区分，不能猜。

跑法：
    uv run python scripts/probe_m3_variants.py
"""

from __future__ import annotations

import asyncio
import json
import os
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

# 候选：清单里的 + 猜出来的非 Flash 变体 + M2.7-highspeed 作为「同代更快档」的对照
CANDIDATES = [
    "MiniMax-M3.1-Flash-Preview",  # 清单里有（当前主力）
    "MiniMax-M3.1",                # 猜：非 Flash 的 3.1
    "MiniMax-M3.1-max",            # 猜：带 max 后缀
    "MiniMax-M3",                  # 已知可用，作为「能打通」的基线
]


async def one(client, url, headers, base: dict, model: str) -> dict:
    payload = {**base, "model": model}
    t0 = time.perf_counter()
    r = await client.post(url, json=payload, headers=headers)
    ms = round((time.perf_counter() - t0) * 1000)
    if r.status_code != 200:
        return {"ms": ms, "http": r.status_code, "err": r.text[:220]}
    body = r.json()
    ch = (body.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    content = msg.get("content") or ""
    return {
        "ms": ms,
        "http": 200,
        "model_returned": body.get("model"),
        "finish_reason": ch.get("finish_reason"),
        "n_tool_calls": len(msg.get("tool_calls") or []),
        "content_len": len(content),
        "has_think_tag": "<think>" in content,
        "prompt_tokens": usage.get("prompt_tokens") or 0,
        "completion_tokens": usage.get("completion_tokens") or 0,
        "reasoning_tokens": details.get("reasoning_tokens") or 0,
    }


async def main() -> int:
    import httpx

    messages, meta = build_messages()
    tools = [convert_to_openai_tool(CortexOutput)]
    base = {
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

    print(f"prompt={meta['system_chars']}字 + {meta['screenshot_b64_kb']}KB 截图\n")
    rows = []
    async with httpx.AsyncClient(timeout=240) as client:
        for model in CANDIDATES:
            try:
                res = await one(client, url, headers, base, model)
            except Exception as e:  # noqa: BLE001
                res = {"ms": 0, "err": f"{type(e).__name__}: {e}"}
            rows.append((model, res))
            if res.get("http") == 200:
                print(f"  [可用] {model:<28} {res['ms']:>6}ms  tool_calls={res['n_tool_calls']}  "
                      f"回显model={res['model_returned']}  completion_tok={res['completion_tokens']}")
            else:
                print(f"  [不可用] {model:<26} {res.get('http') or '-'}  {res.get('err', '')[:150]}")
            await asyncio.sleep(0.5)

    out = ROOT / "probe_out" / "minimax_m3_variants.json"
    out.write_text(
        json.dumps({m: r for m, r in rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  => {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
