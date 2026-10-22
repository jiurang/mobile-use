"""Does SiliconFlow's automatic prefix cache work, and can mobile-use see it?

Two views of the same calls:
  * raw HTTP  -> SiliconFlow's own usage fields (prompt_cache_hit_tokens ...)
  * langchain -> what mobile-use's telemetry actually reads (cache_read)

Run:
    SF_KEY=sk-... python scripts/probe_siliconflow_cache.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

KEY = os.environ.get("SF_KEY", "")
if not KEY:
    sys.exit("SF_KEY is not set")

SF_BASE = os.environ.get("SF_BASE", "https://api.siliconflow.cn/v1")
MODEL = os.environ.get("SF_TEXT_MODEL", "Qwen/Qwen2.5-72B-Instruct")

# A long, stable prefix — the shape mobile-use's prompts have (system prompt
# + tool schemas + perception block all sit in front of the volatile tail).
PREFIX = (
    "你是一个手机操作智能体。以下是操作规范，请严格遵守。\n"
    + "".join(f"规则 {i}：当屏幕上出现不确定的控件时，先用 ui_hierarchy 确认 bounds，再执行点击。\n" for i in range(1, 120))
)
QUESTION = "现在屏幕上有一个「确定」按钮，下一步做什么？"


def raw_chat(tag: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PREFIX},
            {"role": "user", "content": QUESTION},
        ],
        "max_tokens": 64,
    }
    req = urllib.request.Request(
        SF_BASE + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {"error": exc.read().decode("utf-8", "replace")[:300]}
    usage = body.get("usage", {})
    print(f"\n--- raw HTTP {tag} ---")
    print("usage:", json.dumps(usage, ensure_ascii=False))
    return usage


async def main() -> None:
    from minitap.mobile_use.services.llm import ProviderOverride, get_openai_llm

    # --- raw view: SiliconFlow's own numbers ---
    print(f"prefix chars: {len(PREFIX)}")
    u1 = raw_chat("call 1 (cold)")
    u2 = raw_chat("call 2 (same prefix)")

    # --- langchain view: what mobile-use's telemetry reads ---
    override = ProviderOverride(api_key=KEY, base_url=SF_BASE)
    client = get_openai_llm(MODEL, 1, override)
    for tag in ("call 3 (langchain ainvoke)", "call 4 (langchain ainvoke, same prefix)"):
        msg = await client.ainvoke(
            [{"role": "system", "content": PREFIX}, {"role": "user", "content": QUESTION}]
        )
        print(f"\n--- {tag} ---")
        print("usage_metadata:", msg.usage_metadata)

    print("\n===== verdict =====")
    for name, u in (("raw1", u1), ("raw2", u2)):
        print(
            f"{name}: prompt_tokens={u.get('prompt_tokens')} "
            f"cached_tokens={u.get('prompt_tokens_details', {}).get('cached_tokens')} "
            f"prompt_cache_hit_tokens={u.get('prompt_cache_hit_tokens')} "
            f"prompt_cache_miss_tokens={u.get('prompt_cache_miss_tokens')}"
        )


if __name__ == "__main__":
    asyncio.run(main())
