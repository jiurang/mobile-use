"""探针：DeepSeek 各模型在「强制 tool_choice」下的真实行为。

要回答的问题：manager 里那条 deepseek 渠道为什么一直失败？
先确认 DeepSeek 现在到底支持哪些模型名，以及强制 tool_choice 在哪些模型上会 400。

跑法：
    .\\.venv\\Scripts\\python.exe scripts/probe_deepseek_models.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

BASE = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Probe",
            "description": "p",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
        },
    }
]


async def main() -> int:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("DEEPSEEK_API_KEY 没配。")
        return 1
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    url = BASE + "/chat/completions"

    async with httpx_client() as c:
        r = await c.get(BASE + "/models", headers=headers)
        if r.status_code == 200:
            ids = [m.get("id") for m in (r.json().get("data") or [])]
            print(f"可用模型: {ids}")
        else:
            print(f"/models HTTP {r.status_code}: {r.text[:200]}")

        for model in ["deepseek-chat", "deepseek-reasoner", "deepseek-flash"]:
            for label, extra in (
                (
                    "强制tool_choice",
                    {"tool_choice": {"type": "function", "function": {"name": "Probe"}}},
                ),
                (
                    "强制tool_choice+thinking.disabled",
                    {
                        "tool_choice": {"type": "function", "function": {"name": "Probe"}},
                        "thinking": {"type": "disabled"},
                    },
                ),
            ):
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": TOOLS,
                    **extra,
                }
                r = await c.post(url, json=payload, headers=headers)
                if r.status_code != 200:
                    try:
                        detail = json.dumps(r.json(), ensure_ascii=False)[:200]
                    except Exception:  # noqa: BLE001
                        detail = r.text[:200]
                    print(f"  {model:<18} {label:<36} HTTP {r.status_code}: {detail}")
                else:
                    msg = (r.json().get("choices") or [{}])[0].get("message") or {}
                    print(
                        f"  {model:<18} {label:<36} HTTP 200 "
                        f"tool_calls={len(msg.get('tool_calls') or [])} "
                        f"reasoning={bool(msg.get('reasoning_content'))}"
                    )
    return 0


def httpx_client():
    import httpx

    return httpx.AsyncClient(timeout=120)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
