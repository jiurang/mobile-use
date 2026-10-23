"""一次性探针：检查 .env 里各家的 key 现在还对不对。

跑的��:
    .\\.venv\\Scripts\\python.exe scripts/probe_env_keys.py
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

#: (名字, 端点, 模型, .env 变量名)
TARGETS = [
    ("deepseek", "https://api.deepseek.com/chat/completions", "deepseek-chat", "DEEPSEEK_API_KEY"),
    ("minimax", "https://api.minimaxi.com/v1/chat/completions", "MiniMax-M3", "MINIMAX_API_KEY"),
    ("siliconflow", "https://api.siliconflow.cn/v1/chat/completions", "Qwen/Qwen2.5-7B-Instruct", "SILICONFLOW_API_KEY"),
    ("anthropic", "https://api.anthropic.com/v1/messages", "claude-haiku-4-5-20251001", "ANTHROPIC_API_KEY"),
]


async def main() -> int:
    import httpx

    async with httpx.AsyncClient(timeout=60) as c:
        for name, url, model, env in TARGETS:
            key = os.environ.get(env)
            if not key:
                print(f"  {name:<12} – .env 里没有 {env}")
                continue
            if name == "anthropic":
                headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
                payload = {"model": model, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
            else:
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                payload = {"model": model, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
            try:
                r = await c.post(url, json=payload, headers=headers)
            except Exception as e:  # noqa: BLE001
                print(f"  {name:<12} ❌ 请求异常 {type(e).__name__}: {e}")
                continue
            if r.status_code == 200:
                print(f"  {name:<12} ✅ HTTP 200  (key 尾号 {key[-4:]})")
            else:
                try:
                    detail = json.dumps(r.json(), ensure_ascii=False)[:200]
                except Exception:  # noqa: BLE001
                    detail = r.text[:200]
                print(f"  {name:<12} ❌ HTTP {r.status_code} (key 尾号 {key[-4:]}): {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
