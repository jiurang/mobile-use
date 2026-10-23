"""一次性探针：用新给的 key 直接打两个平台，看它对谁有效。

跑的��:
    .\\.venv\\Scripts\\python.exe scripts/probe_key_where.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))

CANDIDATE = os.environ.get("CANDIDATE_KEY", "")

TARGETS = [
    ("deepseek", "https://api.deepseek.com/chat/completions", {"model": "deepseek-chat"}),
    ("minimax", "https://api.minimaxi.com/v1/chat/completions", {"model": "MiniMax-M3"}),
]


async def main() -> int:
    if not CANDIDATE:
        print("CANDIDATE_KEY 没给")
        return 1

    import httpx

    async with httpx.AsyncClient(timeout=60) as c:
        for name, url, extra in TARGETS:
            payload = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, **extra}
            try:
                r = await c.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {CANDIDATE}",
                        "Content-Type": "application/json",
                    },
                )
            except Exception as e:  # noqa: BLE001
                print(f"  {name:<10} 请求异常: {type(e).__name__}: {e}")
                continue
            if r.status_code == 200:
                body = r.json()
                msg = (body.get("choices") or [{}])[0].get("message") or {}
                print(f"  {name:<10} ✅ HTTP 200  model={body.get('model')} content={str(msg.get('content'))[:40]!r}")
            else:
                try:
                    detail = json.dumps(r.json(), ensure_ascii=False)[:220]
                except Exception:  # noqa: BLE001
                    detail = r.text[:220]
                print(f"  {name:<10} ❌ HTTP {r.status_code}: {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
