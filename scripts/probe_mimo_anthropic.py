"""探针：MiMo Token Plan 的 Anthropic 兼容端点（路径是 /v1/messages，不是 chat/completions）。

跑法：
    .\\.venv\\Scripts\\python.exe scripts/probe_mimo_anthropic.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))

FAKE_KEY = "tp-probe-not-a-real-key"


async def main() -> int:
    import httpx

    targets = [
        (
            "TokenPlan 中国 / Anthropic 协议",
            "https://token-plan-cn.xiaomimimo.com/anthropic/v1/messages",
        ),
        ("按量付费 / Anthropic 协议", "https://api.xiaomimimo.com/anthropic/v1/messages"),
    ]
    async with httpx.AsyncClient(timeout=30) as c:
        for name, url in targets:
            r = await c.post(
                url,
                json={
                    "model": "mimo-v2.6-pro",
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={
                    "x-api-key": FAKE_KEY,
                    "api-key": FAKE_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
            )
            try:
                detail = json.dumps(r.json(), ensure_ascii=False)[:180]
            except Exception:  # noqa: BLE001
                detail = r.text[:180]
            verdict = (
                "✅ 端点存在（只是 key 不对）" if r.status_code in (401, 403) else "⚠️ 别的状态"
            )
            print(f"  {name:<32} {verdict} HTTP {r.status_code}: {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
