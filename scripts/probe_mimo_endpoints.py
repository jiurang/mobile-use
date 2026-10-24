"""探针：小米 MiMo Token Plan 的端点到底存不存在、长什么样。

为什么单独探
------------
渠道目录里的默认端点必须是**真的能用**的那个 —— 填错了的症状是
「请求带着我的 key 打过去，对方报鉴权失败」，而错误信息完全指不到真正的原因
（deepseek 渠道指着 MiniMax 地址那次就是这样）。

所以：拿一把**假的** key 打过去，如果返回的是「鉴权失败」而不是「域名解析不了 /
404 / 连接超时」，就说明**地址是对的**（服务在那儿，只是没给对的 key）。

跑法：
    .\\.venv\\Scripts\\python.exe scripts/probe_mimo_endpoints.py
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

#: (名字, Base URL, 模型)
TARGETS = [
    ("TokenPlan 中国 / OpenAI 协议", "https://token-plan-cn.xiaomimimo.com/v1", "mimo-v2.6-pro"),
    (
        "TokenPlan 中国 / Anthropic 协议",
        "https://token-plan-cn.xiaomimimo.com/anthropic",
        "mimo-v2.6-pro",
    ),
    ("TokenPlan 新加坡 / OpenAI 协议", "https://token-plan-sgp.xiaomimimo.com/v1", "mimo-v2.6-pro"),
    ("TokenPlan 欧洲 / OpenAI 协议", "https://token-plan-ams.xiaomimimo.com/v1", "mimo-v2.6-pro"),
    ("按量付费 / OpenAI 协议", "https://api.xiaomimimo.com/v1", "mimo-v2.6-pro"),
]

FAKE_KEY = "tp-probe-not-a-real-key"


async def main() -> int:
    import httpx

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
        for name, base, model in TARGETS:
            url = f"{base}/chat/completions"
            headers = {
                "Authorization": f"Bearer {FAKE_KEY}",
                "api-key": FAKE_KEY,
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 8,
            }
            try:
                r = await c.post(url, json=payload, headers=headers)
            except Exception as e:  # noqa: BLE001
                print(f"  {name:<32} ❌ 连接失败 {type(e).__name__}: {e}")
                continue
            try:
                detail = json.dumps(r.json(), ensure_ascii=False)[:180]
            except Exception:  # noqa: BLE001
                detail = r.text[:180]
            verdict = (
                "✅ 端点存在（只是 key 不对）" if r.status_code in (401, 403) else "⚠️ 别的状态"
            )
            print(f"  {name:<32} {verdict} HTTP {r.status_code}: {detail}")

    # 真的 key 在环境里的话，顺手验证一次。
    key = os.environ.get("MIMO_API_KEY") or os.environ.get("MIMO_TOKEN_PLAN_API_KEY")
    if key:
        print("\n  环境里有 MIMO key，验一次真的调用：")
        base = "https://token-plan-cn.xiaomimimo.com/v1"
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(
                f"{base}/chat/completions",
                json={
                    "model": "mimo-v2.6-pro",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_completion_tokens": 8,
                },
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
            print(f"    HTTP {r.status_code}: {r.text[:200]}")
    else:
        print("\n  （环境里没有 MIMO_API_KEY —— 端点存在性已验证，真调用要用户自己的 tp- key）")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
