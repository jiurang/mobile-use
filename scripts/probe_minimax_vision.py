"""实测 MiniMax 到底能不能吃图 —— 以及**能不能在工具结果里吃图**。

为什么必须实测
--------------
仓库里有过一条图片通道（2026-10-23 下线，见 commit cd2d46d）。它当时用的是
``{"type": "image_url", ...}`` 挂在 **HumanMessage** 上。现在要把它做成
**一个工具**：工具结果是一条 **ToolMessage**，图挂在 tool 消息的 content 里。

这两条路 provider 的支持面**不一定一样**（OpenAI 兼容端点对 role=tool 的
多模态 content 经常比 role=user 更严）。所以这里一次打三组：

    A. 纯文本 user 消息                      —— 基线（token / 延迟）
    B. user 消息里带 image_url               —— 模型到底看不看得懂图
    C. assistant(tool_calls) + tool 消息带图  —— **工具结果带图**（本次要用的形状）

判定「看得懂」不是看 HTTP 200，而是看**回答里能不能读出图里的字** ——
图里画的是 `SCREENSHOT 2026`，模型答不上来就是没看见。

跑法：
    uv run python scripts/probe_minimax_vision.py
    uv run python scripts/probe_minimax_vision.py --model MiniMax-M3.1-Flash-Preview

产物：
    probe_out/minimax_vision_probe.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import sys
import time
from io import BytesIO
from pathlib import Path

# Windows 控制台默认 GBK，打 ✅/❌ 会直接 UnicodeEncodeError 崩掉探针 ——
# 探针崩了就没证据了，所以先把这层兜住。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent

#: 图里写的字。模型答出这个串 ⇒ 真的看见了图。
CANARY = "SCREENSHOT 2026"


def _load_dotenv(path: Path) -> dict[str, str]:
    """极简 .env 读取（只取需要的键，不打印任何密钥）。"""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in values:
            values[key] = val
    return values


def build_canary_jpeg(width: int = 480, height: int = 200) -> str:
    """画一张带醒目文字的图，返回裸 base64（不带 data URI 前缀）。"""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([10, 10, width - 10, height - 10], outline=(0, 0, 0), width=4)
    draw.text((30, 80), CANARY, fill=(0, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_payloads(image_b64: str, model: str) -> list[tuple[str, dict]]:
    """三组要打的消息体。"""
    text_block = {
        "type": "text",
        "text": "What exact text is written in this image? Answer only with the text.",
    }
    image_block = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
    }

    a_text = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: PONG"}],
    }

    b_user_image = {
        "model": model,
        "messages": [{"role": "user", "content": [text_block, image_block]}],
    }

    c_tool_image = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Take a screenshot of the screen and report what text it shows.",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_probe_1",
                        "type": "function",
                        "function": {"name": "screenshot", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_probe_1",
                "content": [
                    {"type": "text", "text": "Screenshot captured. What text is on it?"},
                    image_block,
                ],
            },
        ],
    }

    return [
        ("A. 纯文本 user（基线）", a_text),
        ("B. user 消息带图", b_user_image),
        ("C. tool 消息带图（工具结果带图）", c_tool_image),
    ]


async def call(client, url: str, headers: dict, payload: dict) -> dict:
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=payload, headers=headers, timeout=180)
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "err": f"{type(e).__name__}: {e}",
            "ms": round((time.perf_counter() - t0) * 1000),
        }

    ms = round((time.perf_counter() - t0) * 1000)
    if r.status_code != 200:
        return {"ok": False, "http": r.status_code, "err": r.text[:400], "ms": ms}

    body = r.json()
    choice = (body.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    if isinstance(content, list):  # 有些端点回数组
        content = " ".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
    usage = body.get("usage") or {}
    return {
        "ok": True,
        "ms": ms,
        "finish_reason": choice.get("finish_reason"),
        "answer": content[:300],
        "saw_canary": bool(re.search(r"SCREENSHOT\s*2026", content, re.IGNORECASE)),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


async def run(model: str) -> int:
    import httpx

    env = _load_dotenv(ROOT / ".env")
    api_key = env.get("MINIMAX_API_KEY") or ""
    base_url = (env.get("MINIMAX_BASE_URL") or "https://api.minimaxi.com/v1").rstrip("/")
    if not api_key:
        print("!! .env 里没有 MINIMAX_API_KEY，跑不了")
        return 2

    url = base_url + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    image_b64 = build_canary_jpeg()
    print(f"\n=== {model} :: 图片承载能力实测（图 {len(image_b64) // 1024}KB base64）===")

    results: dict[str, dict] = {}
    async with httpx.AsyncClient(timeout=180) as client:
        for label, payload in build_payloads(image_b64, model):
            res = await call(client, url, headers, payload)
            results[label] = res
            if res.get("ok"):
                print(f"\n  【{label}】HTTP 200  {res['ms']}ms")
                print(f"    回答: {res['answer'][:160]}")
                print(
                    f"    读到图里的字: {'✅ 是' if res['saw_canary'] else '❌ 否（没看见图）'}"
                    f"   prompt_tokens={res['prompt_tokens']}"
                )
            else:
                print(f"\n  【{label}】失败 {res.get('http', '')} {res.get('err', '')[:300]}")

    out = ROOT / "probe_out" / "minimax_vision_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"model": model, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  => {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="MiniMax-M3.1-Flash-Preview")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.model)))
