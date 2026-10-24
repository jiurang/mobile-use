"""真 API 实测：**一张截图在 MiniMax 上值多少 token**，以及怎么压。

为什么要量
----------
图是「缓存尾巴」的一部分（图和元素表一样每轮重发）。尾巴越大 ⇒
命中率越低、每轮越贵。所以「图能不能更小」是个直接的杠杆。

⚠️ **别用 tile 公式估算**（仓库 `utils/image_compress.py` 里那张
``(w//768+1)*(h//768+1)*258`` 是 OpenAI/Gemini 的口径）—— 不同 provider
口径不一样，**以真 API 的 prompt_tokens 为准**。

跑法：
    .venv\\Scripts\\python.exe scripts\\probe_image_token_cost.py
"""

from __future__ import annotations

import base64
import json
import sys
import time
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minitap"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

import httpx  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from probe_executor_cache_measure import cached_of, load_env  # noqa: E402

MODEL = "MiniMax-M3.1-Flash-Preview"
ENV = load_env()
API_KEY = ENV["MINIMAX_API_KEY"]
BASE = ENV.get("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")

#: 候选尺寸：原图 / 仓库默认 / 更小 / 极小
SIZES = [(1220, 2712), (640, 1422), (480, 1066), (320, 711)]


def make_jpeg(w: int, h: int) -> str:
    img = Image.new("RGB", (w, h), (40, 120, 60))
    d = ImageDraw.Draw(img)
    for i in range(0, h, max(1, h // 12)):
        d.rectangle([10, i, w - 10, i + h // 24], fill=(200, 60 + i % 120, 30))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=50, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_once(messages: list[dict], tag: str) -> dict:
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 16,
        "temperature": 1,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    last = ""
    for attempt in range(6):
        try:
            with httpx.Client(timeout=180.0) as client:
                resp = client.post(f"{BASE}/chat/completions", json=body, headers=headers)
            if resp.status_code == 200:
                return resp.json()
            last = f"HTTP {resp.status_code}: {resp.text[:160]}"
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:120]}"
        time.sleep(3.0 * (attempt + 1))
    raise RuntimeError(last)


def main() -> int:
    # 基线：同样的文字，不带图
    base_text = "How many pixels wide is the image? Answer with one number. " + "x" * 40
    rows = []
    base = call_once([{"role": "user", "content": base_text}], "base")
    base_tokens = (base.get("usage") or {}).get("prompt_tokens", 0)
    print(f"基线（无图）prompt_tokens = {base_tokens}\n")

    for w, h in SIZES:
        b64 = make_jpeg(w, h)
        url = f"data:image/jpeg;base64,{b64}"
        res = call_once(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": base_text},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ],
            "img",
        )
        u = res.get("usage") or {}
        p = u.get("prompt_tokens", 0)
        kb = len(b64) * 3 // 4 // 1024
        rows.append(
            {
                "size": f"{w}x{h}",
                "kb": kb,
                "prompt_tokens": p,
                "image_tokens": p - base_tokens,
                "cached": cached_of(u),
            }
        )
        print(
            f"  {w:>5}x{h:<5}  {kb:>3}KB  prompt_tokens={p:>6,}  图占 ≈{p - base_tokens:>5,} token"
        )
        time.sleep(0.5)

    out = ROOT / "probe_out" / "image_token_cost.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"baseline": base_tokens, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  => {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
