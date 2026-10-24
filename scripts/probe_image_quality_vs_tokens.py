"""真 API 实测：**JPEG 质量压到什么程度，会影响图片的 token 计价吗？**

用户的约束是「**不要压宽高，只压质量**」—— 这条约束值不值，取决于
MiniMax 给图片计价到底看什么：

* 若 token 只跟**像素尺寸**有关（tile 计价）⇒ 压质量**一个 token 都省不了**，
  只省上传字节；宽高才是唯一的 token 杠杆。
* 若 token 也跟**编码后的体积/质量**有关 ⇒ 压质量能省钱。

这个脚本一次把两件事都测出来（同一张图，只改质量 / 只改尺寸）。

跑法：
    .venv\\Scripts\\python.exe scripts\\probe_image_quality_vs_tokens.py
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

from probe_executor_cache_measure import load_env  # noqa: E402

MODEL = "MiniMax-M3.1-Flash-Preview"
ENV = load_env()
API_KEY = ENV["MINIMAX_API_KEY"]
BASE = ENV.get("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")

TEXT = "Describe this screen in one short sentence. " + "pad " * 30


def make_jpeg(w: int, h: int, quality: int) -> str:
    img = Image.new("RGB", (w, h), (40, 120, 60))
    d = ImageDraw.Draw(img)
    for i in range(0, h, max(1, h // 14)):
        d.rectangle([10, i, w - 10, i + h // 28], fill=(200, 60 + i % 120, 30))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_once(b64: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TEXT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 32,
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
    # 基线：只有文字
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": TEXT}],
        "max_tokens": 16,
        "temperature": 1,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    with httpx.Client(timeout=180.0) as client:
        base = client.post(f"{BASE}/chat/completions", json=body, headers=headers).json()
    base_tokens = (base.get("usage") or {}).get("prompt_tokens", 0)
    print(f"基线（无图）prompt_tokens = {base_tokens}\n")

    rows = []
    print("【只改质量，尺寸固定 1220x2712】")
    for q in (95, 75, 50, 30, 15):
        b64 = make_jpeg(1220, 2712, q)
        kb = len(b64) * 3 // 4 // 1024
        u = (call_once(b64).get("usage") or {})
        p = u.get("prompt_tokens", 0)
        rows.append({"case": "quality", "size": "1220x2712", "quality": q, "kb": kb, "tokens": p})
        print(f"  q={q:>3}  {kb:>3}KB  prompt_tokens={p:>6,}  图占 ≈{p - base_tokens:>5,}")
        time.sleep(0.4)

    print("\n【只改尺寸，质量固定 50】")
    for w, h in ((1220, 2712), (640, 1422), (480, 1066), (320, 711)):
        b64 = make_jpeg(w, h, 50)
        kb = len(b64) * 3 // 4 // 1024
        u = (call_once(b64).get("usage") or {})
        p = u.get("prompt_tokens", 0)
        rows.append({"case": "size", "size": f"{w}x{h}", "quality": 50, "kb": kb, "tokens": p})
        print(f"  {w:>5}x{h:<5}  {kb:>3}KB  prompt_tokens={p:>6,}  图占 ≈{p - base_tokens:>5,}")
        time.sleep(0.4)

    out = ROOT / "probe_out" / "image_quality_vs_tokens.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"baseline": base_tokens, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  => {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
