"""Can a SiliconFlow model do BOTH vision AND tool calls? (the executor's job)

The executor is the only role that must (a) see the screenshot and (b) answer
with a tool call (tap / swipe / type). SiliconFlow's vision models were
measured to reject a forced ``tool_choice`` and to return empty content when
only ``tools`` is offered — if that holds for an executor-shaped prompt, the
executor role cannot be served by SiliconFlow at all.

Run:
    SF_KEY=sk-... python scripts/probe_siliconflow_executor.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import urllib.error
import urllib.request

KEY = os.environ.get("SF_KEY", "")
if not KEY:
    sys.exit("SF_KEY is not set")
BASE = os.environ.get("SF_BASE", "https://api.siliconflow.cn/v1")


def png_data_url() -> str:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (240, 160), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 120, 140], fill="black")
    draw.text((140, 70), "OK", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tap",
            "description": "Tap on the element with the given index on the screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "element index"},
                    "reasoning": {"type": "string", "description": "why"},
                },
                "required": ["index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete",
            "description": "Report that the task is complete.",
            "parameters": {
                "type": "object",
                "properties": {"reasoning": {"type": "string"}},
                "required": ["reasoning"],
            },
        },
    },
]

SYSTEM = (
    "You are a mobile phone operator. Look at the screenshot, decide the next "
    "action, and call exactly one tool."
)


def post(model: str, with_image: bool, force: bool) -> tuple[int, str]:
    content: list[dict] = [{"type": "text", "text": "Tap the black rectangle on the left."}]
    if with_image:
        content.append({"type": "image_url", "image_url": {"url": png_data_url()}})
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
        ],
        "max_tokens": 256,
        "tools": TOOLS,
        "parallel_tool_calls": False,
    }
    if force:
        payload["tool_choice"] = {"type": "function", "function": {"name": "tap"}}
    else:
        payload["tool_choice"] = "auto"

    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            msg = body["choices"][0]["message"]
            if msg.get("tool_calls"):
                fn = msg["tool_calls"][0]["function"]
                return 200, f"TOOL_CALL {fn['name']}({fn['arguments'][:120]})"
            return 200, f"no tool_calls; content={repr((msg.get('content') or '')[:160])}"
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:200]
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


MODELS = [
    ("Qwen/Qwen3-VL-32B-Instruct", True),
    ("Qwen/Qwen3-VL-8B-Instruct", True),
    ("zai-org/GLM-4.5V", True),
    ("Qwen/Qwen3-Omni-30B-A3B-Instruct", True),
    ("Qwen/Qwen3.5-4B", False),
    ("Qwen/Qwen2.5-72B-Instruct", False),
]

if __name__ == "__main__":
    for model, with_image in MODELS:
        print(f"\n########## {model} (image={with_image}) ##########")
        print("  auto tool_choice :", post(model, with_image, force=False))
        print("  forced tool_choice:", post(model, with_image, force=True))
