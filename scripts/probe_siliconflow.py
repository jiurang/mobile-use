"""Probe the live SiliconFlow OpenAI-compatible API.

Key comes from the SF_KEY env var (never written into the file).

Run:
    SF_KEY=sk-... python scripts/probe_siliconflow.py models
    SF_KEY=sk-... python scripts/probe_siliconflow.py features
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("SF_BASE", "https://api.siliconflow.cn/v1")
KEY = os.environ.get("SF_KEY", "")
if not KEY:
    sys.exit("SF_KEY is not set")

TIMEOUT = 90


def _request(method: str, path: str, payload: dict | None = None) -> tuple[int, object]:
    url = BASE + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = resp.read().decode("utf-8", "replace")
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, body[:800]
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


def chat(model: str, messages: list[dict], **extra) -> tuple[int, object]:
    payload = {"model": model, "messages": messages, "max_tokens": 512}
    payload.update(extra)
    return _request("POST", "/chat/completions", payload)


def show(tag: str, status: int, body: object) -> None:
    print(f"\n===== {tag} | HTTP {status} =====")
    if isinstance(body, dict) and "choices" in body:
        msg = body["choices"][0].get("message", {})
        print("content:", repr((msg.get("content") or "")[:400]))
        print("tool_calls:", json.dumps(msg.get("tool_calls"), ensure_ascii=False)[:400])
        print("finish_reason:", body["choices"][0].get("finish_reason"))
        print("usage:", json.dumps(body.get("usage"), ensure_ascii=False))
    else:
        print(json.dumps(body, ensure_ascii=False)[:900])


def cmd_models() -> None:
    status, body = _request("GET", "/models")
    print(f"HTTP {status}")
    if isinstance(body, dict) and "data" in body:
        ids = [m.get("id", "?") for m in body["data"]]
        print(f"total models: {len(ids)}")
        for mid in sorted(ids):
            print(" ", mid)
    else:
        print(json.dumps(body, ensure_ascii=False)[:900])


def cmd_features() -> None:
    text_model = os.environ.get("SF_TEXT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    vl_model = os.environ.get("SF_VL_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")

    # A. plain chat
    show(
        "A plain chat",
        *chat(text_model, [{"role": "user", "content": "说「ok」两个字"}]),
    )

    # B. response_format = json_object
    show(
        "B json_object",
        *chat(
            text_model,
            [
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": "给上海今天天气，返回 JSON，字段 city / status"},
            ],
            response_format={"type": "json_object"},
        ),
    )

    # C. response_format = json_schema  (the critical one for mobile-use)
    schema = {
        "name": "CityStatus",
        "schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "status": {"type": "string"},
            },
            "required": ["city", "status"],
            "additionalProperties": False,
        },
    }
    show(
        "C json_schema",
        *chat(
            text_model,
            [{"role": "user", "content": "上海现在怎么样？"}],
            response_format={"type": "json_schema", "json_schema": schema},
        ),
    )

    # D. tool calling (executor path)
    show(
        "D tool_calls",
        *chat(
            text_model,
            [{"role": "user", "content": "点一下屏幕坐标 100,200"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "tap",
                        "description": "tap at x,y",
                        "parameters": {
                            "type": "object",
                            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                            "required": ["x", "y"],
                        },
                    },
                }
            ],
            tool_choice="auto",
        ),
    )

    # E. vision
    show(
        "E vision (VL)",
        *chat(
            vl_model,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "这张图里有什么？一句话。"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
                            },
                        },
                    ],
                }
            ],
        ),
    )

    # F. json_schema on the VL model (does VL support it?)
    show(
        "F json_schema on VL",
        *chat(
            vl_model,
            [{"role": "user", "content": "上海现在怎么样？"}],
            response_format={"type": "json_schema", "json_schema": schema},
        ),
    )


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "models"
    if cmd == "models":
        cmd_models()
    elif cmd == "features":
        cmd_features()
    else:
        sys.exit(f"unknown command: {cmd}")
