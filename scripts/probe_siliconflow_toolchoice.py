"""Isolate which SiliconFlow constraint bit us: the model, or the schema?

Observed failure (real langchain path, CortexOutput on Qwen/Qwen3-VL-32B-Instruct):

    400 {'code': 20015, 'message': 'Value error, Specifying functions for
    tool_choice is not yet supported.'}

Hypothesis under test: forced ``tool_choice`` is rejected by *some models*
(the vision ones), while ``response_format=json_schema`` works on all of them.

Run:
    SF_KEY=sk-... python scripts/probe_siliconflow_toolchoice.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

KEY = os.environ.get("SF_KEY", "")
if not KEY:
    sys.exit("SF_KEY is not set")
BASE = os.environ.get("SF_BASE", "https://api.siliconflow.cn/v1")

SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "status": {"type": "string"}},
    "required": ["city", "status"],
}

# The real CortexOutput shape: nested + a stringified JSON object.
CORTEX_SCHEMA = {
    "type": "object",
    "properties": {
        "next_subgoal": {"type": "string"},
        "subgoal_completed": {"type": "boolean"},
        "completion_reason": {"type": "string"},
        "decisions": {"type": "string"},
    },
    "required": ["next_subgoal", "subgoal_completed", "completion_reason", "decisions"],
}


def post(payload: dict) -> tuple[int, str]:
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
                return 200, "tool_calls=" + json.dumps(msg["tool_calls"][0]["function"], ensure_ascii=False)[:160]
            return 200, "content=" + repr((msg.get("content") or "")[:160])
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:200]
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


def tool_payload(model: str, schema: dict, *, force: bool) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "上海现在怎么样？"}],
        "max_tokens": 256,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "CityStatus",
                    "description": "return city status",
                    "parameters": schema,
                },
            }
        ],
        "parallel_tool_calls": False,
    }
    if force:
        payload["tool_choice"] = {"type": "function", "function": {"name": "CityStatus"}}
    else:
        payload["tool_choice"] = "auto"
    return payload


def schema_payload(model: str, schema: dict) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "上海现在怎么样？"}],
        "max_tokens": 256,
        "response_format": {"type": "json_schema", "json_schema": {"name": "CityStatus", "schema": schema}},
    }


MODELS = [
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
]

if __name__ == "__main__":
    for model in MODELS:
        print(f"\n########## {model} ##########")
        print("  tools + FORCED tool_choice :", post(tool_payload(model, SIMPLE_SCHEMA, force=True)))
        print("  tools + auto tool_choice    :", post(tool_payload(model, SIMPLE_SCHEMA, force=False)))
        print("  response_format json_schema :", post(schema_payload(model, SIMPLE_SCHEMA)))
    print("\n########## complex schema (CortexOutput shape) ##########")
    print(
        "  Qwen/Qwen2.5-72B tools+forced:",
        post(tool_payload("Qwen/Qwen2.5-72B-Instruct", CORTEX_SCHEMA, force=True)),
    )
    print(
        "  Qwen/Qwen3-VL-32B json_schema:",
        post(schema_payload("Qwen/Qwen3-VL-32B-Instruct", CORTEX_SCHEMA)),
    )
