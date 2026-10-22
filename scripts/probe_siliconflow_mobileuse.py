"""Test SiliconFlow through mobile-use's real code path.

This is the decisive test for "no code change": the manager's channel
(provider=openai + baseUrl + apiKey) resolves to exactly this factory call in
minitap/mobile_use/services/llm.py:get_llm() -> get_openai_llm().

Key comes from the SF_KEY env var (never written into the file).

Run:
    SF_KEY=sk-... python scripts/probe_siliconflow_mobileuse.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

KEY = os.environ.get("SF_KEY", "")
if not KEY:
    sys.exit("SF_KEY is not set")

SF_BASE = os.environ.get("SF_BASE", "https://api.siliconflow.cn/v1")
TEXT_MODEL = os.environ.get("SF_TEXT_MODEL", "Qwen/Qwen2.5-72B-Instruct")
VL_MODEL = os.environ.get("SF_VL_MODEL", "Qwen/Qwen3-VL-32B-Instruct")


def _banner(tag: str) -> None:
    print(f"\n===== {tag} =====", flush=True)


def _png_data_url() -> str:
    """A real, valid PNG (black square on white) — a truncated base64 blob is
    rejected by the server as a broken file, which looks like a model failure
    but is not."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (128, 96), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([24, 20, 88, 76], fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


async def main() -> None:
    from minitap.mobile_use.agents.cortex.types import CortexOutput
    from minitap.mobile_use.services.llm import ProviderOverride, get_openai_llm

    override = ProviderOverride(api_key=KEY, base_url=SF_BASE)

    def client_for(model: str, temperature: float = 1):
        return get_openai_llm(model, temperature, override)

    # 1. plain invoke — the executor's non-structured path
    _banner(f"1. plain invoke | {TEXT_MODEL}")
    llm = client_for(TEXT_MODEL)
    msg = await llm.ainvoke("用一句话说明什么是光合作用。")
    print("content:", repr(str(msg.content)[:200]))
    print("usage_metadata:", msg.usage_metadata)

    # 2. with_structured_output on a REAL mobile-use schema
    #    (this is what planner / orchestrator / contextor / cortex / hopper /
    #     outputter all do; langchain-openai 1.1.10 defaults to json_schema)
    _banner(f"2. with_structured_output(CortexOutput) | {TEXT_MODEL}")
    structured = client_for(TEXT_MODEL).with_structured_output(CortexOutput)
    result = await structured.ainvoke(
        "屏幕上是微信首页，用户想给妈妈发一条消息「我到家了」。"
        "给出下一步决策。"
    )
    print("type:", type(result).__name__)
    print("parsed:", json.dumps(result.model_dump(), ensure_ascii=False)[:600])

    # 3. tool calling — the executor path (16 tools bound in production)
    _banner(f"3. bind_tools | {TEXT_MODEL}")
    from langchain_core.tools import tool

    @tool
    def tap(x: int, y: int) -> str:
        """Tap the screen at x,y."""
        return "tapped"

    bound = client_for(TEXT_MODEL).bind_tools([tap])
    msg = await bound.ainvoke("点一下屏幕坐标 100,200")
    print("tool_calls:", json.dumps(msg.tool_calls, ensure_ascii=False)[:400])
    print("usage_metadata:", msg.usage_metadata)

    # 4. vision — executor's screenshot path
    _banner(f"4. vision | {VL_MODEL}")
    vl = client_for(VL_MODEL)
    msg = await vl.ainvoke(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "图里是什么形状？只回一个词。"},
                    {"type": "image_url", "image_url": {"url": _png_data_url()}},
                ],
            }
        ]
    )
    print("content:", repr(str(msg.content)[:200]))
    print("usage_metadata:", msg.usage_metadata)

    # 5. vision + json_schema on the VL model (the risky combination)
    _banner(f"5. vision + with_structured_output | {VL_MODEL}")
    vl_structured = client_for(VL_MODEL).with_structured_output(CortexOutput)
    result = await vl_structured.ainvoke(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "图里是什么形状？用户想点它，给出决策。"},
                    {"type": "image_url", "image_url": {"url": _png_data_url()}},
                ],
            }
        ]
    )
    print("parsed:", json.dumps(result.model_dump(), ensure_ascii=False)[:400])

    # 6. streaming + usage — is usage visible without stream_usage=True?
    _banner(f"6. stream + usage_metadata | {TEXT_MODEL}")
    chunks = []
    async for chunk in client_for(TEXT_MODEL).astream("数到三"):
        chunks.append(chunk)
    last = chunks[-1]
    print("chunks:", len(chunks))
    print("usage_metadata:", getattr(last, "usage_metadata", None))


if __name__ == "__main__":
    asyncio.run(main())
