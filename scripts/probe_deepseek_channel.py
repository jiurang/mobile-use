"""复现「manager 渠道选 deepseek 时请求一直失败」。

为什么必须走渠道路径
--------------------
直接调 ``get_deepseek_llm()`` 走的是 .env 那条老路（settings 里读
``DEEPSEEK_THINKING``）。而 manager 下发的渠道是另一条路：
``allow_env_fallback=False`` → ``ProviderOverride.strict=True`` → 密钥/端点只认渠道
给的。两条路的 thinking 处理**不一样**，所以老路的探针证明不了渠道这条路。

要回答三个问题：

1. 渠道里把 ``thinking`` 设成 false，请求体里**真的**带上了 ``thinking.disabled`` 吗？
2. executor / reporter 用的 ``get_structured_llm`` 会不会**强制** tool_choice？
   （DeepSeek 思考模式下强制指定 tool_choice 是硬 400。）
3. 真打一次 API，是 400 还是 200？

跑法：
    .\\.venv\\Scripts\\python.exe scripts/probe_deepseek_channel.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))

from pydantic import BaseModel, Field  # noqa: E402


class _ProbeOutput(BaseModel):
    """一个最小的结构化输出 schema，形状与 planner 用的那个一样。"""

    subgoals: list[str] = Field(description="要做的子目标")


def _channel_override(thinking: bool | None):
    """复刻 runner-host 的 ``_to_llm_spec`` → ``ProviderOverride`` 那条链。"""
    from minitap.mobile_use.config import LLM
    from minitap.mobile_use.services.llm import ProviderOverride

    llm = LLM(
        provider="deepseek",
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        thinking=thinking,
        # 渠道路径：不再从设备本地 .env 补齐。
        allow_env_fallback=False,
    )
    return ProviderOverride(
        api_key=llm.api_key,
        base_url=llm.base_url,
        thinking=llm.thinking,
        strict=not llm.allow_env_fallback,
    )


async def _main() -> int:
    from minitap.mobile_use.services.llm import (
        _ToolCallStructuredRunnable,
        get_deepseek_llm,
    )

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("DEEPSEEK_API_KEY 没配，跳过实测。")
        return 1

    print("=" * 78)
    print("问题 1：渠道里把 thinking 设成 false，请求体里带上了吗？")
    print("=" * 78)
    for thinking in (False, True, None):
        override = _channel_override(thinking)
        client = get_deepseek_llm(model_name="deepseek-chat", temperature=1, override=override)
        extra = client.extra_body
        print(
            f"  渠道 thinking={thinking!r:<6} → extra_body={extra!r}"
            f"   {'✅ 下发了' if extra else '❌ 没下发（渠道的开关被忽略）'}"
        )

    print()
    print("=" * 78)
    print("问题 2：executor / reporter 走的那条路，会不会强制 tool_choice？")
    print("=" * 78)
    override = _channel_override(False)
    client = get_deepseek_llm(model_name="deepseek-chat", temperature=1, override=override)
    # get_structured_llm 的默认分支：恒 force_tool_choice=True。
    forced = _ToolCallStructuredRunnable(client, _ProbeOutput)
    has_choice = "tool_choice" in forced._bound.kwargs
    print(f"  get_structured_llm 那条路：tool_choice 被强制 = {has_choice}")
    print(
        "  ❌ DeepSeek 思考模式下强制 tool_choice = 400 "
        "「Thinking mode does not support this tool_choice」"
        if has_choice
        else "  ✅ 没有强制"
    )

    print()
    print("=" * 78)
    print("问题 3：真打一次 DeepSeek API")
    print("=" * 78)
    import httpx
    from langchain_core.utils.function_calling import convert_to_openai_tool

    tools = [convert_to_openai_tool(_ProbeOutput)]
    url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com") + "/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    base_payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "把「买牛奶」拆成子目标。"}],
        "tools": tools,
    }

    async with httpx.AsyncClient(timeout=120) as client_http:
        for label, extra in (
            ("现状：强制 tool_choice + 不带 thinking", {}),
            (
                "现状 + 强制 tool_choice + thinking.disabled",
                {"thinking": {"type": "disabled"}},
            ),
            (
                "只带 thinking.disabled，tool_choice 不强制",
                {"thinking": {"type": "disabled"}},
            ),
        ):
            payload = dict(base_payload)
            if "tool_choice" in label or "强制" in label:
                payload["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tools[0]["function"]["name"]},
                }
            payload.update(extra)
            r = await client_http.post(url, json=payload, headers=headers)
            try:
                body = r.json()
            except Exception:  # noqa: BLE001
                body = {"raw": r.text[:300]}
            if r.status_code != 200:
                msg = body.get("error", body)
                print(f"  ❌ HTTP {r.status_code}  {label}")
                print(f"       {str(msg)[:220]}")
            else:
                msg = (body.get("choices") or [{}])[0].get("message") or {}
                print(
                    f"  ✅ HTTP 200  {label}"
                    f"  tool_calls={len(msg.get('tool_calls') or [])}"
                    f"  content={str(msg.get('content'))[:60]!r}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
