"""从 LLM 响应里尽力提取 token 用量。

为什么不放在 services/ 或直接写在 telemetry 里
---------------------------------------------
TUI 的 `tui/events.py` 里本来就有一份成熟实现，覆盖 langchain 各版本、各 provider
的多种 usage 形状，并且有单测锁定。这里把它提为共用函数，让 TUI 与 telemetry
引用**同一份**解析逻辑 —— 两处各写一份必然漂移，而 usage 形状恰恰是最容易变的部分。

支持的三类形状（按顺序尝试）
---------------------------
1. ``usage_metadata``  —— langchain-core >= 0.3 的标准字段
2. ``llm_output["token_usage"]`` —— OpenAI 风格
3. ``response_metadata["usage"]`` —— Gemini 等 provider 的多种拼写

另外还要能拆开 ``LLMResult``：``on_llm_end`` 拿到的是 LLMResult 而不是 AIMessage，
真正的字段在 ``result.generations[0][0].message`` 上，不拆就永远读不到。
"""

from __future__ import annotations

from typing import Any


def _int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def extract_usage(response: Any) -> tuple[int, int] | None:
    """尽力返回 (input_tokens, output_tokens)；完全读不到时返回 None。

    返回 None 而不是 (0, 0) 是刻意的：OpenAI 兼容端点完全可以不返回 usage，
    那种情况下报 (0,0) 会让统计看起来像「这次调用免费」，比不显示更糟。
    """
    if response is None:
        return None

    # 拆开 LLMResult -> ChatGeneration -> AIMessage
    generations = getattr(response, "generations", None)
    if generations:
        try:
            message = generations[0][0].message
        except (IndexError, AttributeError, TypeError):
            message = None
        if message is not None and message is not response:
            return extract_usage(message)

    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict) and usage:
        return _int(usage.get("input_tokens")), _int(usage.get("output_tokens"))

    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        token_usage = llm_output.get("token_usage") or llm_output.get("usage")
        if isinstance(token_usage, dict) and token_usage:
            return _int(token_usage.get("prompt_tokens", token_usage.get("input_tokens"))), _int(
                token_usage.get("completion_tokens", token_usage.get("output_tokens"))
            )

    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        raw = metadata.get("usage") or metadata.get("token_usage")
        if isinstance(raw, dict) and raw:
            return _int(
                raw.get("input_tokens", raw.get("prompt_tokens", raw.get("promptTokenCount")))
            ), _int(
                raw.get(
                    "output_tokens",
                    raw.get("completion_tokens", raw.get("candidatesTokenCount")),
                )
            )

    return None
