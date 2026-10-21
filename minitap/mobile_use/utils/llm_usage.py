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

捕获槽（为什么需要它）
--------------------
结构化输出路径（``with_structured_output``）返回的是 pydantic 对象，原始 AIMessage
落不到调用方手里，于是 ``extract_usage`` 永远读不到 usage —— telemetry 里 cortex /
planner / orchestrator 的 token 恒为 0。这里加一个 ContextVar 捕获槽：调用方开槽，
客户端在拿到原始响应时把消息塞进来，调用方再统一取用。
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

# 本次 LLM 调用的捕获槽。异步任务会复制上下文，但复制的是**引用**，
# 所以槽里的 list 是同一个对象，跨 create_task 追加是可见的。
_capture_sink: ContextVar[list[Any] | None] = ContextVar("llm_usage_capture", default=None)


def open_capture() -> list[Any]:
    """开一个捕获槽并返回可写的 list。返回空 list 表示本次没捕获到原始响应。"""
    sink: list[Any] = []
    _capture_sink.set(sink)
    return sink


def close_capture() -> None:
    """关掉当前捕获槽。必须在 finally 里调用，否则会漏到下一次调用。"""
    _capture_sink.set(None)


def record_raw_response(response: Any) -> None:
    """把 provider 的原始响应塞进当前捕获槽；没有开槽就静默丢弃。

    这个函数会被每条 LLM 调用路径调到，**不能让它自己抛错影响主流程**。
    """
    sink = _capture_sink.get()
    if sink is None:
        return
    try:
        sink.append(response)
    except Exception:  # noqa: BLE001 - 埋点永远不能把任务带崩
        pass


def first_captured(sink: list[Any]) -> Any:
    """取捕获槽里第一条可解析出 usage 的原始响应，取不到返回 None。"""
    for response in sink:
        if extract_usage(response) is not None:
            return response
    return None


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


# 各家 provider 报「命中前缀缓存的 token 数」用的字段名。
# MiniMax / OpenAI 兼容：usage.prompt_tokens_details.cached_tokens
# DeepSeek：usage.prompt_cache_hit_tokens
# Anthropic：usage.cache_read_input_tokens
# langchain 归一化后：usage_metadata['input_token_details']['cache_read']
_CACHE_KEYS = (
    "cached_tokens",
    "prompt_cache_hit_tokens",
    "cache_read_input_tokens",
    "cache_read",
)


def _cache_from_usage_dict(usage: Any) -> int:
    """从一个 usage 字典里挖出缓存命中量；没有就 0。"""
    if not isinstance(usage, dict):
        return 0
    for key in _CACHE_KEYS:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return max(0, value)
    for holder in ("prompt_tokens_details", "input_token_details"):
        details = usage.get(holder)
        if isinstance(details, dict):
            for key in _CACHE_KEYS:
                value = details.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    return max(0, value)
    return 0


def extract_cache_tokens(response: Any) -> int:
    """尽力返回「本次输入中命中前缀缓存的 token 数」；读不到返回 0。

    为什么单列一个函数
    ------------------
    `extract_usage` 只认 input / output 两个数，而**缓存命中量是第三个数**：
    provider 报的是「本次输入里有多少 token 直接读了缓存、没有重算」，它是
    input 的子集。把它混进 input 会算错成本，丢掉它则**遥测里根本没有
    命中率这个指标** —— 而「提示词分层到底有没有用」只能靠它证明。

    2026-10-21 实测（MiniMax M3.1-Flash，OpenAI 兼容端点）：
        usage.prompt_tokens_details.cached_tokens = 10,477
    同一个 body 换个尾部内容再发，立刻掉到 2,329 —— 这个数就是击穿现场。
    """
    if response is None:
        return 0

    generations = getattr(response, "generations", None)
    if generations:
        try:
            message = generations[0][0].message
        except (IndexError, AttributeError, TypeError):
            message = None
        if message is not None and message is not response:
            return extract_cache_tokens(message)

    cached = _cache_from_usage_dict(getattr(response, "usage_metadata", None))
    if cached:
        return cached

    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        token_usage = llm_output.get("token_usage") or llm_output.get("usage")
        cached = _cache_from_usage_dict(token_usage)
        if cached:
            return cached

    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        raw = metadata.get("usage") or metadata.get("token_usage")
        cached = _cache_from_usage_dict(raw)
        if cached:
            return cached

    return 0


@dataclass(frozen=True)
class UsageInfo:
    """一次 LLM 调用的完整用量。

    ``cached_tokens`` 是**第三次元**（2026-10-21 加）：provider 报的是
    「本次输入里有多少 token 直接读了缓存、没有重算」，它是 ``input_tokens``
    的子集。把它并进 input 会算错成本，丢掉它则遥测里没有命中率这个指标。
    """

    input_tokens: int
    output_tokens: int
    cached_tokens: int
    model: str | None = None


def _model_of(response: Any) -> str | None:
    """从响应里取模型名；取不到返回 None。"""
    for holder in ("response_metadata", "llm_output"):
        meta = getattr(response, holder, None)
        if isinstance(meta, dict):
            for key in ("model_name", "model", "modelVersion"):
                value = meta.get(key)
                if isinstance(value, str) and value:
                    return value
    meta = getattr(response, "response_metadata", None)
    if isinstance(meta, dict):
        provider = meta.get("model_provider")
        if isinstance(provider, str) and provider:
            return provider
    return None


def extract_usage_full(response: Any) -> UsageInfo | None:
    """读出 input / output / cached / model；**完全读不到时返回 None**。

    返回 None 而不是「全 0」是刻意的：**「provider 没报」和「零消耗」是两件事**，
    报 0 会让成本看起来像免费。manager 那边也是这个口径（没上报时字段是 null）。

    2026-10-21 真机实测：headless 路径上原来的 `_record_usage` 记的全是 0 ——
    因为它压根没拿到响应，却按「provider 没报」记了 0。把「没拿到」和
    「provider 报了 0」分开之后，缺口就显形了。
    """
    if response is None:
        return None

    generations = getattr(response, "generations", None)
    if generations:
        try:
            message = generations[0][0].message
        except (IndexError, AttributeError, TypeError):
            message = None
        if message is not None and message is not response:
            return extract_usage_full(message)

    model = _model_of(response)
    for holder in ("usage_metadata",):
        usage = getattr(response, holder, None)
        if isinstance(usage, dict) and usage:
            info = _usage_info(usage, model)
            if info is not None:
                return info

    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        raw = llm_output.get("token_usage") or llm_output.get("usage")
        info = _usage_info(raw, model)
        if info is not None:
            return info

    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        raw = metadata.get("usage") or metadata.get("token_usage")
        info = _usage_info(raw, model)
        if info is not None:
            return info

    return None


def _usage_info(raw: Any, model: str | None) -> UsageInfo | None:
    """从一个 usage 字典里读出四元组；没有可用数字就返回 None。"""
    if not isinstance(raw, dict) or not raw:
        return None
    input_tokens = _int(raw.get("input_tokens", raw.get("prompt_tokens")))
    output_tokens = _int(raw.get("output_tokens", raw.get("completion_tokens")))
    cached = _cache_from_usage_dict(raw)
    if not (input_tokens or output_tokens or cached):
        return None
    return UsageInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached,
        model=model,
    )
