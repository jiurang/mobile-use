import asyncio
import functools
import inspect
import logging
import re
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr, ValidationError

from minitap.mobile_use.config import (
    AgentNode,
    AgentNodeWithFallback,
    LLMUtilsNode,
    LLMUtilsNodeWithFallback,
    LLMWithFallback,
    settings,
)
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.services.telemetry import telemetry
from minitap.mobile_use.utils.llm_usage import (
    close_capture,
    extract_usage,
    first_captured,
    open_capture,
    record_raw_response,
)
from minitap.mobile_use.utils.logger import get_logger

if TYPE_CHECKING:
    # Imported for typing only. The Google SDKs are imported inside the
    # factories below: their dependency tree (google-aiplatform -> numexpr,
    # bottleneck, shapely) is native-heavy and unavailable on some platforms we
    # support, so only the Gemini/Vertex providers pay for it.
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_google_vertexai import ChatVertexAI

    # Same reasoning, same trap: `langchain_azure_ai` is a heavy provider tree
    # that only the Azure provider needs, but importing it at module scope makes
    # every other provider — and every host that merely wants to read
    # MiniMaxChatOpenAI — fail when it is absent. An on-device APK is exactly
    # that case. NOTE this file has no `from __future__ import annotations`, so
    # the return annotation below must stay quoted; a bare name here evaluates at
    # `def` time and raises NameError on import, taking the whole SDK with it.
    from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel

# Logger for internal messages (ex: fallback)
llm_logger = logging.getLogger(__name__)
# Logger for user messages
user_messages_logger = get_logger(__name__)

# Upper bound on a single model request. See get_minimax_llm for the failure this
# exists to prevent; measured latency on a real phone is seconds, so this is
# deliberately far above normal and only bites on a hung connection.
LLM_REQUEST_TIMEOUT_SECONDS = 180.0


@dataclass(frozen=True)
class ProviderOverride:
    """运行期下发的接入参数。字段为 ``None`` 表示「没下发,走 settings」。

    这是 manager 侧「渠道」下发过来的 key / 端点在这里的落点。每个 provider 工厂都按
    「override 优先、settings 兜底」解析,于是没下发渠道的部署行为与之前逐字节一致 ——
    这是这条路径能被安全打开的前提。
    """

    api_key: SecretStr | None = None
    base_url: str | None = None

    def key_or(self, attr: str) -> SecretStr | None:
        return self.api_key or getattr(settings, attr, None)

    def url_or(self, attr: str, default: str | None = None) -> str | None:
        return self.base_url or getattr(settings, attr, None) or default


def _warn_base_url_ignored(provider: str) -> None:
    """这个 provider 的客户端没有端点参数,渠道下发的端点用不上。

    明确报出来而不是静默丢弃 —— 「配置改了没反应」是这类问题里最难查的一种。
    """
    llm_logger.warning(
        f"渠道下发了 base_url,但 provider={provider} 的客户端不支持自定义端点,已忽略"
    )


async def invoke_llm_with_timeout_message[T](
    llm_call: Coroutine[Any, Any, T],
    timeout_seconds: int = 10,
    agent_name: str | None = None,
    task_id: str | None = None,
    used_fallback: bool = False,
) -> T:
    """
    Send a LLM call and display a timeout message if it takes too long.

    这是**所有** agent 调 LLM 的唯一汇聚点，所以 token 埋点放在这里 ——
    一处覆盖 cortex / executor / planner / orchestrator / contextor / outputter /
    hopper / video_analyzer，不用逐个 agent 改。

    Args:
        llm_call: The coroutine of the LLM call to execute.
        timeout_seconds: The delay in seconds before displaying the message.
        agent_name: 调用方 agent 名，用于成本归因。
        task_id: 任务 trace id。
        used_fallback: 本次是否走 fallback 模型（主模型失败时为 True）。

    Returns:
        The result of the LLM call.
    """
    started_at = time.perf_counter()
    # 捕获槽必须在 create_task **之前**开：任务复制的是创建那一刻的上下文，
    # 开晚了子任务里就看不到槽，原始响应也就塞不进来。
    sink = open_capture()
    try:
        llm_task = asyncio.create_task(llm_call)
        waiter_task = asyncio.create_task(asyncio.sleep(timeout_seconds))

        done, _ = await asyncio.wait(
            {llm_task, waiter_task}, return_when=asyncio.FIRST_COMPLETED
        )

        if llm_task in done:
            # The LLM call has finished before the timeout, cancel the timer
            waiter_task.cancel()
            result = llm_task.result()
        else:
            # The timeout has been reached, display the message and wait for the call to finish
            user_messages_logger.info("Waiting for LLM call response...")
            result = await llm_task
    finally:
        close_capture()

    _record_usage(
        result,
        raw_response=first_captured(sink),
        agent_name=agent_name,
        task_id=task_id,
        duration_seconds=time.perf_counter() - started_at,
        used_fallback=used_fallback,
    )
    return result


def _record_usage(
    result: Any,
    *,
    raw_response: Any,
    agent_name: str | None,
    task_id: str | None,
    duration_seconds: float,
    used_fallback: bool,
) -> None:
    """记录 token 用量。任何异常都不允许影响主流程。

    usage 必须从**原始响应**上取。cortex / planner / orchestrator 走的是
    `with_structured_output`，返回 pydantic 对象，usage 落不到 result 上；
    原始 AIMessage 由客户端侧（generate/agenerate 包装）塞进捕获槽，这里优先用它。
    两个都读不到才记 0 —— 那说明 provider 根本没回报 usage，不是我们丢的。
    """
    if not agent_name:
        return
    try:
        usage = extract_usage(raw_response) or extract_usage(result)
        input_tokens, output_tokens = usage if usage else (0, 0)
        telemetry.capture_llm_usage(
            task_id=task_id or "unknown",
            agent=agent_name,
            model=_extract_model_name(raw_response) or _extract_model_name(result),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_seconds=duration_seconds,
            used_fallback=used_fallback,
        )
    except Exception as e:  # noqa: BLE001
        # 埋点永远不能把任务带崩
        llm_logger.debug(f"Failed to record LLM usage: {e}")


def _extract_model_name(result: Any) -> str | None:
    """从响应里取模型名。provider 各有各的字段名，逐个试，都拿不到就返回 None。"""
    for attr in ("model", "response_metadata", "llm_output"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            for key in ("model_name", "model", "modelVersion"):
                if value.get(key):
                    return str(value[key])
    return None


def _capture_from_llm_result(result: Any) -> None:
    """把 LLMResult 里的第一条 AIMessage 交给捕获槽。读不到就静默跳过。"""
    try:
        record_raw_response(result.generations[0][0].message)
    except (AttributeError, IndexError, TypeError):
        pass


def _wrap_generate(original: Any) -> Any:
    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = await original(self, *args, **kwargs)
            _capture_from_llm_result(result)
            return result

    else:

        @functools.wraps(original)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
            result = original(self, *args, **kwargs)
            _capture_from_llm_result(result)
            return result

    wrapper._usage_capture_wrapped = True  # type: ignore[attr-defined]
    return wrapper


_usage_capture_installed = False


def _install_usage_capture() -> None:
    """给客户端类的 generate / agenerate 包一层，把原始响应塞进捕获槽。

    为什么非要在**类上**打，而不是给实例挂回调：结构化输出路径返回 pydantic
    对象，原始 AIMessage 落不到调用方手里；而 langchain callback 在这条路径上
    实测不触发（见 doc/prompt-i18n-ab.md 四·2）。同步 invoke 走 generate、
    异步 ainvoke 走 agenerate，**只补一个会漏**（同一份文档里踩过）。
    MiniMax / DeepSeek 的客户端是 ChatOpenAI 的子类，包基类即可覆盖。
    """
    global _usage_capture_installed
    if _usage_capture_installed:
        return
    for cls in (ChatOpenAI, ChatAnthropic):
        for name in ("generate", "agenerate"):
            original = getattr(cls, name, None)
            if original is None or getattr(original, "_usage_capture_wrapped", False):
                continue
            setattr(cls, name, _wrap_generate(original))
    _usage_capture_installed = True


def _build_custom_headers(agent_name: str | None = None) -> dict[str, str]:
    """Build custom headers for LLM API calls."""
    headers: dict[str, str] = {}
    if agent_name:
        headers["X-Agent-Name"] = agent_name
    if settings.PROJECT_NAME:
        headers["X-Project-Name"] = settings.PROJECT_NAME
    return headers


def get_minitap_llm(
    trace_id: str,
    model: str = "google/gemini-2.5-pro",
    temperature: float | None = None,
    max_retries: int | None = None,
    api_key: str | None = None,
    agent_name: str | None = None,
    override: ProviderOverride | None = None,
) -> ChatOpenAI:
    override = override or ProviderOverride()
    if api_key:
        effective_api_key = SecretStr(api_key)
    elif override.api_key:
        effective_api_key = override.api_key
    elif settings.MINITAP_API_KEY:
        effective_api_key = settings.MINITAP_API_KEY
    else:
        raise ValueError("MINITAP_API_KEY must be provided or set in environment")

    if max_retries is None and model.startswith("google/"):
        max_retries = 2

    custom_headers = _build_custom_headers(agent_name)

    client = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        api_key=effective_api_key,
        base_url=override.url_or("MINITAP_API_BASE_URL"),
        default_query={
            "sessionId": trace_id,
        },
        default_headers=custom_headers or None,
    )
    return client


def get_google_llm(
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.7,
    override: ProviderOverride | None = None,
) -> "ChatGoogleGenerativeAI":  # quoted: import is TYPE_CHECKING-only, see line 29
    override = override or ProviderOverride()
    if override.base_url:
        _warn_base_url_ignored("google")
    api_key = override.key_or("GOOGLE_API_KEY")
    assert api_key is not None
    from langchain_google_genai import ChatGoogleGenerativeAI

    client = ChatGoogleGenerativeAI(
        model=model_name,
        max_tokens=None,
        temperature=temperature,
        api_key=api_key,
        max_retries=2,
    )
    return client


def get_vertex_llm(
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.7,
    override: ProviderOverride | None = None,
) -> "ChatVertexAI":  # quoted: import is TYPE_CHECKING-only, see line 29
    # Vertex 只认 ADC + project,没有端点和密钥两个旋钮。渠道下发了也用不上,
    # 但要报出来 —— 静默丢弃会变成「配置改了没反应」。
    override = override or ProviderOverride()
    if override.api_key:
        _warn_base_url_ignored("vertexai (api_key)")
    if override.base_url:
        _warn_base_url_ignored("vertexai (base_url)")
    from langchain_google_vertexai import ChatVertexAI

    client = ChatVertexAI(
        model_name=model_name,
        max_tokens=None,
        temperature=temperature,
        max_retries=2,
    )
    return client


class AnthropicCacheControlChat(ChatAnthropic):
    """给 Anthropic 打**显式** cache_control 断点。

    Anthropic 与 OpenAI 兼容端点的缓存机制不同：前者必须打断点才缓存，
    后者自动做前缀缓存。断点加在 system 块上 —— 那是整段静态、且超过
    Anthropic 1024 token 最小可缓存长度的稳定前缀（cortex.md 渲染后约 1.5k token）。

    更细的断点（例如把计划/进度块也标上）需要调用方知道自己消息分层的边界，
    这里不做 —— 猜测出来的断点一旦落在会变的那一块上，等于没有。
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs) -> dict:  # type: ignore[override]
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        system = payload.get("system")
        if isinstance(system, str) and system:
            payload["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        elif isinstance(system, list) and system and isinstance(system[-1], dict):
            system[-1]["cache_control"] = {"type": "ephemeral"}
        return payload


def get_anthropic_llm(
    model_name: str = "claude-sonnet-4-6",
    temperature: float = 1,
    override: ProviderOverride | None = None,
) -> ChatAnthropic:
    override = override or ProviderOverride()
    api_key = override.key_or("ANTHROPIC_API_KEY")
    assert api_key is not None
    client = AnthropicCacheControlChat(
        model_name=model_name,
        api_key=api_key,
        # `anthropic_api_url` 是 ChatAnthropic 的端点字段名(不是 `base_url`,
        # 见 langchain_anthropic/chat_models.py:946)。这里只认渠道下发的值 ——
        # Settings 里没有 ANTHROPIC_BASE_URL 这个键,不要凭空发明一个。
        anthropic_api_url=override.base_url,
        temperature=temperature,
        max_tokens=8192,  # type: ignore[call-arg]  # field aliased to max_tokens_to_sample
        max_retries=2,
        # Was None, i.e. "never give up" — see get_minimax_llm for why an
        # unbounded request timeout is never the safe default. The field is
        # `default_request_timeout` on ChatAnthropic (alias "timeout"); passing
        # `request_timeout` here is silently ignored, which is how this stayed
        # broken the first time.
        default_request_timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        stop=None,
    )
    return client


def get_openai_llm(
    model_name: str = "o3",
    temperature: float = 1,
    override: ProviderOverride | None = None,
) -> ChatOpenAI:
    override = override or ProviderOverride()
    api_key = override.key_or("OPENAI_API_KEY")
    assert api_key is not None
    client = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=override.url_or("OPENAI_BASE_URL"),
        temperature=temperature,
    )
    return client


def get_openrouter_llm(
    model_name: str,
    temperature: float = 1,
    override: ProviderOverride | None = None,
):
    override = override or ProviderOverride()
    api_key = override.key_or("OPEN_ROUTER_API_KEY")
    assert api_key is not None
    client = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=override.url_or("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    return client


def get_grok_llm(
    model_name: str,
    temperature: float = 1,
    override: ProviderOverride | None = None,
) -> ChatOpenAI:
    override = override or ProviderOverride()
    api_key = override.key_or("XAI_API_KEY")
    assert api_key is not None
    client = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        base_url=override.url_or("XAI_BASE_URL", "https://api.x.ai/v1"),
    )
    return client


def get_azure_llm(
    model_name: str,
    temperature: float = 1,
    override: ProviderOverride | None = None,
) -> "AzureAIOpenAIApiChatModel":
    # Local import on purpose: only the Azure provider should pay for this tree.
    from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel

    override = override or ProviderOverride()
    endpoint = override.url_or("AZURE_BASE_URL")
    assert endpoint is not None

    api_key: str | None = None
    key = override.key_or("AZURE_API_KEY")
    if key:
        api_key = key.get_secret_value()

    if api_key:
        client = AzureAIOpenAIApiChatModel(
            model=model_name,
            endpoint=endpoint,
            credential=api_key,
            temperature=temperature,
        )
    else:
        from azure.identity import DefaultAzureCredential

        client = AzureAIOpenAIApiChatModel(
            model=model_name,
            project_endpoint=endpoint,
            credential=DefaultAzureCredential(),
            temperature=temperature,
        )
    return client


class _ToolCallStructuredRunnable(Runnable):
    """Bind a Pydantic schema as a tool and parse the AIMessage ourselves.

    Used by providers whose OpenAI-compatible endpoint cannot serve
    langchain-openai's default structured-output path.

    Why this exists (both cases verified against the live API, not inferred):

    * ``ChatOpenAI.with_structured_output`` in langchain-openai 1.1.x defaults to
      ``method="json_schema"`` — note this is the **ChatOpenAI** override; the
      ``BaseChatOpenAI`` default is still ``"function_calling"``. It binds the
      schema as ``response_format=<PydanticClass>``, which the OpenAI SDK turns
      into ``{"type": "json_schema", ...}``. Hooking ``_get_request_payload``
      shows the payload has ``response_format`` and *no* ``tools``.
    * MiniMax ignores ``response_format`` and replies with ``<think>...</think>``
      (thinking on) or ```` ```json ```` (thinking off) — both unparseable by the
      OpenAI SDK, so ``result.parsed`` comes back ``None``.
    * DeepSeek rejects it outright: ``400 invalid_request_error: This
      response_format type is unavailable now``.

    So we bind the schema as a tool ourselves and parse the response:

    1. If ``tool_calls`` are present, validate the first one's ``args``.
    2. Otherwise strip ```json fences from ``content`` and validate (covers
       providers that answer in markdown JSON instead of calling the tool).

    ``force_tool_choice`` exists because DeepSeek answers
    ``400 invalid_request_error: Thinking mode does not support this
    tool_choice`` when a specific function is forced while thinking mode is on
    (its default). Passing ``False`` binds the tool without forcing it; the
    model then calls the tool on its own, verified live.

    Other call sites that use ``with_structured_output`` (orchestrator, cortex,
    executor, contextor, hopper, outputter) all use Pydantic schemas, so this
    single code path covers everything.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        schema: type[BaseModel],
        force_tool_choice: bool = True,
    ):
        from langchain_core.utils.function_calling import convert_to_openai_tool

        self.llm = llm
        self.schema = schema
        # Pre-bind the schema as an OpenAI tool. Without a tool definition in the
        # request, MiniMax-M3 (thinking on) answers by emitting ```json...```
        # inside its <think>...</think> content block instead of using
        # tool_calls. Pre-binding makes M3 emit proper tool_calls, which we then
        # parse in ``_parse``. We disable parallel tool calls to keep the choice
        # deterministic.
        bind_kwargs: dict[str, Any] = {"parallel_tool_calls": False}
        if force_tool_choice:
            tool_name = convert_to_openai_tool(schema)["function"]["name"]
            bind_kwargs["tool_choice"] = tool_name
        self._bound = llm.bind_tools([schema], **bind_kwargs)

    def invoke(self, input, config=None, **kwargs):
        msg = self._bound.invoke(input, config=config, **kwargs)
        return self._parse(msg)

    async def ainvoke(self, input, config=None, **kwargs):
        msg = await self._bound.ainvoke(input, config=config, **kwargs)
        return self._parse(msg)

    @staticmethod
    def _strip_markdown_json(text: str) -> str:
        text = text.strip()
        # Drop MiniMax-M3 <think>...</think> reasoning block first. With
        # thinking on, M3 wraps the JSON inside the thinking block AND mimics
        # the ```json fence it saw in the planner.md prompt, so the response
        # looks like:
        #     <think>...reasoning...\n```json\n{...}\n```</think>
        # Strip thinking before stripping fences, otherwise the leading <think>
        # survives and the JSON parser fails on "<" at column 1.
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        # ```json\n{...}\n```  or  ```\n{...}\n```
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        return text.strip()

    @staticmethod
    def _extract_json_object(text: str) -> str | None:
        """Return the first balanced ``{...}`` block in ``text``, else ``None``.

        Deliberately a character scan rather than a regex: ``CortexOutput.decisions``
        is itself a *stringified* JSON object, so the text contains nested braces and
        escaped quotes that a ``\\{.*\\}`` pattern would truncate on.
        """
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def _parse(self, msg: AIMessage) -> BaseModel:
        # 1) tool_calls 路径 (M3 thinking 开 / M2.7 等)
        # Note: LangChain normalises OpenAI tool_calls into ``{"name", "args",
        # "id", "type"}`` dicts where ``args`` is already parsed into a Python
        # object, NOT a JSON string. So we use ``model_validate`` (dict) or
        # ``model_validate_json`` (str) depending on the type we actually get.
        for tc in msg.tool_calls or []:
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
            if args is None:
                continue
            try:
                if isinstance(args, str):
                    return self.schema.model_validate_json(args)
                return self.schema.model_validate(args)
            except ValidationError:
                continue

        # 2) content 路径 (M3 thinking 关 / Markdown JSON 兜底)
        content = msg.content if isinstance(msg.content, str) else ""
        if not content.strip():
            raise ValueError(
                f"MiniMax response had no tool_calls and no content. "
                f"finish_reason={getattr(msg, 'response_metadata', {}).get('finish_reason')!r}"
            )

        stripped = self._strip_markdown_json(content)
        # 2a) 整段就是 JSON
        # 2b) JSON 被包在一堆解释性文字里（"好的，我来…" / "[orchestrator] …"）——
        #     整段当 JSON 解析必然失败，退回提取第一个配平的 {..} 块。
        for candidate in (stripped, self._extract_json_object(stripped)):
            if not candidate:
                continue
            try:
                return self.schema.model_validate_json(candidate)
            except ValidationError:
                continue

        # 两条路都失败。这里必须自己抛：pydantic 的 json_invalid 会把 input_value
        # 截断成 '...'，现场根本看不出模型到底说了什么，而 with_fallback 正是靠这条
        # message 上报给用户的。带上原文才算可诊断。
        raise ValueError(
            f"MiniMax returned no parseable {self.schema.__name__} "
            f"(tool_calls={len(msg.tool_calls or [])}, "
            f"finish_reason={getattr(msg, 'response_metadata', {}).get('finish_reason')!r}). "
            f"content[:1000]={content[:1000]!r}"
        )


class MiniMaxChatOpenAI(ChatOpenAI):
    """ChatOpenAI variant for MiniMax that bypasses LangChain's broken
    ``with_structured_output`` integration.

    See ``_ToolCallStructuredRunnable`` for the full rationale.
    """

    def with_structured_output(self, schema=None, **kwargs):  # type: ignore[override]
        # langchain-openai's signature accepts dict / TypedDict / Pydantic for
        # ``schema``. mobile-use only ever passes Pydantic classes, but keep
        # the door open: if a non-Pydantic schema slips in, fall back to the
        # upstream implementation so we don't silently misbehave elsewhere.
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            kwargs.setdefault("method", "function_calling")
            return super().with_structured_output(schema, **kwargs)
        return _ToolCallStructuredRunnable(self, schema)


MINIMAX_EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")


def resolve_minimax_effort() -> str | None:
    """把 ``MINIMAX_EFFORT`` 归一化成官方文档里的合法档位，或 ``None``（不发）。

    归一化而不是裸传，是因为这个值直接进请求体：大小写或空格写错，服务端要么
    400，要么静默回落到默认（默认是 ``max``，最贵那档）—— 实测里最难查的那类
    「配置改了没反应」。

    返回 ``None`` 表示**不发** ``output_config``，走服务端默认。
    """
    raw = settings.MINIMAX_EFFORT
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if not value:
        return None
    if value in MINIMAX_EFFORT_LEVELS:
        return value
    llm_logger.warning(
        f"未知的 MINIMAX_EFFORT={raw!r}，按「不发送」处理"
        f"（即服务端默认 max）。可选：{', '.join(MINIMAX_EFFORT_LEVELS)}"
    )
    return None


def get_minimax_llm(
    model_name: str = "MiniMax-M2.7",
    temperature: float = 1,
    override: ProviderOverride | None = None,
) -> ChatOpenAI:
    """构造 MiniMax 客户端。

    下发 ``output_config.effort``（默认 ``max``）。M3.1-Flash-Preview **关不掉**思考
    （官方：always thinks），只能调深度；这个值决定思考花多少 token 和多少时间。

    这里**不下发** ``thinking`` 参数。M3.1 官方明确 "If type is sent it must be
    adaptive; disabled returns HTTP 400"，M2.x 则是 "disabled is accepted but
    ignored" —— 两边都是死开关，发它只会制造 400 风险，没有任何收益。

    为什么非要换到 M3.1：真机同批 prompt n=16 实测，M3 会把思考内容写进
    ``content`` 且经常不闭合，``_ToolCallStructuredRunnable._parse`` 靠 ``</think>``
    剥不掉 -> 没有 tool_calls -> "returned no parseable CortexOutput" -> 整个
    goal 崩。M3.1 的思考走独立字段，16 次 ``content`` 里 0 次混入 <think>。
    """
    override = override or ProviderOverride()
    api_key = override.key_or("MINIMAX_API_KEY")
    assert api_key is not None
    clamped_temperature = max(0.0, min(1.0, temperature))

    extra_body: dict[str, Any] = {}
    effort = resolve_minimax_effort()
    if effort is not None:
        extra_body["output_config"] = {"effort": effort}

    client = MiniMaxChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=clamped_temperature,
        base_url=override.url_or("MINIMAX_BASE_URL", "https://api.minimax.io/v1"),
        # A bounded request timeout, because the default is not one. With
        # `request_timeout` unset the OpenAI SDK waits 600s and langchain retries
        # twice, so a single hung connection parks the whole agent for 30
        # minutes with only "Waiting for LLM call response..." printed — there
        # is no error, no retry limit, no way back. Reproduced on a phone over
        # USB: the LLM call at 22:23:59 had still produced nothing at 22:36.
        # Measured normal latency is seconds, so 180s is generous; one retry
        # covers a single dropped connection without compounding a real outage.
        request_timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        max_retries=1,
        # 显式传而不是 **dict 解包：解包会让类型检查器把整个 dict 当成
        # **kwargs 逐个参数校验，一堆误报。空的时候给 None（不发这个字段）。
        extra_body=extra_body or None,
    )
    return client


class DeepSeekChatOpenAI(ChatOpenAI):
    """ChatOpenAI variant for DeepSeek's OpenAI-compatible endpoint.

    **Do not simplify this back to plain ``ChatOpenAI`` without a live call.**
    Two DeepSeek-specific constraints, both reproduced against the real API:

    1. langchain-openai 1.1.x's ``ChatOpenAI.with_structured_output`` defaults to
       ``method="json_schema"`` (the ``BaseChatOpenAI`` default is still
       ``"function_calling"``), sending ``response_format=json_schema``. DeepSeek
       answers ``400 invalid_request_error: This response_format type is
       unavailable now`` — it only implements ``response_format=json_object``.
       Every agent node 400s before emitting a token.
    2. Binding the schema as a tool *with a forced specific ``tool_choice``* also
       fails while thinking mode is on (DeepSeek's default):
       ``400 invalid_request_error: Thinking mode does not support this
       tool_choice``. So we force it only when the user has explicitly turned
       thinking off via ``DEEPSEEK_THINKING=false``; otherwise the tool is
       offered and the model calls it on its own (verified: it does).
    """

    def with_structured_output(self, schema=None, **kwargs):  # type: ignore[override]
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            kwargs.setdefault("method", "function_calling")
            return super().with_structured_output(schema, **kwargs)
        return _ToolCallStructuredRunnable(
            self,
            schema,
            force_tool_choice=settings.DEEPSEEK_THINKING is False,
        )


def get_deepseek_llm(
    model_name: str = "deepseek-flash",
    temperature: float = 1,
    override: ProviderOverride | None = None,
) -> ChatOpenAI:
    """DeepSeek on its domestic platform (https://api.deepseek.com).

    Two upstream behaviours worth knowing about:

    * **Thinking mode is on by default** (effort=high). The CoT comes back in
      ``reasoning_content`` and costs extra latency + completion tokens on
      every planner/orchestrator/cortex/executor call. ``DEEPSEEK_THINKING=false``
      sends ``{"thinking": {"type": "disabled"}}`` to turn it off; unset, we
      send nothing and keep DeepSeek's default.
    * ``temperature`` (and the penalty params) are **ignored while thinking**,
      they only apply in non-thinking mode. They are accepted either way, so
      there is nothing to clamp here.
    """
    override = override or ProviderOverride()
    api_key = override.key_or("DEEPSEEK_API_KEY")
    assert api_key is not None
    extra_kwargs: dict[str, Any] = {}
    if settings.DEEPSEEK_THINKING is not None:
        extra_kwargs["extra_body"] = {
            "thinking": {"type": "enabled" if settings.DEEPSEEK_THINKING else "disabled"}
        }
    client = DeepSeekChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        base_url=override.url_or("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        **extra_kwargs,
    )
    return client


@overload
def get_llm(
    ctx: MobileUseContext,
    name: AgentNodeWithFallback,
    *,
    use_fallback: bool = False,
    temperature: float = 1,
) -> BaseChatModel: ...


@overload
def get_llm(
    ctx: MobileUseContext,
    name: LLMUtilsNode,
    *,
    is_utils: Literal[True],
    temperature: float = 1,
) -> BaseChatModel: ...


@overload
def get_llm(
    ctx: MobileUseContext,
    name: LLMUtilsNodeWithFallback,
    *,
    is_utils: Literal[True],
    use_fallback: bool = False,
    temperature: float = 1,
) -> BaseChatModel: ...


def get_llm(
    ctx: MobileUseContext,
    name: AgentNode | LLMUtilsNode | AgentNodeWithFallback,
    is_utils: bool = False,
    use_fallback: bool = False,
    temperature: float = 1,
) -> BaseChatModel:
    # 所有 agent 都从这里拿客户端，所以 usage 捕获的补丁也在这里装一次（幂等）。
    _install_usage_capture()
    llm = (
        ctx.llm_config.get_utils(name)  # type: ignore
        if is_utils
        else ctx.llm_config.get_agent(name)  # type: ignore
    )
    if use_fallback:
        if isinstance(llm, LLMWithFallback):
            llm = llm.fallback
        else:
            raise ValueError("LLM has no fallback!")

    # 渠道下发的接入参数。没下发时 llm.base_url / api_key 都是 None,
    # ProviderOverride 就是空的,工厂照样落回 settings —— 行为与改动前一致。
    override = ProviderOverride(api_key=llm.api_key, base_url=llm.base_url)
    # 温度同理:渠道给了就用渠道的,没给就用各 agent 调用点的硬编码值
    # (hopper=0 / video_analyzer=0.2 / 其余 1)。
    effective_temperature = llm.temperature if llm.temperature is not None else temperature

    if llm.provider == "anthropic":
        return get_anthropic_llm(llm.model, effective_temperature, override)
    elif llm.provider == "openai":
        return get_openai_llm(llm.model, effective_temperature, override)
    elif llm.provider == "google":
        return get_google_llm(llm.model, effective_temperature, override)
    elif llm.provider == "vertexai":
        return get_vertex_llm(llm.model, effective_temperature, override)
    elif llm.provider == "openrouter":
        return get_openrouter_llm(llm.model, effective_temperature, override)
    elif llm.provider == "xai":
        return get_grok_llm(llm.model, effective_temperature, override)
    elif llm.provider == "azure":
        return get_azure_llm(llm.model, effective_temperature, override)
    elif llm.provider == "minimax":
        return get_minimax_llm(llm.model, effective_temperature, override)
    elif llm.provider == "deepseek":
        return get_deepseek_llm(llm.model, effective_temperature, override)
    elif llm.provider == "minitap":
        return get_minitap_llm(
            trace_id=ctx.trace_id,
            model=llm.model,
            temperature=effective_temperature,
            agent_name=name,
            override=override,
        )
    else:
        raise ValueError(f"Unsupported provider: {llm.provider}")


T = TypeVar("T")


async def with_fallback[T](
    main_call: Callable[[], Awaitable[T]],
    fallback_call: Callable[[], Awaitable[T]],
    none_should_fallback: bool = True,
) -> T:
    try:
        result = await main_call()
        if result is None and none_should_fallback:
            llm_logger.warning("Main LLM inference returned None. Falling back...")
            return await fallback_call()
        return result
    except Exception as e:
        llm_logger.warning(f"❗ Main LLM inference failed: {e}. Falling back...")
        return await fallback_call()
