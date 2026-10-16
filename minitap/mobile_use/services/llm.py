import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, Literal, TypeVar, overload

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
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
from minitap.mobile_use.utils.logger import get_logger

# Logger for internal messages (ex: fallback)
llm_logger = logging.getLogger(__name__)
# Logger for user messages
user_messages_logger = get_logger(__name__)


async def invoke_llm_with_timeout_message[T](
    llm_call: Coroutine[Any, Any, T],
    timeout_seconds: int = 10,
) -> T:
    """
    Send a LLM call and display a timeout message if it takes too long.

    Args:
        llm_call: The coroutine of the LLM call to execute.
        timeout_seconds: The delay in seconds before displaying the message.

    Returns:
        The result of the LLM call.
    """
    llm_task = asyncio.create_task(llm_call)
    waiter_task = asyncio.create_task(asyncio.sleep(timeout_seconds))

    done, _ = await asyncio.wait({llm_task, waiter_task}, return_when=asyncio.FIRST_COMPLETED)

    if llm_task in done:
        # The LLM call has finished before the timeout, cancel the timer
        waiter_task.cancel()
        return llm_task.result()
    else:
        # The timeout has been reached, display the message and wait for the call to finish
        user_messages_logger.info("Waiting for LLM call response...")
        return await llm_task


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
) -> ChatOpenAI:
    if api_key:
        effective_api_key = SecretStr(api_key)
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
        base_url=settings.MINITAP_API_BASE_URL,
        default_query={
            "sessionId": trace_id,
        },
        default_headers=custom_headers or None,
    )
    return client


def get_google_llm(
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.7,
) -> ChatGoogleGenerativeAI:
    assert settings.GOOGLE_API_KEY is not None
    client = ChatGoogleGenerativeAI(
        model=model_name,
        max_tokens=None,
        temperature=temperature,
        api_key=settings.GOOGLE_API_KEY,
        max_retries=2,
    )
    return client


def get_vertex_llm(
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.7,
) -> ChatVertexAI:
    client = ChatVertexAI(
        model_name=model_name,
        max_tokens=None,
        temperature=temperature,
        max_retries=2,
    )
    return client


def get_anthropic_llm(
    model_name: str = "claude-sonnet-4-6",
    temperature: float = 1,
) -> ChatAnthropic:
    assert settings.ANTHROPIC_API_KEY is not None
    client = ChatAnthropic(
        model_name=model_name,
        api_key=settings.ANTHROPIC_API_KEY,
        temperature=temperature,
        max_tokens=8192,  # type: ignore[call-arg]  # field aliased to max_tokens_to_sample
        max_retries=2,
        timeout=None,
        stop=None,
    )
    return client


def get_openai_llm(
    model_name: str = "o3",
    temperature: float = 1,
) -> ChatOpenAI:
    assert settings.OPENAI_API_KEY is not None
    client = ChatOpenAI(
        model=model_name,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        temperature=temperature,
    )
    return client


def get_openrouter_llm(model_name: str, temperature: float = 1):
    assert settings.OPEN_ROUTER_API_KEY is not None
    client = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=settings.OPEN_ROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    return client


def get_grok_llm(model_name: str, temperature: float = 1) -> ChatOpenAI:
    assert settings.XAI_API_KEY is not None
    client = ChatOpenAI(
        model=model_name,
        api_key=settings.XAI_API_KEY,
        temperature=temperature,
        base_url="https://api.x.ai/v1",
    )
    return client


def get_azure_llm(model_name: str, temperature: float = 1) -> AzureAIOpenAIApiChatModel:
    assert settings.AZURE_BASE_URL is not None

    api_key: str | None = None
    if settings.AZURE_API_KEY:
        api_key = settings.AZURE_API_KEY.get_secret_value()

    if api_key:
        client = AzureAIOpenAIApiChatModel(
            model=model_name,
            endpoint=settings.AZURE_BASE_URL,
            credential=api_key,
            temperature=temperature,
        )
    else:
        from azure.identity import DefaultAzureCredential

        client = AzureAIOpenAIApiChatModel(
            model=model_name,
            project_endpoint=settings.AZURE_BASE_URL,
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


def get_minimax_llm(model_name: str = "MiniMax-M2.7", temperature: float = 1) -> ChatOpenAI:
    assert settings.MINIMAX_API_KEY is not None
    clamped_temperature = max(0.0, min(1.0, temperature))
    client = MiniMaxChatOpenAI(
        model=model_name,
        api_key=settings.MINIMAX_API_KEY,
        temperature=clamped_temperature,
        base_url=settings.MINIMAX_BASE_URL or "https://api.minimax.io/v1",
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


def get_deepseek_llm(model_name: str = "deepseek-flash", temperature: float = 1) -> ChatOpenAI:
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
    assert settings.DEEPSEEK_API_KEY is not None
    extra_kwargs: dict[str, Any] = {}
    if settings.DEEPSEEK_THINKING is not None:
        extra_kwargs["extra_body"] = {
            "thinking": {"type": "enabled" if settings.DEEPSEEK_THINKING else "disabled"}
        }
    client = DeepSeekChatOpenAI(
        model=model_name,
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=temperature,
        base_url=settings.DEEPSEEK_BASE_URL or "https://api.deepseek.com",
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
    if llm.provider == "anthropic":
        return get_anthropic_llm(llm.model, temperature)
    elif llm.provider == "openai":
        return get_openai_llm(llm.model, temperature)
    elif llm.provider == "google":
        return get_google_llm(llm.model, temperature)
    elif llm.provider == "vertexai":
        return get_vertex_llm(llm.model, temperature)
    elif llm.provider == "openrouter":
        return get_openrouter_llm(llm.model, temperature)
    elif llm.provider == "xai":
        return get_grok_llm(llm.model, temperature)
    elif llm.provider == "azure":
        return get_azure_llm(llm.model, temperature)
    elif llm.provider == "minimax":
        return get_minimax_llm(llm.model, temperature)
    elif llm.provider == "deepseek":
        return get_deepseek_llm(llm.model, temperature)
    elif llm.provider == "minitap":
        return get_minitap_llm(
            trace_id=ctx.trace_id,
            model=llm.model,
            temperature=temperature,
            agent_name=name,
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
