import asyncio
import functools
import inspect
import logging
import re
import sys
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar, cast, overload

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, PrivateAttr, SecretStr, ValidationError

from minitap.mobile_use.config import (
    AgentNode,
    AgentNodeWithFallback,
    LLMUtilsNode,
    LLMUtilsNodeWithFallback,
    LLMWithFallback,
    settings,
)
from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.services.channels import (
    ChannelSpec,
    normalize_base_url,
    require_channel,
)
from minitap.mobile_use.services.telemetry import telemetry
from minitap.mobile_use.utils.llm_usage import (
    close_capture,
    extract_cache_tokens,
    extract_usage,
    first_captured,
    open_capture,
    record_raw_response,
)
from minitap.mobile_use.utils.logger import get_logger
from minitap.mobile_use.utils.tool_schema_surgery import apply_tool_schema_surgery

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

    这是 manager 侧「渠道」下发过来的 key / 端点在这里的落点。每个 provider 工厂默认
    都按「override 优先、settings 兜底」解析,于是没下发渠道的部署行为与之前逐字节
    一致 —— 这是这条路径能被安全打开的前提。

    ``strict`` 打开时,「settings 兜底」这一层被**去掉**:只认渠道下发的值。
    它由 :attr:`~minitap.mobile_use.config.LLM.allow_env_fallback` 的反值决定,
    也就是「这条接入参数是运行期下发的渠道」时为 True。

    为什么渠道必须 strict:manager 明确指定了用哪个模型、哪个端点、哪把密钥,
    这时候再从设备本地的 .env 里补一把,症状是「我明明选了 M3.1,跑起来是 .env 里
    那个模型」—— 而且它**不报错**。渠道下发时缺了什么,应该响亮地失败,
    而不是悄悄用另一个接入点。
    """

    api_key: SecretStr | None = None
    base_url: str | None = None
    strict: bool = False

    #: 渠道下发的「要不要思考」开关。None = 没下发 = 用 provider / settings 的默认。
    #: 消费它的是 ``llamacpp``（见 :func:`get_llamacpp_llm`）和 ``minimax``
    #: （见 :func:`get_minimax_llm`）；能不能真关掉由 provider 自己判断。
    thinking: bool | None = None

    #: 渠道下发的「思考强度」。None = 没下发 = 用服务端默认档位。
    #:
    #: 消费它的只有目录里 ``effort_style="reasoning_effort"`` 的渠道（DeepSeek）。
    #: 能不能发、有哪些档位由渠道目录里的 ``effort_levels`` 决定；写错档位是
    #: **显式报错**而不是静默回落 —— 静默回落的症状是「选了 low，花的还是 high
    #: 的 token」，没有任何地方能看出来。
    reasoning_effort: str | None = None

    def key_or(self, attr: str, required: bool = True) -> SecretStr | None:
        if self.strict:
            if not self.api_key:
                if not required:
                    # 这个 provider 可以不要密钥（azure 走 DefaultAzureCredential）。
                    return None
                # 到这里说明渠道漏了密钥。明确说清楚「哪一环断了 + 去哪补」,
                # 而不是让调用方的 assert 打出一个没有信息量的 AssertionError。
                raise ValueError(
                    f"渠道没有下发 API Key，而且这条路径不允许回退到设备本地的 .env（{attr}）。"
                    f"请在 manager 的渠道配置里补上密钥，或换一条带密钥的渠道。"
                )
            return self.api_key
        return self.api_key or getattr(settings, attr, None)

    def url_or(self, attr: str | None, default: str | None = None) -> str | None:
        if self.strict:
            # 渠道没给端点时用 provider 自带的默认端点（那是代码里的常量，
            # 不是设备本地环境的一部分），但绝不读 .env 里的 <PROVIDER>_BASE_URL。
            return self.base_url or default
        env_value = getattr(settings, attr, None) if attr else None
        return self.base_url or env_value or default

    def thinking_or(self, attr: str | None) -> bool | None:
        """「要不要思考」：渠道下发的优先，没下发时按 strict 决定要不要读 .env。

        和 :meth:`key_or` / :meth:`url_or` 同一套语义：**渠道下发的东西就是全部**。
        strict 时绝不读 .env —— 渠道说了「开」就开、说了「关」就关、没说就不发这个
        字段（走供应商自己的默认），而不是被设备本地的环境变量改写。

        以前 DeepSeek 只读 ``settings.DEEPSEEK_THINKING``，**完全无视渠道下发的
        thinking**：症状是「我在 manager 里把思考关了，它还在思考」，于是
        「思考模式下强制 tool_choice」被服务端拒绝（400），每个 agent 节点全挂。
        """
        if self.thinking is not None:
            return self.thinking
        if self.strict or not attr:
            return None
        value = getattr(settings, attr, None)
        # isinstance 检查是为了挡住「settings 被测试打桩成 MagicMock」这种情况 ——
        # 那时 ``settings.DEEPSEEK_THINKING`` 是个对象而不是 bool，当作「没设置」。
        return value if isinstance(value, bool) else None


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

        done, _ = await asyncio.wait({llm_task, waiter_task}, return_when=asyncio.FIRST_COMPLETED)

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

    **拿不到就不记**（而不是记 0）：「provider 没报 usage」和「这次调用零消耗」
    是两件事，记 0 会让成本看起来像免费。

    ⚠️ **本段曾写着一条假结论，已于 2026-10-21 实测推翻并改写。**
    原文是：「在 LangGraph 的图里那个捕获槽根本收不到东西（ContextVar 跨不过
    图的任务上下文），图里的 usage 由 `services/usage_bridge.py` 的 graph-config
    callback 上报」。两处都是错的：

    1. **`services/usage_bridge.py` 从未存在**（git 全历史无此文件），
       也从来没有任何 import —— 所谓「图里的 usage 由它上报」是**不存在的机制**。
    2. **捕获槽跨得过去**。`utils/llm_usage.py` 的槽里装的是**同一个 list
       对象**（ContextVar 被复制时复制的是引用），所以图里 append 的原始
       AIMessage 在外面读得到。

    实测证据（真机 graph 运行，2026-10-21）：
    ``probe_out/executor_e2e.json`` —— executor 21 次调用 / 120,772 input /
    65,672 cached / 18,114 output，**全是真实非零值**；
    ``probe_out/device_usage.jsonl`` 里 cortex / hopper / planner 同样非零。

    那次「每一行都是 0」的真因不是 ContextVar，而是**客户端没开
    ``stream_usage=True``**（provider 压根没报 usage）—— 已在
    `services/llm.py:778` 修好。**结论：图内 usage 埋点是通的，不要再造
    usage_bridge。**
    """
    if not agent_name:
        return
    try:
        usage = extract_usage(raw_response) or extract_usage(result)
        if usage is None:
            return
        input_tokens, output_tokens = usage
        telemetry.capture_llm_usage(
            task_id=task_id or "unknown",
            agent=agent_name,
            model=_extract_model_name(raw_response) or _extract_model_name(result),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            # 前缀缓存命中量是第三个数，单独记 —— 它是「提示词分层有没有用」
            # 唯一的可观测证据（2026-10-21 起）。
            cached_tokens=extract_cache_tokens(raw_response),
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
    else:
        # key_or 自己处理两种语义：strict（渠道路径）只认渠道下发的密钥，缺了就报错；
        # 非 strict 才是「渠道 → settings」的老兜底。minitap 原来是自己手写的
        # 三段 if，漏掉了 strict 这一层。
        effective_api_key = override.key_or("MINITAP_API_KEY")
        if effective_api_key is None:
            raise ValueError("MINITAP_API_KEY must be provided or set in environment")

    # 显式给的 key 优先于渠道/环境里那份。
    override = ProviderOverride(
        api_key=effective_api_key,
        base_url=override.base_url,
        thinking=override.thinking,
        reasoning_effort=override.reasoning_effort,
        strict=override.strict,
    )

    return build_openai_protocol_client(
        require_channel("minitap"),
        model,
        temperature if temperature is not None else 1,
        override,
        agent_name=agent_name,
        trace_id=trace_id,
    )


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


class ToolSchemaSurgeryMixin:
    """给 OpenAI 兼容端点的**工具 schema** 做结构手术。

    手术内容：删 ``title``、把 ``anyOf: [X, null]`` 折成 ``X``（保留
    ``default: null``）。只动结构，不动任何语义描述。

    为什么在这里做
    --------------
    ``_get_request_payload`` 是 langchain 把**所有** ``bind_tools`` 汇成请求体的
    唯一出口 —— executor 的 16 个工具和 cortex 的结构化输出 schema 都从这儿走。
    挂在这里的好处：
    - 一处覆盖全部 OpenAI 兼容 provider（minimax / deepseek / openai /
      openrouter / xai / minitap），不用逐个调用点改；
    - **对工具实现零侵入**（工具函数不知道自己被改了 schema）；
    - 手术只影响**发给 provider 的 body**，本地 pydantic 模型原样不动。

    实测：executor 的 16 个工具 4,333 → 3,871 token（−462，−10.7%）。

    **provider 不校验 tool_call 参数的 schema**（实测 MiniMax：显式 null 的参数
    在手术前后都是 HTTP 200），所以 ``anyOf`` 折叠不会因为「模型传了 null」被拒。
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs) -> dict:  # type: ignore[override]
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)  # type: ignore[misc]
        return apply_tool_schema_surgery(payload)

    def should_force_tool_choice(self) -> bool:
        """这次调用能不能**强制**指定 tool_choice（把 schema 当工具绑上去时）。

        默认能 —— 多数端点随便强制。DeepSeek 覆盖它：思考模式下强制是硬 400
        （"Thinking mode does not support this tool_choice"）。
        """
        return True


class SurgeryChatOpenAI(ToolSchemaSurgeryMixin, ChatOpenAI):
    """``ChatOpenAI`` + 工具 schema 结构手术。"""


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
    # 端点归一化：Anthropic 协议的客户端拼的是 ``/v1/messages``，所以基址要停在
    # 根上 —— 用户粘 ``https://api.anthropic.com/v1/messages`` 进来时，
    # 不剥掉就会请求 ``.../v1/v1/messages``。
    anthropic_url = normalize_base_url(override.base_url, "anthropic")
    client = AnthropicCacheControlChat(
        model_name=model_name,
        api_key=api_key,
        # `anthropic_api_url` 是 ChatAnthropic 的端点字段名(不是 `base_url`,
        # 见 langchain_anthropic/chat_models.py:946)。这里只认渠道下发的值 ——
        # Settings 里没有 ANTHROPIC_BASE_URL 这个键,不要凭空发明一个。
        anthropic_api_url=anthropic_url,
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
    """OpenAI 官方端点。

    现在只是一层薄封装：真正的实现是 :func:`build_openai_protocol_client`
    （OpenAI 兼容协议的那**一份**实现），渠道之间的差别全在渠道目录里。
    """
    return build_openai_protocol_client(
        require_channel("openai"), model_name, temperature, override
    )


def get_openrouter_llm(
    model_name: str,
    temperature: float = 1,
    override: ProviderOverride | None = None,
):
    return build_openai_protocol_client(
        require_channel("openrouter"), model_name, temperature, override
    )


def get_grok_llm(
    model_name: str,
    temperature: float = 1,
    override: ProviderOverride | None = None,
) -> ChatOpenAI:
    return build_openai_protocol_client(require_channel("xai"), model_name, temperature, override)


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
    # azure 可以完全不带 key（走 DefaultAzureCredential），所以这里不能要求一定有。
    key = override.key_or("AZURE_API_KEY", required=False)
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


class StructuredOutputError(ValueError):
    """结构化输出解析失败，带一段**能直接回灌给模型**的理由。

    为什么要有这个类
    --------------
    解析失败只有两种常见形态，而它们要**告诉模型的话完全不同**：

    * **喊错了工具名** —— 真机实测（2026-10-22，Redmi Turbo 4 / 快手极速版，
      执行 812）：executor 66 次调用里有 **37 次**喊的是 prompt「工具表」里的
      真实工具名（``focus_and_input_text`` / ``tap`` / ``wait_for_delay`` …），
      而不是唯一绑定的 ``ExecutorOutput``。
      ⚠️ **病因不是「模型不听话」**：``tool_choice`` 当时是**强制**的，模型照样
      喊错 —— 因为**局部历史在教它模仿**（历史里 assistant 消息的 tool_calls
      带着设备工具名）。修历史在
      ``agents/executor/prompt.py:rewrite_history_tool_calls``；这里只是**兜底**，
      防止漏网的那个把动作吞掉。
    * **回了一堆散文** —— 压根没调工具，直接写了一段话。

    以前这两种都变成同一句笼统的「读不懂」，模型不知道错在哪 → 连着重犯
    （实测：同一处打字动作连着 8 轮零动作，每次都是同一条无关的拒绝理由）。

    ⇒ ``reason`` 是**给模型**的（短、能照着改），``facts`` 是**可进日志的摘要**
    （工具名、finish_reason —— 不含屏幕原文），``detail`` 是**完整诊断**
    （含模型原文，只在异常 message 里，不外传）。

    为什么要分三层：模型的原文里可能有**屏幕上的文字**（用户数据）。日志是要
    长期留存的，把原文抄进去等于把用户数据存到日志系统里。遥测那边早有同款
    约定（见 ``agents/cortex/dispatcher.py`` 的 ``_rejection_code``：只放代码，
    不放理由原文）。

    继承 ``ValueError``：老代码按 ``ValueError`` 抓的地方不用改。
    """

    def __init__(self, reason: str, facts: str = "", detail: str = ""):
        super().__init__(f"{reason} [detail: {detail}]" if detail else reason)
        self.reason = reason
        self.facts = facts
        self.detail = detail


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

    1. If a ``tool_call`` **whose name is the schema's own tool name** is
       present, validate its ``args``.  The name check is **not** optional:
       a model may ignore ``tool_choice`` and call a different tool, and
       validating that call's args against the schema would silently
       succeed (every field has a default) and swallow the action.  See
       :meth:`_parse`.
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
        # 绑定的工具名。**解析时必须拿它对账**（见 `_parse`）：模型可以去调别的
        # 工具名（真机实测 2026-10-22，executor 上 66 次调用里有 37 次喊的是
        # 局部历史里的设备工具名）—— 哪怕 ``tool_choice`` 是强制的也一样。
        tool_name = convert_to_openai_tool(schema)["function"]["name"]
        self._tool_name = tool_name
        if force_tool_choice:
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

    @staticmethod
    def _call_field(call: Any, field: str) -> Any:
        """从一条 tool call 里取字段 —— **dict 与对象两种形态都认**。

        langchain 的 ``ToolCall`` / ``InvalidToolCall`` 运行时都是 TypedDict
        （就是 dict），但测试里常用带属性的替身 ⇒ 两种取法都得认。
        """
        if isinstance(call, dict):
            return call.get(field)
        return getattr(call, field, None)

    def _parse(self, msg: AIMessage) -> BaseModel:
        # 1) tool_calls 路径 (M3 thinking 开 / M2.7 等)
        # Note: LangChain normalises OpenAI tool_calls into ``{"name", "args",
        # "id", "type"}`` dicts where ``args`` is already parsed into a Python
        # object, NOT a JSON string. So we use ``model_validate`` (dict) or
        # ``model_validate_json`` (str) depending on the type we actually get.
        #
        # ⚠️ **必须核对工具名**。模型可以去调别的工具（真机实测：它照着**局部
        # 历史**里 assistant 消息的设备工具名学，见
        # ``agents/executor/prompt.py:rewrite_history_tool_calls``），
        # 而这里以前**不查名字**、直接拿 args 去 validate —— schema 的字段
        # 全都有默认值（``actions`` 默认 ``[]``）、pydantic 默认 ``extra='ignore'``，
        # **校验会「通过」**，于是产出一个空动作的假结果：动作被静默吞掉，
        # 外面只看见「你返回了空动作列表」，模型不知道自己错在哪 → 连着重犯。
        # （真机实测：执行 812 的 executor 66 次调用里 37 次这样，浪费 243 秒。）
        wrong_tools: list[str] = []
        for tc in msg.tool_calls or []:
            name = self._call_field(tc, "name")
            args = self._call_field(tc, "args")
            if args is None:
                continue
            if name != self._tool_name:
                wrong_tools.append(str(name))
                continue
            try:
                if isinstance(args, str):
                    return self.schema.model_validate_json(args)
                return self.schema.model_validate(args)
            except ValidationError:
                continue

        # 1b) 🔴 **invalid_tool_calls** —— 模型**确实调了工具**，但 ``arguments``
        #     不是合法 JSON，langchain 解析失败就把整个调用丢进
        #     ``msg.invalid_tool_calls``（⚠️ 是 AIMessage 的**顶层属性**，
        #     **不在** ``additional_kwargs`` 里 —— 查错地方会以为它不存在）。
        #
        #     框架以前只读 ``msg.tool_calls`` ⇒ 这类响应被当成
        #     「模型什么都没说」（"your answer was empty"）⇒ **凭空消失**。
        #
        #     实测（2026-10-24，DeepSeek deepseek-flash，真实 API）：模型偶尔
        #     把 arguments 包成 markdown 围栏或前后加说明文字 ⇒ 落在这一路。
        #
        #     这里的 ``args`` 是**原始字符串**（langchain 没解析成功），
        #     所以要自己救：去掉围栏 / 取第一个配平的 {..} 块再校验。
        invalid_calls = list(getattr(msg, "invalid_tool_calls", None) or [])
        for itc in invalid_calls:
            name = self._call_field(itc, "name")
            raw_args = self._call_field(itc, "args")
            if name != self._tool_name:
                wrong_tools.append(str(name))
                continue
            if isinstance(raw_args, dict):
                try:
                    return self.schema.model_validate(raw_args)
                except ValidationError:
                    continue
            if not isinstance(raw_args, str):
                continue
            recovered = self._extract_json_object(self._strip_markdown_json(raw_args))
            if not recovered:
                continue
            try:
                return self.schema.model_validate_json(recovered)
            except ValidationError:
                continue

        # 2) content 路径 (M3 thinking 关 / Markdown JSON 兜底)
        content = msg.content if isinstance(msg.content, str) else ""

        stripped = self._strip_markdown_json(content) if content.strip() else ""
        # 2a) 整段就是 JSON
        # 2b) JSON 被包在一堆解释性文字里（"好的，我来…" / "[orchestrator] …"）——
        #     整段当 JSON 解析必然失败，退回提取第一个配平的 {..} 块。
        for candidate in (stripped, self._extract_json_object(stripped) if stripped else None):
            if not candidate:
                continue
            try:
                return self.schema.model_validate_json(candidate)
            except ValidationError:
                continue

        # 两条路都失败。抛**带理由**的错，让上层能把「错在哪」讲回模型 ——
        # 光说「读不懂」模型只会换个姿势再犯（这是执行 812 空转 42 轮的根因）。
        #
        # facts 是**可进日志的摘要**：工具名、finish_reason —— 不含屏幕原文。
        # detail 带上原文：pydantic 的 json_invalid 会把 input_value 截断成
        # '...'，现场根本看不出模型到底说了什么，而 with_fallback 正是靠这条
        # message 上报给用户的。带上原文才算可诊断 —— 但它只进异常 message，
        # 不进日志（原文里可能有屏幕上的文字）。
        invalid_names = [self._call_field(itc, "name") for itc in invalid_calls]
        facts = (
            f"tool_calls={len(msg.tool_calls or [])}, "
            f"invalid_tool_calls={len(invalid_calls)}, "
            f"wrong_tools={sorted(set(wrong_tools))}, "
            f"invalid_tool_names={sorted({str(n) for n in invalid_names if n})}, "
            f"finish_reason={getattr(msg, 'response_metadata', {}).get('finish_reason')!r}"
        )
        detail = f"{facts}, content[:1000]={content[:1000]!r}"
        if wrong_tools:
            listed = ", ".join(f"'{n}'" for n in sorted(set(wrong_tools)))
            reason = (
                f"you called the tool {listed} but the only tool you may call "
                f"is '{self._tool_name}' — call '{self._tool_name}' instead"
            )
        elif invalid_calls:
            # 模型调了工具，但 arguments 不是合法 JSON（围栏 / 说明文字），
            # 而救不回来 ⇒ 说清楚是**哪一环**坏了，别让它以为「什么都没说」。
            reason = (
                f"you called '{self._tool_name}' but its arguments were not valid JSON — "
                f"pass the arguments as a single JSON object with no markdown fences "
                f"and no extra text"
            )
        elif content.strip():
            reason = (
                f"your answer was not readable as {self.schema.__name__} — call the "
                f"'{self._tool_name}' tool instead of writing prose"
            )
        else:
            reason = f"your answer was empty — call the '{self._tool_name}' tool"
        raise StructuredOutputError(reason=reason, facts=facts, detail=detail)


class MiniMaxChatOpenAI(ToolSchemaSurgeryMixin, ChatOpenAI):
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


def minimax_model_family(model_name: str) -> str:
    """从模型名里取出家族号：``MiniMax-M3.1-Flash-Preview`` -> ``m3.1``。

    MiniMax 的模型名是 ``MiniMax-<家族><.小版本>[-后缀]``：``MiniMax-M3``、
    ``MiniMax-M3.1-Flash``、``MiniMax-M2.7-highspeed``。取 ``MiniMax-`` 之后
    到下一个 ``-`` 为止的部分就是家族号。
    """
    name = (model_name or "").strip().lower()
    if name.startswith("minimax-"):
        name = name[len("minimax-") :]
    return name.split("-", 1)[0]


#: **实测**能把思考真正关掉的家族（``thinking.type=disabled`` 生效且不 400）。
#:
#: 2026-10-22 真机实测，OpenAI 兼容端点 ``https://api.minimaxi.com/v1``
#: （探针：mobile-manager/scripts/probe_minimax_thinking.py）：
#:
#: ============  =========  ==========================  ======================
#: 家族          默认        ``thinking.type=disabled``   ``thinking.type=adaptive``
#: ============  =========  ==========================  ======================
#: ``m2.x``      思考开      200，但**无效**（思考照旧）   200
#: ``m3``        思考开      ✅ 200，思考真的关掉了       200
#: ``m3.1``      强制开      ❌ 400（"requires adaptive"） 200
#: ============  =========  ==========================  ======================
#:
#: 另外 ``thinking.type=enabled``（M2 时代的写法）在 **M3 上直接 400** ——
#: 只有 ``adaptive`` 和 ``disabled`` 两种取值是 M3 认的。
#:
#: 官方文档只覆盖 Anthropic 兼容端点（那里写「MiniMax-M3 省略 thinking 时默认
#: 关闭」）。本项目用的是 **OpenAI 兼容**端点，实测**默认是开的** —— 以实测为准。
MINIMAX_THINKING_DISABLEABLE_FAMILIES = frozenset({"m3"})


def resolve_minimax_thinking(model_name: str, requested: bool | None) -> dict[str, Any] | None:
    """把「要不要思考」翻译成 MiniMax 请求体里的 ``thinking`` 片段。

    返回 ``None`` 表示**不发**这个字段（走服务端默认）。

    为什么不能把渠道的开关直接透传给 API：**M3.1 发 ``disabled`` 是硬 400**
    （每个请求都 400，agent 直接死），M2.x 发 ``disabled`` 是「接受了但没生效」
    （用户以为关掉了，实际还在思考）。所以这里先看模型家族再决定发不发。

    关不掉的时候**不硬拦**（照常跑），但要**报出来** —— 静默忽略会变成
    「我明明拨了开关，模型还在思考」。
    """
    if requested is None:
        return None
    if requested:
        # ``adaptive`` 是三个家族都认的写法（M3.1 只认 adaptive）。
        return {"type": "adaptive"}

    family = minimax_model_family(model_name)
    if family in MINIMAX_THINKING_DISABLEABLE_FAMILIES:
        return {"type": "disabled"}

    llm_logger.warning(
        f"渠道里把 {model_name!r} 的思考模式关了，但这个模型关不掉："
        f"它属于 {family!r} 家族，官方强制开启思考"
        f"（发 disabled 会直接 400，所以这次没发）。"
        f"想省 token 得换 MiniMax-M3 —— M3 实测能关。"
    )
    return None


def get_minimax_llm(
    model_name: str = "MiniMax-M2.7",
    temperature: float = 1,
    override: ProviderOverride | None = None,
) -> ChatOpenAI:
    """构造 MiniMax 客户端。

    下发 ``output_config.effort``（默认 ``max``）。M3.1-Flash **强制**思考
    （官方：always thinks），只能调深度；这个值决定思考花多少 token 和多少时间。

    **思考开关**：渠道下发（``thinking``）优先，否则用 ``settings.MINIMAX_THINKING``，
    都没有就不发这个字段。能不能真关掉**看模型家族**（实测表见
    :data:`MINIMAX_THINKING_DISABLEABLE_FAMILIES`）：

    * ``MiniMax-M3`` —— **能关**：发 ``{"thinking": {"type": "disabled"}}``，
      实测 200、回答里没有 ``<think>``、``reasoning_tokens`` 也没有了。
      顺带把「M3 的思考块写进 content 且经常不闭合」这个解析杀手也解决了。
    * ``MiniMax-M3.1-Flash`` / ``M2.x`` —— **关不掉**：前者发 ``disabled`` 直接
      400（"model requires adaptive thinking"），后者接受了也不生效。
      这两种情况照常跑（不硬拦），但会打一条 warning 说明「这次没关成」。

    为什么非要换到 M3.1：真机同批 prompt n=16 实测，M3 会把思考内容写进
    ``content`` 且经常不闭合，``_ToolCallStructuredRunnable._parse`` 靠 ``</think>``
    剥不掉 -> 没有 tool_calls -> "returned no parseable CortexOutput" -> 整个
    goal 崩。M3.1 的思考走独立字段，16 次 ``content`` 里 0 次混入 <think>。
    （M3 也能解这个问题：把思考关掉就行 —— 见上面的开关。）

    实现本身在 :func:`build_openai_protocol_client`（OpenAI 兼容协议的那一份），
    这里只是渠道入口。
    """
    return build_openai_protocol_client(
        require_channel("minimax"), model_name, temperature, override
    )


#: 一次请求的思考状态。
#:
#: ``unknown`` = 请求体里**没有**思考这个字段 ⇒ 走服务端自己的默认。
#:
#: 用 Literal 而不是裸 ``str``：这三个值是穷举的，返回别的就是 bug
#: （规则：types-literal-for-fixed-sets）。
ThinkingState = Literal["on", "off", "unknown"]


class ThinkingAwareChatOpenAI(ToolSchemaSurgeryMixin, ChatOpenAI):
    """「思考开着就不能强制 tool_choice」的那类 OpenAI 兼容端点。

    **Do not simplify this back to plain ``ChatOpenAI`` without a live call.**
    这类端点（DeepSeek / MiMo）有两个共同约束：

    1. ``response_format=json_schema`` 不被支持（DeepSeek 明确回 400
       「This response_format type is unavailable now」）。⇒ 结构化输出必须
       走「schema 当工具绑」那条路，不能用 langchain 的默认实现。
    2. **思考模式下强制指定 tool_choice 会被服务端拒绝**：DeepSeek 回 400
       「Thinking mode does not support this tool_choice」；MiMo 官方文档说
       开启 thinking 后「tool_calls 输出不完整」，建议调工具时关掉思考。
       ⇒ 「能不能强制」不是一条写死的常量，而是**这次请求思考开着没有**。

    实测（2026-10-24，真实 API）::

        deepseek-flash    + 强制 tool_choice                    → HTTP 400（思考模式）
        deepseek-flash    + 强制 tool_choice + thinking.disabled → HTTP 200
        deepseek-reasoner + 强制 tool_choice                    → HTTP 400（思考模式）
        deepseek-chat     + 强制 tool_choice                    → HTTP 200（它默认不思考）
    """

    #: 该端点「**没指定**思考开关时」要不要按「开着思考」处理。
    #:
    #: 由渠道目录的 ``thinking_default_on`` 填进来（见
    #: :func:`build_openai_protocol_client`）。它决定「渠道没说思考开关」时
    #: 能不能强制 tool_choice：
    #:
    #: * DeepSeek 的推理模型**默认开**思考 ⇒ 没说 → 不强制（强制会 400）。
    #: * MiMo 的思考是**按需开启**的 ⇒ 没说 → 可以强制（思考没开，强制是安全的）。
    #:
    #: 用 PrivateAttr 而不是普通字段：普通字段会被 pydantic 当成模型参数，
    #: 而这个值**只**是客户端的本地知识，不该进请求体。
    _thinking_default_on: bool = PrivateAttr(default=True)

    def _thinking_state(self) -> ThinkingState:
        """这次请求的思考状态：``on`` / ``off`` / ``unknown``。"""
        body = self.extra_body or {}
        thinking = body.get("thinking")
        if isinstance(thinking, dict):
            kind = thinking.get("type")
            if kind == "enabled":
                return "on"
            if kind == "disabled":
                return "off"
        return "unknown"

    def should_force_tool_choice(self) -> bool:
        # 思考**开着**就不能强制（服务端会拒 / 输出不完整）；关着或明确关掉 → 强制，
        # 模型更听话，结构化输出也可靠。
        state = self._thinking_state()
        if state == "on":
            return False
        if state == "off":
            return True
        # 没指定 → 按该端点自己的默认。
        return not self._thinking_default_on

    def set_thinking_default(self, default_on: bool) -> None:
        """渠道告诉客户端「没说思考开关时，默认算开着还是关着」。

        为什么走**方法**而不是工厂里 ``isinstance(client, ThinkingAwareChatOpenAI)``：
        测试里这个类会被 ``@patch("...llm.ThinkingAwareChatOpenAI")`` 换成一个 Mock
        ⇒ ``isinstance`` 的第二个参数变成 Mock 对象，直接抛
        ``TypeError: isinstance() arg 2 must be a type``。

        ⇒ 改成「问客户端自己要不要这个值」：是这类客户端就调它的方法，
        不是就压根没有这个方法（``getattr`` 拿不到）。
        """
        self._thinking_default_on = default_on

    def with_structured_output(self, schema=None, **kwargs):  # type: ignore[override]
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            kwargs.setdefault("method", "function_calling")
            return super().with_structured_output(schema, **kwargs)
        return _ToolCallStructuredRunnable(
            self,
            schema,
            force_tool_choice=self.should_force_tool_choice(),
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
    return build_openai_protocol_client(
        require_channel("deepseek"), model_name, temperature, override
    )


class _SiliconFlowStructuredRunnable(Runnable):
    """结构化输出走**两条路**：原生 json_schema 为主，工具调用为备。

    为什么必须两条
    --------------
    硅基流动上的模型分两类，能力不一样（实测，2026-10-21）：

    ==========================  ==============  ==============  ==============
    模型                          强制 tool_choice  自动 tool_choice  json_schema
    ==========================  ==============  ==============  ==============
    ``Qwen/Qwen3-VL-32B-Instruct``  400 不支持       返回空、无 tool_calls  正常
    ``Qwen/Qwen3-VL-8B-Instruct``   400 不支持       返回空              正常
    ``Qwen/Qwen3.5-4B``             正常            正常              正常
    ``Qwen/Qwen2.5-72B-Instruct``  正常            正常              正常
    ==========================  ==============  ==============  ==============

    视觉模型在硅基流动上**不支持函数调用**：强制指定函数直接被拒
    （``400 code 20015 Value error, Specifying functions for tool_choice is not
    yet supported``），不强制则返回空内容、不带 tool_calls。

    而 ``cortex`` / ``executor`` 必须用视觉模型（它们读截图），所以
    **json_schema 只能是主路径** —— 它是唯一对所有模型都成立的那条。

    备路径（把 schema 当工具绑上去）在文本模型上更硬：模型必须吐
    tool_calls，参数由我们自己用 pydantic 校验。它有两个作用：
    主路径解析失败时的第二次机会，以及给文本模型一条结构更强的路。
    备路径**不强制** tool_choice —— 强制的会被视觉模型 400，而我们要的
    是「这条备路在更多模型上能跑」。

    两条都失败时报出**两条**的原始原因，而不是只报最后一条。
    """

    def __init__(self, llm: ChatOpenAI, schema: type[BaseModel]):
        self.llm = llm
        self.schema = schema
        # 主路径：原生 response_format=json_schema。显式走 ChatOpenAI 的实现，
        # 绕开 SiliconFlowChatOpenAI 自己的覆盖（否则会无限递归）。
        self._native = ChatOpenAI.with_structured_output(llm, schema, method="json_schema")
        # 备路径：schema 当工具绑上去，不强制 tool_choice。
        self._tool = _ToolCallStructuredRunnable(llm, schema, force_tool_choice=False)

    def _fail(self, native_error: Exception, tool_error: Exception) -> ValueError:
        # 保住 tool 路径那条路的**可回灌理由**（模型喊错工具名时最关键），
        # 同时把两条路的原因都留在 detail 里 —— 测试要断言两条都在。
        reason = getattr(tool_error, "reason", None) or (
            f"the model did not return a parseable {self.schema.__name__}"
        )
        facts = getattr(tool_error, "facts", "") or f"tool path: {type(tool_error).__name__}"
        return StructuredOutputError(
            reason=reason,
            facts=f"{facts}, json_schema path: {type(native_error).__name__}",
            detail=(
                f"json_schema path: {type(native_error).__name__}: {native_error}; "
                f"tool path: {type(tool_error).__name__}: {tool_error}"
            ),
        )

    def invoke(self, input, config=None, **kwargs):
        try:
            return self._native.invoke(input, config=config, **kwargs)
        except Exception as native_error:
            try:
                return self._tool.invoke(input, config=config, **kwargs)
            except Exception as tool_error:
                raise self._fail(native_error, tool_error) from tool_error

    async def ainvoke(self, input, config=None, **kwargs):
        try:
            return await self._native.ainvoke(input, config=config, **kwargs)
        except Exception as native_error:
            try:
                return await self._tool.ainvoke(input, config=config, **kwargs)
            except Exception as tool_error:
                raise self._fail(native_error, tool_error) from tool_error


class SiliconFlowChatOpenAI(ToolSchemaSurgeryMixin, ChatOpenAI):
    """ChatOpenAI variant for SiliconFlow (硅基流动).

    **Do not simplify this back to plain ``ChatOpenAI`` without a live call.**
    The platform's model families differ in what they support, and the naive
    ``with_structured_output`` default is wrong for the one family that
    matters most here — see :class:`_SiliconFlowStructuredRunnable` for the
    measured capability table and why the schema goes out as
    ``response_format`` first.

    Tool calling, JSON mode, streaming and vision all work as expected on
    the models that support them.
    """

    def with_structured_output(self, schema=None, **kwargs):  # type: ignore[override]
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            kwargs.setdefault("method", "function_calling")
            return super().with_structured_output(schema, **kwargs)
        return _SiliconFlowStructuredRunnable(self, schema)


def get_siliconflow_llm(
    model_name: str = "Qwen/Qwen2.5-72B-Instruct",
    temperature: float = 1,
    override: ProviderOverride | None = None,
) -> ChatOpenAI:
    """SiliconFlow (硅基流动) on its OpenAI-compatible endpoint.

    Two platforms serve the same API (``api.siliconflow.cn`` and
    ``api.siliconflow.com``); we default to the ``.cn`` one and let
    ``SILICONFLOW_BASE_URL`` / a pushed channel point at either.

    ``temperature`` is passed through unchanged: the endpoint accepts 0..2
    (verified live for 0, 0.01 and 2), so the ``0`` that ``hopper`` hardcodes
    for reproducibility keeps working.
    """
    return build_openai_protocol_client(
        require_channel("siliconflow"), model_name, temperature, override
    )


# ---------------------------------------------------------------------------
# llama.cpp / llama-server（本地 OpenAI 兼容端点）
# ---------------------------------------------------------------------------
#
# 这个 provider 的所有参数都是**实测**出来的（2026-10-22，本机 llama-server
# b11782，模型 unsloth/Qwen3.5-0.8B-GGUF:Q6_K）。三件事必须记住：
#
# 1. **思考链默认开着，而且会吃光 token 预算。** 不关掉时模型的输出全在
#    ``reasoning_content`` 里，``content`` 恒为空 → 结构化输出必然解析失败。
#    决定性对照（真实 schema + 真实客户端）：
#      思考 ON  + max_tokens=256 → LengthFinishReasonError（JSON 没写完）
#      思考 ON  + 不限长         → 挂死（20s / 300s 两次超时都没返回）
#      思考 OFF + 不限长         → 2.8s 正常解析
#    ⇒ 关思考用请求级的 ``chat_template_kwargs.enable_thinking``，**不用重启服务器**。
# 2. **不限长 = 挂死。** llama.cpp 的 ``n_predict`` 默认 -1，而思考链会一直续写。
#    所以 ``max_tokens`` 必须有界（见 ``settings.LLAMACPP_MAX_TOKENS``）。
# 3. **结构化输出走 json_schema 语法约束。** llama.cpp 用 GBNF 硬约束输出，
#    实测 5 个角色的 schema 全部解析成功；「schema 当工具绑」那条路在 0.8B 上
#    会「不调工具、直接写散文」（实测 1/2 失败）。


class _LlamaCppStructuredRunnable(Runnable):
    """llama.cpp 的结构化输出：json_schema 主路径 + 工具备路径。

    与 :class:`_SiliconFlowStructuredRunnable` 同形，差别在两条路径的成功率：
    llama.cpp 的 json_schema 是**语法级**约束（GBNF），模型写不出不合法的 JSON，
    实测 5/5 解析成功；工具绑定那条路在这里只是兜底（0.8B 经常不调工具）。
    """

    def __init__(self, llm: ChatOpenAI, schema: type[BaseModel]):
        self.llm = llm
        self.schema = schema
        # 主路径：原生 response_format=json_schema。显式走 ChatOpenAI 的实现，
        # 绕开 LlamaCppChatOpenAI 自己的覆盖（否则会无限递归）。
        self._native = ChatOpenAI.with_structured_output(llm, schema, method="json_schema")
        # 备路径：schema 当工具绑上去，不强制 tool_choice。
        self._tool = _ToolCallStructuredRunnable(llm, schema, force_tool_choice=False)

    def _fail(self, native_error: Exception, tool_error: Exception) -> ValueError:
        # 同 SiliconFlow：保住 tool 路径那条路的可回灌理由，detail 里留两条路的原因。
        reason = getattr(tool_error, "reason", None) or (
            f"the model did not return a parseable {self.schema.__name__}"
        )
        facts = getattr(tool_error, "facts", "") or f"tool path: {type(tool_error).__name__}"
        return StructuredOutputError(
            reason=reason,
            facts=f"{facts}, json_schema path: {type(native_error).__name__}",
            detail=(
                f"json_schema path: {type(native_error).__name__}: {native_error}; "
                f"tool path: {type(tool_error).__name__}: {tool_error}"
            ),
        )

    def invoke(self, input, config=None, **kwargs):
        try:
            return self._native.invoke(input, config=config, **kwargs)
        except Exception as native_error:
            try:
                return self._tool.invoke(input, config=config, **kwargs)
            except Exception as tool_error:
                raise self._fail(native_error, tool_error) from tool_error

    async def ainvoke(self, input, config=None, **kwargs):
        try:
            return await self._native.ainvoke(input, config=config, **kwargs)
        except Exception as native_error:
            try:
                return await self._tool.ainvoke(input, config=config, **kwargs)
            except Exception as tool_error:
                raise self._fail(native_error, tool_error) from tool_error


class LlamaCppChatOpenAI(ToolSchemaSurgeryMixin, ChatOpenAI):
    """llama-server (llama.cpp) 的 OpenAI 兼容端点。

    **Do not simplify this back to plain ``ChatOpenAI`` without a live call.**
    见本节开头的三条实测结论 —— 少一样都会让结构化输出直接失败。
    """

    #: 这条 provider 的结构化输出走哪条路（见 :func:`get_structured_llm`）。
    #: "tool" 是默认（把 schema 当工具绑），"json_schema" 是原生语法约束。
    #: 必须是 ClassVar —— ChatOpenAI 是 pydantic 模型，未标注的类属性会被当成字段。
    structured_output_strategy: ClassVar[str] = "json_schema"

    def with_structured_output(self, schema=None, **kwargs):  # type: ignore[override]
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            kwargs.setdefault("method", "function_calling")
            return super().with_structured_output(schema, **kwargs)
        return _LlamaCppStructuredRunnable(self, schema)


def get_llamacpp_llm(
    model_name: str = "unsloth/Qwen3.5-0.8B-GGUF:Q6_K",
    temperature: float = 1,
    override: ProviderOverride | None = None,
) -> ChatOpenAI:
    """构造 llama.cpp / llama-server 的客户端。

    三个旋钮的语义（都实测过）：

    * **思考开关**：渠道下发的 ``thinking`` 优先，否则用
      ``settings.LLAMACPP_ENABLE_THINKING``（默认关）。关掉时请求里带
      ``chat_template_kwargs={"enable_thinking": false}`` —— 这是 llama.cpp 的
      模板级开关，**不需要重启服务器**。
    * **max_tokens**：必须有界。思考链开着时预算要宽得多（思考 + 正文共用），
      所以有两个上限而不是一个。
    * **密钥**：本地 server 默认不鉴权，占位 key 即可；server 起了 ``--api-key``
      时才需要真密钥（渠道下发，或写进 ``.env`` 的 ``LLAMACPP_API_KEY``）。
    """
    return build_openai_protocol_client(
        require_channel("llamacpp"), model_name, temperature, override
    )


@overload
def get_llm(
    ctx: MobileUseContext,
    name: LLMUtilsNode,
    *,
    is_utils: Literal[True],
    temperature: float = 1,
) -> BaseChatModel: ...


#: OpenAI 兼容协议下的客户端变体（见
#: :data:`~minitap.mobile_use.services.channels.ClientKind`）。
#:
#: 这里存的是**类名**而不是类本身：类要在模块加载完才都定义好，而且用类名
#: 延迟解析，测试里 ``patch("...llm.ThinkingAwareChatOpenAI")`` 那种替换仍然生效
#: （直接持有类引用的话，替换发生在引用之后，就拦不住了）。
#:
#: 同一个协议，不同供应商对「结构化输出」「工具 schema」「思考链」的处理不一样，
#: 所以客户端分几个变体 —— 但**协议只有一份实现**（下面的
#: :func:`build_openai_protocol_client`）。
#:
#: ⚠️ **一条渠道 = 一个类**：DeepSeek 和 MiMo 都是「思考感知」那一类，行为
#: 完全相同，差别只在渠道目录的 ``thinking_default_on`` 里 ⇒ 它们共用
#: ``thinking_aware``，不再单开一个行为相同的 ``deepseek`` 变体
#: （两个名字指同一份行为 = 两个所有者，迟早走样）。
_OPENAI_CLIENT_KINDS: dict[str, str] = {
    "surgery": "SurgeryChatOpenAI",
    "minimax": "MiniMaxChatOpenAI",
    "thinking_aware": "ThinkingAwareChatOpenAI",
    "siliconflow": "SiliconFlowChatOpenAI",
    "llamacpp": "LlamaCppChatOpenAI",
}


def _resolve_thinking(
    spec: ChannelSpec,
    model_name: str,
    override: ProviderOverride,
) -> tuple[bool | None, dict[str, Any]]:
    """把「要不要思考」翻译成这条渠道的请求体片段。

    返回 ``(我们发出去的值, 请求体片段)``。值是 ``None`` 表示**没发**这个字段
    —— 走供应商自己的默认。

    渠道之间的形状不一样（llama.cpp 是 ``chat_template_kwargs.enable_thinking``，
    minimax 是 ``thinking.type``，DeepSeek 也是 ``thinking.type`` 但取值不同），
    所以这个翻译按渠道目录里的 ``thinking_style`` 走，而不是让渠道去猜请求体。
    """
    requested = override.thinking_or(spec.thinking_env)
    style = spec.thinking_style

    if style == "none":
        return None, {}
    if style == "minimax":
        body = resolve_minimax_thinking(model_name, requested)
        # 关不掉时 resolve 返回 None → 什么都不发（走服务端默认），并打一条
        # warning 说明「这次没关成」。
        if not body:
            return None, {}
        return requested, {"thinking": body}
    if style == "thinking_type":
        if requested is None:
            # 没指定 → 不发这个字段，走该端点自己的默认。
            return None, {}
        return requested, {"thinking": {"type": "enabled" if requested else "disabled"}}
    if style == "chat_template_kwargs":
        value = requested if requested is not None else spec.thinking_fallback
        if value is None:
            return None, {}
        return value, {"chat_template_kwargs": {"enable_thinking": value}}
    return None, {}


def _resolve_reasoning_effort(
    spec: ChannelSpec,
    thinking_value: bool | None,
    requested: str | None,
) -> str | None:
    """把渠道下发的「思考强度」归一化成 ``reasoning_effort`` 的合法取值。

    返回 ``None`` 表示**不发**这个字段（走服务端默认档位）。

    为什么写错档位要报错而不是静默回落：DeepSeek 对不认识的取值要么 400、
    要么按默认跑 —— 后者的症状是「选了低档，token 还是按高档烧」，没有任何
    地方能看出来。所以这里把可选档位一并说出来。

    ``thinking=False`` 时直接返回 ``None``：思考都关了，强度没有意义，发出去
    只会让服务端看到一对矛盾参数。关思考请用 ``thinking``，不要用
    ``reasoning_effort="none"``。
    """
    if spec.effort_style != "reasoning_effort":
        return None
    if thinking_value is False:
        return None
    if requested is None:
        return None
    value = str(requested).strip().lower()
    if not value:
        return None
    if value in spec.effort_levels:
        return value
    hint = (
        "；关掉思考请用 thinking=false，不是 reasoning_effort=none"
        if value == "none"
        else ""
    )
    raise ValueError(
        f"渠道 {spec.id!r} 不支持 reasoning_effort={requested!r}；"
        f"可选：{', '.join(spec.effort_levels)}{hint}"
    )


def build_openai_protocol_client(
    spec: ChannelSpec,
    model_name: str,
    temperature: float = 1,
    override: ProviderOverride | None = None,
    *,
    agent_name: str | None = None,
    trace_id: str | None = None,
) -> ChatOpenAI:
    """OpenAI 兼容协议的**唯一**实现。

    所有 OpenAI 兼容的渠道（DeepSeek / MiniMax / 硅基流动 / OpenRouter / xAI /
    minitap / llama.cpp / OpenAI 官方 / 自定义）都走这里。渠道之间的差别全部
    来自渠道目录里的**数据**（默认端点、密钥变量名、思考开关的形状、token 上限
    策略……），不再靠一堆 if/elif 分支。

    ⇒ **加一个供应商 = 加一条目录数据**，不是加一个工厂函数。

    以前这里是 8 个几乎一样的工厂（``get_openai_llm`` / ``get_grok_llm`` /
    ``get_openrouter_llm`` / ``get_minitap_llm`` / ``get_minimax_llm`` /
    ``get_deepseek_llm`` / ``get_siliconflow_llm`` / ``get_llamacpp_llm``），
    各自决定「思考开关长什么样」「能不能强制 tool_choice」—— 于是同一件事有 8 种
    写法，DeepSeek 那种「思考模式下强制 tool_choice 会 400」的坑就只在一个工厂
    里被处理了，别的路径照踩（executor / reporter 走的那条路就是照踩的那个）。
    """
    override = override or ProviderOverride()

    # ---- 密钥 ----
    if spec.api_key_env:
        api_key = override.key_or(spec.api_key_env, required=spec.requires_api_key)
    else:
        # 目录里没有对应的环境变量（「自定义」那条渠道）：密钥只能来自渠道本身。
        api_key = override.api_key
    if api_key is None:
        # OpenAI SDK 不接受空 api_key；llama server 没起 --api-key 时不校验这个头。
        api_key = SecretStr("sk-local")

    # ---- 端点 ----
    base_url = override.url_or(spec.base_url_env, spec.base_url)
    if base_url is None and spec.requires_base_url:
        raise ValueError(
            f"渠道 {spec.id!r} 必须给一个访问端点（base_url）：它没有默认地址。"
            f"请在渠道配置里填上 —— 不填的话请求会打到别的服务商，"
            f"而错误信息完全指不到真正的原因。"
        )
    # 端点归一化：用户常常直接粘供应商文档里的**完整端点**
    # （``.../v1/embeddings``、``.../v1/chat/completions``），而这个客户端会
    # 在 base_url 后面自己拼 ``/chat/completions`` —— 不剥掉尾巴就变成
    # ``.../embeddings/chat/completions`` → 404。
    # 详见 :func:`minitap.mobile_use.services.channels.normalize_base_url`。
    base_url = normalize_base_url(base_url, spec.protocol)

    # ---- 温度钳制 ----
    if spec.temperature_range:
        low, high = spec.temperature_range
        temperature = max(low, min(high, temperature))

    # ---- 请求体旋钮 ----
    thinking_value, thinking_body = _resolve_thinking(spec, model_name, override)
    extra_body: dict[str, Any] = dict(thinking_body)
    # 思考强度：只有目录里声明了 ``reasoning_effort`` 的渠道才消费它。
    reasoning_effort = _resolve_reasoning_effort(spec, thinking_value, override.reasoning_effort)
    if spec.effort_style == "output_config":
        # MiniMax 的思考深度档位。实测这个档位在本负载上不是个有效的旋钮
        # （low/medium/high/max 两列都非单调，max 反而最快）—— 但它决定花多少
        # token，写死在配置里比吃服务端默认值可控。
        effort = resolve_minimax_effort()
        if effort is not None:
            extra_body["output_config"] = {"effort": effort}

    # ---- 重试次数 ----
    max_retries = spec.max_retries
    for prefix, retries in spec.retry_rules:
        if model_name.startswith(prefix):
            max_retries = retries
            break

    client_cls = getattr(sys.modules[__name__], _OPENAI_CLIENT_KINDS[spec.client_kind])
    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "temperature": temperature,
        "base_url": base_url,
        # 显式传而不是 **dict 解包：解包会让类型检查器把整个 dict 当成
        # **kwargs 逐个参数校验，一堆误报。空的时候给 None（不发这个字段）。
        "extra_body": extra_body or None,
    }
    if reasoning_effort is not None:
        # 走 ChatOpenAI 自己的字段，而不是塞进 extra_body：langchain-openai 的
        # ``_default_params`` 会把它放进 chat/completions 的请求体；塞进
        # extra_body 反而会在它切到 Responses API 那条分支时变成另一套形状。
        kwargs["reasoning_effort"] = reasoning_effort
    if spec.stream_usage:
        # 必须显式开：langchain-openai 只有在流式请求且 stream_usage=True 时
        # 才把 provider 报的 usage 转发回来。不开的话，**每一次 LLM 调用的
        # usage_metadata 都是 None** —— 于是遥测记 0（看起来像「这次调用免费」），
        # 缓存命中率更是无从计算。
        kwargs["stream_usage"] = True
    if spec.request_timeout is not None:
        # A bounded request timeout, because the default is not one. With
        # `request_timeout` unset the OpenAI SDK waits 600s and langchain retries
        # twice, so a single hung connection parks the whole agent for 30
        # minutes with only "Waiting for LLM call response..." printed — there
        # is no error, no retry limit, no way back. Reproduced on a phone over
        # USB: the LLM call at 22:23:59 had still produced nothing at 22:36.
        # Measured normal latency is seconds, so 180s is generous; one retry
        # covers a single dropped connection without compounding a real outage.
        kwargs["request_timeout"] = spec.request_timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    if spec.max_tokens_policy == "llamacpp":
        # 思考链和正文**共用**这一个预算，所以思考开时要宽得多。
        # （实测思考 ON + 256 token 的结果是 LengthFinishReasonError：思考吃满
        # 预算，JSON 没写完。）
        kwargs["max_tokens"] = (  # type: ignore[call-arg]  # field aliased to max_completion_tokens
            settings.LLAMACPP_MAX_TOKENS_THINKING
            if thinking_value
            else settings.LLAMACPP_MAX_TOKENS
        )
    if spec.session_query_param and trace_id:
        kwargs["default_query"] = {spec.session_query_param: trace_id}
        headers = _build_custom_headers(agent_name)
        if headers:
            kwargs["default_headers"] = headers

    client = client_cls(**kwargs)
    # 该端点「没说思考开关时」默认算开着还是关着 —— 客户端据此决定能不能
    # 强制 tool_choice（见 ThinkingAwareChatOpenAI.should_force_tool_choice）。
    #
    # ⚠️ 问**客户端自己**要不要这个值，而不是 ``isinstance`` 判断：测试里
    # ``ThinkingAwareChatOpenAI`` 会被 patch 成 Mock，isinstance 的第二个参数
    # 就变成了 Mock 对象 ⇒ 直接抛 TypeError。
    apply_thinking_default = getattr(client, "set_thinking_default", None)
    if callable(apply_thinking_default):
        apply_thinking_default(spec.thinking_default_on)
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
    #
    # 渠道下发的配置 allow_env_fallback=False,于是 strict=True:工厂只认渠道里
    # 给的值,不再从设备本地的 .env 补 —— 「我明明选了 A,跑起来是 B」不能发生。
    override = ProviderOverride(
        api_key=llm.api_key,
        base_url=llm.base_url,
        # 渠道下发的「要不要思考」。None = 没下发 = provider 自己的默认。
        thinking=llm.thinking,
        # 渠道下发的「思考强度」。None = 没下发 = 服务端默认档位。
        reasoning_effort=llm.reasoning_effort,
        strict=not llm.allow_env_fallback,
    )
    # 温度同理:渠道给了就用渠道的,没给就用各 agent 调用点的硬编码值
    # (hopper=0 / video_analyzer=0.2 / 其余 1)。
    effective_temperature = llm.temperature if llm.temperature is not None else temperature

    # 按**协议**分派，不是按渠道 id 分派。
    #
    # 渠道 → 协议 → 客户端实现。渠道之间的差别（默认端点、思考开关的形状、
    # 能不能强制 tool_choice）全在渠道目录里；这里只决定「用哪套客户端」。
    spec = require_channel(llm.provider)
    if spec.protocol == "openai":
        return build_openai_protocol_client(
            spec,
            llm.model,
            effective_temperature,
            override,
            agent_name=name,
            trace_id=ctx.trace_id,
        )
    elif spec.protocol == "anthropic":
        return get_anthropic_llm(llm.model, effective_temperature, override)
    elif spec.protocol == "google":
        return get_google_llm(llm.model, effective_temperature, override)
    elif spec.protocol == "vertexai":
        return get_vertex_llm(llm.model, effective_temperature, override)
    elif spec.protocol == "azure":
        return get_azure_llm(llm.model, effective_temperature, override)
    else:  # pragma: no cover —— 目录里没有别的协议了
        raise ValueError(f"Unsupported provider: {llm.provider}")


def get_structured_llm(
    ctx: MobileUseContext,
    name: AgentNode,
    schema: type[BaseModel],
    *,
    use_fallback: bool = False,
    temperature: float = 1,
) -> Runnable:
    """拿一个「按 schema 出结构化结果」的客户端。

    为什么需要这个入口：executor / reporter 之前**直接** new 一个
    ``_ToolCallStructuredRunnable``（把 schema 当工具绑），绕过了 ``with_structured_output``。
    于是「这个 provider 怎么出结构化输出」被写死在那两个角色里 —— 像 llama.cpp 这种
    「json_schema 语法约束才是可靠路径」的 provider 就吃不到（实测 0.8B 在工具绑定那条路上
    会不调工具、直接写散文）。

    这里按**客户端自己声明的能力**分派：

    * ``structured_output_strategy == "json_schema"``（llamacpp）→ 走
      ``llm.with_structured_output(schema)``，即原生语法约束。
    * 其它（默认）→ ``_ToolCallStructuredRunnable(llm, schema)``。

    「能不能强制 tool_choice」也问客户端自己（``should_force_tool_choice()``），
    **不再写死 True**。

    ⚠️ 这条正是 DeepSeek 之前一直 400 的地方：executor / reporter 走的是这里，
    而这里恒为 ``force_tool_choice=True``，于是「思考模式 + 强制 tool_choice」被
    服务端拒绝（400 "Thinking mode does not support this tool_choice"）——
    DeepSeek 的推理模型默认就开着思考，所以每个 agent 节点全挂。
    现在由客户端根据**这次请求思考开着没有**来回答。
    """
    llm = get_llm(ctx=ctx, name=name, use_fallback=use_fallback, temperature=temperature)
    if getattr(llm, "structured_output_strategy", "tool") == "json_schema":
        return llm.with_structured_output(schema)
    # `_ToolCallStructuredRunnable` 的 llm 形参标的是 ChatOpenAI，但 get_llm 返回的
    # 是 BaseChatModel（anthropic 渠道甚至不是 ChatOpenAI）。它只用到 bind_tools，
    # 所以这里放宽成 Any —— 不去动 services/llm.py 里那个类的签名。
    force = getattr(llm, "should_force_tool_choice", None)
    force_tool_choice = force() if callable(force) else True
    return _ToolCallStructuredRunnable(
        cast("Any", llm), schema, force_tool_choice=force_tool_choice
    )


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
