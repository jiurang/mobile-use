import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal

import google.auth
from dotenv import load_dotenv
from google.auth.exceptions import DefaultCredentialsError
from pydantic import BaseModel, Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings

from minitap.mobile_use.services.channels import (
    CHANNELS,
    ChannelSpec,
    channel_ids,
    channel_requires_api_key,
    channel_supports_endpoint,
    get_channel,
    require_channel,
)
from minitap.mobile_use.utils.file import load_jsonc
from minitap.mobile_use.utils.logger import get_logger

### Environment Variables

load_dotenv(verbose=True)
logger = get_logger(__name__)


class Settings(BaseSettings):
    OPENAI_API_KEY: SecretStr | None = None
    GOOGLE_API_KEY: SecretStr | None = None
    XAI_API_KEY: SecretStr | None = None
    OPEN_ROUTER_API_KEY: SecretStr | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None
    AZURE_API_KEY: SecretStr | None = None
    MINIMAX_API_KEY: SecretStr | None = None
    DEEPSEEK_API_KEY: SecretStr | None = None
    MINITAP_API_KEY: SecretStr | None = None
    SILICONFLOW_API_KEY: SecretStr | None = None
    # 小米 MiMo **Token Plan** 的 key（tp- 开头），中国集群。
    #
    # ⚠️ 它和「按量付费」那把（sk- 开头，https://api.xiaomimimo.com/v1）是
    # **两把独立的 key**，混用会 401 —— 两个渠道要加两个环境变量，别共用。
    #
    # 🔴 这个字段**必须在这里声明**：``model_config`` 是 ``extra="ignore"``，
    # 没声明的变量名会被**静默丢弃** ⇒ ``key_or()`` 拿不到 ⇒ 客户端拿占位 key
    # "sk-local" 去请求 ⇒ 401 "Invalid API Key"，而错误信息完全指不到真正的原因
    # （真正的原因是「配置根本没接上」）。2026-10-24 实测踩过。
    MIMO_API_KEY: SecretStr | None = None

    OPENAI_BASE_URL: str | None = None
    AZURE_BASE_URL: str | None = None
    # OpenRouter / xAI 的端点。渠道目录里写了 ``base_url_env``，但这里**没声明**
    # ⇒ ``extra="ignore"`` 把它们静默丢弃 ⇒ 「在 .env 里改这两个渠道的地址」
    # 这个旋钮是死的（而且不报错）。2026-10-24 由「渠道引用的环境变量必须
    # 在 Settings 里声明过」那条守卫测试抓出来。
    OPEN_ROUTER_BASE_URL: str | None = None
    XAI_BASE_URL: str | None = None
    MINITAP_API_BASE_URL: str = "https://platform.minitap.ai/api/v1"
    # MiniMax runs two separate platforms whose API keys are NOT interchangeable:
    #   domestic  platform.minimaxi.com -> https://api.minimaxi.com/v1
    #   global    platform.minimax.io  -> https://api.minimax.io/v1
    # A domestic key sent to the global endpoint fails with 401 "invalid api key (2049)".
    # Falls back to the global endpoint (upstream default) when unset.
    MINIMAX_BASE_URL: str | None = None
    # `output_config.effort` —— MiniMax M3.1-Flash-Preview 的**思考深度**档位。
    #
    # 官方文档原文："MiniMax-M3.1-Flash-Preview additionally always thinks and
    # supports tunable thinking depth via output_config.effort"，取值
    # low / medium / high / xhigh / max，**默认 max**。
    #
    # 显式写出来而不是吃默认值，是为了让「我们跑的是哪一档」写在配置文件里、
    # 可被 verify_llm_override.py 审计，也避免服务端默认值悄悄变动。
    #
    # 实测（真实 cortex prompt，n=4/档）：low 379.5 tok / 5303ms、medium 399.5 /
    # 5569ms、high 446.0 / 7036ms、max 389.5 / **4247ms**。两列都**非单调**，
    # max 反而最快 —— 说明这个档位在本负载上不是个有效的旋钮，别再去调它。
    #
    # 🔴 **2026-10-26 在 executor 负载上复测过一次（n=12，更大的样本）**：
    # 重放 execution 880 的**真实 executor 提示词**（269 / 443 条消息），
    # 逐档跑 12 次，量 reasoning / 可见输出 / 延迟 / 动作是否与基准一致：
    #
    #   档位    output 中位   相对 max   延迟中位   动作同基准
    #   low        391 tok      −3.0%     4220ms       9/12
    #   medium     410 tok      −1.9%     4374ms       8/12
    #   high       485 tok     +20.3%     4892ms       8/12
    #   xhigh      409 tok      −1.5%     4510ms       5/12
    #   max        403 tok       0.0%     3535ms       9/12
    #
    # ⚠️ **n=2 时结论完全相反**（那次 high 看着省 60%）—— 样本一上去就翻案。
    # ⇒ 这个档位**不省 token、也不改质量**，两次独立实测都非单调。
    # ⇒ **别再调它。** 想降 output token，去降**调用次数**（那是 77% 的
    # reasoning 的真正来源），不是降思考深度。
    #
    # 只对 M3.1-Flash-Preview 有意义；M2.x 会忽略，M3 不认这个字段。
    MINIMAX_EFFORT: str | None = "max"
    # 「要不要思考」。None = 不下发 = 走模型自己的默认。
    #
    # **能不能真关掉看模型家族**（2026-10-22 真机实测，OpenAI 兼容端点）：
    #   MiniMax-M3       → 能关：发 ``{"thinking": {"type": "disabled"}}``，
    #                       实测 200、回答里没有 <think>、reasoning_tokens 归零。
    #   MiniMax-M3.1-Flash → 关不掉：官方强制思考，发 disabled 直接 400
    #                       （"model requires adaptive thinking"）。
    #   M2.x              → 关不掉：disabled 被接受但不生效。
    # 完整实测表见 services/llm.py 的 MINIMAX_THINKING_DISABLEABLE_FAMILIES。
    #
    # 渠道下发的 thinking 优先于它（见 services/llm.py 的 ProviderOverride）。
    MINIMAX_THINKING: bool | None = None

    # DeepSeek's OpenAI-compatible endpoint (domestic platform, platform.deepseek.com).
    # /v1 is accepted too, so both spellings work; we default to the documented one.
    DEEPSEEK_BASE_URL: str | None = None
    # SiliconFlow（硅基流动）—— 国内 OpenAI 兼容平台，platform.siliconflow.cn。
    # 端点两家都能用（.cn / .com），默认走 .cn。
    SILICONFLOW_BASE_URL: str | None = None
    # DeepSeek models answer in "thinking mode" by default (effort=high): the CoT
    # comes back in `reasoning_content` and costs extra latency + completion tokens
    # on every agent call. Set to false to send {"thinking": {"type": "disabled"}}.
    # None -> we don't send the parameter at all and DeepSeek's default applies.
    DEEPSEEK_THINKING: bool | None = None

    # 小米 MiMo 的「要不要思考」（Token Plan / 按量付费都一样）。
    #
    # None = 不发这个字段，走 MiMo 服务端自己的默认 —— 而实测（2026-10-24，
    # token-plan-cn.xiaomimimo.com，mimo-v2.6-pro）**服务端默认是开着思考的**：
    # 不下发 thinking 时 usage 里有 ``reasoning_tokens``（13~17）、
    # message 里有 ``reasoning_content``。
    #
    # ⇒ 显式 false 时发 ``{"thinking": {"type": "disabled"}}``：reasoning_tokens
    # 变成 0、completion_tokens 从 ~35 降到 17，工具调用照样 3/3 命中。
    # ⇒ 显式 true 时发 ``{"thinking": {"type": "enabled"}}``。
    #
    # ⚠️ 同样**必须声明**在这里（``extra="ignore"`` 会静默丢弃没声明的变量名）。
    MIMO_THINKING: bool | None = None

    # ---- llama.cpp / llama-server（本地 OpenAI 兼容端点）----
    #
    # 本地 server 默认不鉴权，所以这里**没有**「必须配密钥」这一说。
    LLAMACPP_BASE_URL: str = "http://127.0.0.1:9931/v1"
    # 只有 server 起了 `--api-key` 时才需要填；不填时客户端用占位 key。
    LLAMACPP_API_KEY: SecretStr | None = None
    # 单次生成的 token 上限。**必须有界**：llama.cpp 的 `n_predict` 默认 -1 = 不限长，
    # 而开着思考链的模型会一直续写思考 —— 实测不限长时请求直接挂死（20s / 300s 两次超时）。
    # 关闭思考时实测正常输出只要 46~100 token，1024 有 10 倍余量。
    LLAMACPP_MAX_TOKENS: int = 1024
    # 开启思考链时，思考 + 正文**共用**这一个预算，所以要比关闭时宽得多。
    # （实测思考 ON + 256 token 的结果是 LengthFinishReasonError：思考吃满预算，JSON 没写完。）
    LLAMACPP_MAX_TOKENS_THINKING: int = 8192
    # 默认**关闭**思考链。本地小模型默认开着思考，思考会吃光 token 预算、
    # 让结构化输出的 content 恒为空 —— 这是这类端点最常见的失败原因。
    # 运行期渠道下发的 `thinking` 开关（见 LLM.thinking）优先于这个默认值。
    LLAMACPP_ENABLE_THINKING: bool = False

    ADB_HOST: str | None = None
    ADB_PORT: int | None = None

    MOBILE_USE_TELEMETRY_ENABLED: bool | None = None

    PROJECT_NAME: str | None = None

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()


def prepare_output_files() -> tuple[str | None, str | None]:
    events_output_path = os.getenv("EVENTS_OUTPUT_PATH") or None
    results_output_path = os.getenv("RESULTS_OUTPUT_PATH") or None

    def validate_and_prepare_file(file_path: str) -> str | None:
        if not file_path:
            return None

        path_obj = Path(file_path)

        if path_obj.exists() and path_obj.is_dir():
            logger.error(f"Error: Path '{file_path}' points to an existing directory, not a file.")
            return None

        if not path_obj.suffix or file_path.endswith(("/", "\\")):
            logger.error(f"Error: Path '{file_path}' appears to be a directory path, not a file.")
            return None

        try:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            path_obj.touch(exist_ok=True)
            return file_path
        except OSError as e:
            logger.error(f"Error creating file '{file_path}': {e}")
            return None

    validated_events_path = (
        validate_and_prepare_file(events_output_path) if events_output_path else None
    )
    validated_results_path = (
        validate_and_prepare_file(results_output_path) if results_output_path else None
    )

    return validated_events_path, validated_results_path


def record_events(output_path: Path | None, events: list[str] | BaseModel | Any):
    if not output_path:
        return

    if isinstance(events, str):
        events_content = events
    elif isinstance(events, BaseModel):
        events_content = events.model_dump_json(indent=2)
    else:
        events_content = json.dumps(events, indent=2)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(events_content)


### LLM Configuration

#: 渠道 id 的取值集合 —— **由渠道目录推导**（见
#: :mod:`minitap.mobile_use.services.channels`）。
#:
#: 以前这是一份手抄的字面量：加一个供应商要同时改目录和这里，漏改的后果是
#: 「渠道在 manager 侧能存，到设备上却因为 provider 非法起不来」。现在只有目录
#: 一处是真的。
LLMProvider = Literal[*channel_ids()]

#: 这些渠道不需要 API Key：它们走环境里的服务账号凭据
#: （azure → DefaultAzureCredential，vertexai → ADC），或者端点根本不鉴权
#: （llamacpp → 本地 llama-server 默认不校验 Authorization 头）。
#: 渠道没给密钥是合法的。其余渠道缺密钥就是缺 —— 见 :func:`provider_requires_api_key`。
KEYLESS_PROVIDERS: frozenset[str] = frozenset(
    cid for cid, spec in CHANNELS.items() if not spec.requires_api_key
)

#: 这些渠道的客户端不接受自定义端点（google 的 SDK、vertexai 的 ADC 都没有
#: 「访问端点」这个旋钮）。渠道给不给 base_url 都一样，所以不强制要求。
ENDPOINTLESS_PROVIDERS: frozenset[str] = frozenset(
    cid for cid, spec in CHANNELS.items() if not spec.supports_endpoint
)


def provider_requires_api_key(provider: str) -> bool:
    """这个渠道是不是必须有一把密钥。

    判据只有一条：能不能用服务账号凭据代替。azure 和 vertexai 能，其余不能。
    """
    return channel_requires_api_key(provider)


def provider_supports_endpoint(provider: str) -> bool:
    """这个渠道是不是接受自定义访问端点。

    google / vertexai 的客户端没有端点参数，渠道里填了也会被忽略并告警。
    """
    return channel_supports_endpoint(provider)


LLMUtilsNode = Literal["outputter", "hopper", "video_analyzer"]
LLMUtilsNodeWithFallback = LLMUtilsNode
AgentNode = Literal[
    "planner",
    "orchestrator",
    "contextor",
    "cortex",
    "executor",
    "reporter",
]
AgentNodeWithFallback = AgentNode

ROOT_DIR = Path(__file__).parent.parent.parent
DEFAULT_LLM_CONFIG_FILENAME = "llm-config.defaults.jsonc"
OVERRIDE_LLM_CONFIG_FILENAME = "llm-config.override.jsonc"


def validate_vertex_ai_credentials():
    try:
        _, project = google.auth.default()
        if not project:
            raise Exception("VertexAI requires a Google Cloud project to be set.")
    except DefaultCredentialsError as e:
        raise Exception(
            f"VertexAI requires valid Google Application Default Credentials (ADC): {e}"
        )


class LLM(BaseModel):
    """一个 LLM 接入点。

    ``provider`` + ``model`` 是必填的。下面三个字段是**运行期下发**的可选接入参数,
    留空表示「沿用今天的行为」:密钥读 ``settings``,端点读 ``settings``,温度用各 agent
    调用点的硬编码值。

    为什么需要这三个字段:manager 侧管理渠道之后,要把 key / 端点 / 温度随任务一起下发到
    runner(apk 里跑的就是这份代码)。在它们存在之前,这三样只能来自 ``settings`` ——
    一个导入时读 env 的模块级单例,任何一次调用都改不了它,于是「换渠道」必须改文件重启。
    """

    provider: LLMProvider
    model: str
    base_url: str | None = None
    api_key: SecretStr | None = None
    temperature: float | None = None

    #: 「要不要让模型思考」的开关。None = 没指定 = 用该 provider 自己的默认。
    #:
    #: 为什么是**布尔**而不是一段自由 JSON：每个 provider 的思考开关在请求体里的
    #: 形状都不一样（llama.cpp 是 ``chat_template_kwargs.enable_thinking``，
    #: minimax 是 ``thinking.type``），那是**provider 的知识**，该待在 provider 的
    #: 工厂里，不该由渠道去猜。渠道只表达「开」还是「关」。
    #:
    #: 消费它的 provider：``llamacpp``（见 ``get_llamacpp_llm``）和 ``minimax``
    #: （见 ``get_minimax_llm``，只有 MiniMax-M3 真能关，M3.1 / M2.x 关不掉）。
    #: 其它 provider 收到也忽略 —— 「一个开关对某些供应商无效」比「每个供应商
    #: 一套请求体形状」好维护。
    thinking: bool | None = None

    #: 「思考强度」档位。None = 没指定 = 用服务端默认档位。
    #:
    #: 只有目录里声明了 ``effort_style="reasoning_effort"`` 的渠道消费它
    #: （DeepSeek：Chat Completions 的顶层 ``reasoning_effort``）。
    #: 合法档位由渠道目录的 ``effort_levels`` 决定（DeepSeek 是 low / high / max）；
    #: 写错档位会在建客户端时直接报错，而不是静默按默认档位跑。
    #:
    #: ``thinking=False`` 时它不生效：思考都关了，强度没有意义。
    reasoning_effort: str | None = None

    #: 缺密钥 / 缺端点时，允不允许从设备本地的环境（.env）里补。
    #:
    #: True = 老行为：settings 兜底（配置文件 ``llm-config.override.jsonc`` 这条路
    #: 一直如此，它里面的 jsonc 只有 provider/model，密钥只能来自 .env）。
    #:
    #: False = **运行期下发的渠道**：manager 明确指定了用哪个模型、哪个端点、
    #: 哪把密钥，这时再从 .env 里补一把就是「我明明选了 A，跑起来是 B」——
    #: 症状最难查，因为它不报错。所以渠道这条路必须自带全部接入参数。
    #:
    #: 由 :func:`minitap.mobile_use.sdk.utils.llm_config_for_role_overrides` 置为 False；
    #: 配置文件那条路保持默认 True。
    allow_env_fallback: bool = True

    def _has_api_key(self, attr: str) -> bool:
        """本次解析有没有可用密钥:配置里带了,或者 settings 里有。"""
        if self.api_key:
            return True
        return bool(getattr(settings, attr, None))

    @property
    def channel(self) -> "ChannelSpec":
        """这条接入点对应的**渠道定义**（默认端点、模型列表、协议、旋钮）。

        ``provider`` 这个字段的值就是渠道 id —— 名字没改（数据库、API、runner
        载荷都靠它），改的是「这个值从哪来、附带什么信息」。
        """
        return require_channel(self.provider)

    @property
    def protocol(self) -> str:
        """这条接入点用哪个协议客户端（``openai`` / ``anthropic`` / …）。

        由渠道 id 推导 —— 协议是**渠道的属性**，不是每次调用现场决定的东西。
        """
        return self.channel.protocol

    def validate_provider(self, name: str):
        # 只校验「能不能拿到密钥」,不校验密钥是哪来的 —— 配置自带是新增的合法来源,
        # 不能因为 settings 里没有就判它无效。
        spec = get_channel(self.provider)
        if spec is None:  # pragma: no cover —— provider 是 Literal,到不了这里
            raise Exception(f"{name}: unknown LLM channel {self.provider!r}")

        if spec.protocol == "vertexai":
            validate_vertex_ai_credentials()
            return

        # 没有默认端点的渠道（azure / 自定义）：必须给一个，否则请求会打到
        # 别的服务商的地址，而错误信息完全指不到真正的原因。
        if spec.requires_base_url and not (
            self.base_url or (spec.base_url_env and getattr(settings, spec.base_url_env, None))
        ):
            raise Exception(
                f"{name} requires {spec.base_url_env or 'a base URL'} for channel "
                f"{spec.id!r} (in .env or in the channel config)"
            )

        # 不校验密钥的渠道：azure 走 DefaultAzureCredential、vertexai 走 ADC、
        # llamacpp 本地 server 不鉴权 —— 加校验会打断那类现有部署。
        if spec.requires_api_key:
            if spec.api_key_env:
                if not self._has_api_key(spec.api_key_env):
                    raise Exception(f"{name} requires {spec.api_key_env} in .env")
            elif not self.api_key:
                # 目录里没有对应的环境变量（「自定义」那条渠道）：密钥只能来自
                # 渠道本身。
                raise Exception(
                    f"{name} requires an api_key for channel {spec.id!r} "
                    f"(no environment variable is known for it — "
                    f"put the key in the channel config)"
                )

    def __str__(self):
        return f"{self.provider}/{self.model}"


class LLMWithFallback(LLM):
    fallback: LLM

    def __str__(self):
        return f"{self.provider}/{self.model} (fallback: {self.fallback})"


class LLMConfigUtils(BaseModel):
    outputter: LLMWithFallback
    hopper: LLMWithFallback
    video_analyzer: LLMWithFallback | None = None


class LLMConfig(BaseModel):
    planner: LLMWithFallback
    orchestrator: LLMWithFallback
    contextor: LLMWithFallback
    cortex: LLMWithFallback
    executor: LLMWithFallback
    # Reporter 是**可选**角色：能单独配 LLM，但不是必填。
    # 做成必填会打断仓库里 22 处 `LLMConfig(...)` 构造点（probe 脚本 / example / 测试），
    # 全部变成 ValidationError（doc/executor-autonomy-plan.md 附录 A.1、审计发现 E6）。
    # None = 没单独配 → get_agent() 回落到 executor 的配置。
    reporter: LLMWithFallback | None = None
    utils: LLMConfigUtils

    def validate_providers(self):
        self.planner.validate_provider("Planner")
        self.orchestrator.validate_provider("Orchestrator")
        self.contextor.validate_provider("Contextor")
        self.cortex.validate_provider("Cortex")
        self.executor.validate_provider("Executor")
        # Reporter 没配就不校验：它走回落，用的是 executor 那条渠道的密钥。
        if self.reporter:
            self.reporter.validate_provider("Reporter")
        self.utils.outputter.validate_provider("Outputter")
        self.utils.hopper.validate_provider("Hopper")
        if self.utils.video_analyzer:
            self.utils.video_analyzer.validate_provider("VideoAnalyzer")

    def __str__(self):
        # Reporter 没单独配时要说清它回落自谁 —— 配置里看不出「谁在跑」是最难查的一类故障。
        reporter = (
            self.reporter if self.reporter is not None else f"↩️ Executor fallback ({self.executor})"
        )
        return f"""
📃 Planner: {self.planner}
🎯 Orchestrator: {self.orchestrator}
🔍 Contextor: {self.contextor}
🧠 Cortex: {self.cortex}
🛠️ Executor: {self.executor}
📰 Reporter: {reporter}
🧩 Utils:
    🔽 Hopper: {self.utils.hopper}
    📝 Outputter: {self.utils.outputter}
    🎬 Video Analyzer: {self.utils.video_analyzer or "Not configured"}
"""

    def get_agent(self, item: AgentNode) -> LLMWithFallback:
        value = getattr(self, item)
        # Reporter 是唯一可选的角色（见上面 reporter 字段的注释）。
        # 没单独配就回落 Executor —— 它只写摘要 + 挑证据，快模型就够
        # （doc/executor-autonomy-plan.md §2.1.1、附录 A.1）。
        if value is None:
            return self.executor
        return value

    def get_utils(self, item: LLMUtilsNode) -> LLMWithFallback:
        value = getattr(self.utils, item)
        if value is None:
            raise ValueError(
                f"Utils '{item}' is not configured. "
                f"Please add it to your LLM config or enable it via AgentConfigBuilder."
            )
        return value


def get_default_llm_config() -> LLMConfig:
    try:
        if not os.path.exists(ROOT_DIR / DEFAULT_LLM_CONFIG_FILENAME):
            raise Exception("Default llm config not found")
        with open(ROOT_DIR / DEFAULT_LLM_CONFIG_FILENAME, encoding="utf-8") as f:
            default_config_dict = load_jsonc(f)
        return LLMConfig.model_validate(default_config_dict["default"])
    except Exception as e:
        logger.error(f"Failed to load default llm config: {e}. Falling back to hardcoded config")
        return LLMConfig(
            planner=LLMWithFallback(
                provider="openai",
                model="gpt-5-nano",
                fallback=LLM(provider="openai", model="gpt-5-mini"),
            ),
            orchestrator=LLMWithFallback(
                provider="openai",
                model="gpt-5-nano",
                fallback=LLM(provider="openai", model="gpt-5-mini"),
            ),
            contextor=LLMWithFallback(
                provider="openai",
                model="gpt-5-nano",
                fallback=LLM(provider="openai", model="gpt-5-mini"),
            ),
            cortex=LLMWithFallback(
                provider="openai",
                model="gpt-5",
                fallback=LLM(provider="openai", model="o4-mini"),
            ),
            executor=LLMWithFallback(
                provider="openai",
                model="gpt-5-nano",
                fallback=LLM(provider="openai", model="gpt-5-mini"),
            ),
            utils=LLMConfigUtils(
                outputter=LLMWithFallback(
                    provider="openai",
                    model="gpt-5-nano",
                    fallback=LLM(provider="openai", model="gpt-5-mini"),
                ),
                hopper=LLMWithFallback(
                    provider="openai",
                    model="gpt-5-nano",
                    fallback=LLM(provider="openai", model="gpt-5-mini"),
                ),
            ),
        )


def get_default_minitap_llm_config(validate: bool = True) -> LLMConfig | None:
    """
    Returns a default LLM config using the Minitap provider.
    Only returns a config if MINITAP_API_KEY is available.

    Returns:
        LLMConfig with minitap provider if API key is available, None otherwise
    """
    if validate and not settings.MINITAP_API_KEY:
        return None

    return LLMConfig(
        planner=LLMWithFallback(
            provider="minitap",
            model="meta-llama/llama-4-scout",
            fallback=LLM(provider="minitap", model="meta-llama/llama-4-maverick"),
        ),
        orchestrator=LLMWithFallback(
            provider="minitap",
            model="openai/gpt-oss-120b",
            fallback=LLM(provider="minitap", model="meta-llama/llama-4-maverick"),
        ),
        contextor=LLMWithFallback(
            provider="minitap",
            model="meta-llama/llama-3.1-8b-instruct",
            fallback=LLM(provider="minitap", model="meta-llama/llama-3.3-70b-instruct"),
        ),
        cortex=LLMWithFallback(
            provider="minitap",
            model="google/gemini-3-pro-preview",
            fallback=LLM(provider="minitap", model="openai/gpt-5"),
        ),
        executor=LLMWithFallback(
            provider="minitap",
            model="meta-llama/llama-3.3-70b-instruct",
            fallback=LLM(provider="minitap", model="openai/gpt-5-mini"),
        ),
        utils=LLMConfigUtils(
            outputter=LLMWithFallback(
                provider="minitap",
                model="openai/gpt-5-nano",
                fallback=LLM(provider="minitap", model="openai/gpt-5-mini"),
            ),
            hopper=LLMWithFallback(
                provider="minitap",
                model="openai/gpt-5-nano",
                fallback=LLM(provider="minitap", model="openai/gpt-5-mini"),
            ),
        ),
    )


def deep_merge_llm_config(default: LLMConfig, override: dict) -> LLMConfig:
    def _deep_merge_dict(base: dict, extra: dict, path: str = ""):
        for key, value in extra.items():
            current_path = f"{path}.{key}" if path else key

            if key not in base:
                logger.warning(
                    f"Unsupported config key '{current_path}' found in override config. "
                    f"Ignoring this key."
                )
                continue

            if isinstance(value, dict) and isinstance(base[key], dict):
                _deep_merge_dict(base[key], value, current_path)
            else:
                base[key] = value

    merged_dict = default.model_dump()
    _deep_merge_dict(merged_dict, override)
    return LLMConfig.model_validate(merged_dict)


def parse_llm_config() -> LLMConfig:
    if not os.path.exists(ROOT_DIR / DEFAULT_LLM_CONFIG_FILENAME):
        return get_default_llm_config()

    override_config_dict = {}
    if os.path.exists(ROOT_DIR / OVERRIDE_LLM_CONFIG_FILENAME):
        logger.info("Loading custom llm config...")
        with open(ROOT_DIR / OVERRIDE_LLM_CONFIG_FILENAME, encoding="utf-8") as f:
            override_config_dict = load_jsonc(f)
    else:
        logger.warning("Custom llm config not found, loading default config")

    try:
        default_config = get_default_llm_config()
        return deep_merge_llm_config(default_config, override_config_dict)

    except ValidationError as e:
        logger.error(f"Invalid llm config: {e}. Falling back to default config")
        return get_default_llm_config()


def initialize_llm_config() -> LLMConfig:
    llm_config = parse_llm_config()
    llm_config.validate_providers()
    logger.success("LLM config initialized")
    return llm_config


### Output config


class OutputConfig(BaseModel):
    structured_output: Annotated[
        type[BaseModel] | dict | None,
        Field(
            default=None,
            description=(
                "Optional structured schema (as a BaseModel or dict) to shape the output. "
                "If provided, it takes precedence over 'output_description'."
            ),
        ),
    ]
    output_description: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Optional natural language description of the expected output format. "
                "Used only if 'structured_output' is not provided. "
                "Example: 'Output a JSON with 3 keys: color, price, websiteUrl'."
            ),
        ),
    ]

    def __str__(self):
        s_builder = ""
        if self.structured_output:
            s_builder += f"Structured Output: {self.structured_output}\n"
        if self.output_description:
            s_builder += f"Output Description: {self.output_description}\n"
        if self.output_description and self.structured_output:
            s_builder += (
                "Both 'structured_output' and 'output_description' are provided. "
                "'structured_output' will take precedence.\n"
            )
        return s_builder

    @model_validator(mode="after")
    def warn_if_both_outputs_provided(self):
        if self.structured_output and self.output_description:
            import warnings

            warnings.warn(
                "Both 'structured_output' and 'output_description' are provided. "
                "'structured_output' will take precedence.",
                stacklevel=2,
            )
        return self

    def needs_structured_format(self):
        return self.structured_output or self.output_description
