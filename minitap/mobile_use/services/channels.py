"""渠道目录（channel catalog）。

为什么有这个模块
----------------
以前「供应商」是**代码**：`config.py` 里一个 12 项的 ``LLMProvider`` 字面量，
`services/llm.py` 里 12 个工厂函数（``get_openai_llm`` / ``get_grok_llm`` /
``get_openrouter_llm`` …）。加一个供应商要改两个文件、抄一遍「api_key + base_url +
temperature」的样板，而且每个工厂各自决定「思考开关长什么样」「能不能强制
tool_choice」—— 于是同件事在 8 个地方有 8 种写法，DeepSeek 那种「思考模式下
强制 tool_choice 会 400」的坑就只在一个工厂里被处理了，别的路径照踩。

现在分成两层：

* **协议层**（:data:`LLMProtocol`）：客户端家族，决定「怎么发请求」。
  * ``openai`` —— OpenAI 兼容端点。DeepSeek / MiniMax / 硅基流动 / OpenRouter /
    xAI / minitap / llama.cpp / 以及任何第三方 OpenAI 兼容服务都在这一层，
    **只有一份实现**。
  * ``anthropic`` —— Anthropic 兼容端点。
  * ``google`` / ``vertexai`` / ``azure`` —— 这三个的 SDK **没有** OpenAI 兼容的
    等价物（认证方式、请求体、响应体都不同），所以保留各自的原生通道。
* **渠道层**（:class:`ChannelSpec`）：一条**数据** —— 默认端点、常用模型名、
  要不要密钥、思考开关长什么样、结构化输出走哪条路。

⇒ **新增一个供应商 = 加一条数据**，不是加一个工厂函数。

数据从哪来
----------
默认端点和模型名不是拍脑袋写的：

* DeepSeek 的模型名取自 ``GET https://api.deepseek.com/models`` 的真实返回
  （2026-10-24 实测：``deepseek-flash`` / ``deepseek-v4-pro``）。
* MiniMax 的取自 ``GET https://api.minimaxi.com/v1/models``（2026-10-24 实测：
  ``MiniMax-M3.1-Flash`` / ``MiniMax-M3`` / ``MiniMax-M2.7`` …）。
* 硅基流动的取自 ``GET https://api.siliconflow.cn/v1/models``（2026-10-24 实测，
  全量 98 个，目录里挑的是仓库配置里已经在用 + 能力对得上的那几个）。
* 其余渠道（openrouter / xai / minitap …）的模型名**留空**：它们是「模型路由
  平台」，模型列表由平台自己变，写死一份反而会过期。目录里留空 = 界面上给自由
  输入框，不假装有推荐列表。

MiniMax 的默认端点用**国内站** ``https://api.minimaxi.com/v1``：国内站的 key 发到
国际站 ``api.minimax.io`` 会 401（``invalid api key (2049)``，2026-10-24 实测），
而「填了 key 就能用」是渠道目录存在的意义。需要国际站就手填那个 URL。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

#: 客户端家族。决定「怎么发请求」。
#:
#: ``openai`` / ``anthropic`` 是**基础协议** —— 绝大多数第三方服务都能落到这两个
#: 协议上；``google`` / ``vertexai`` / ``azure`` 是保留通道，它们的 SDK 没有
#: OpenAI 兼容的等价物。
LLMProtocol = Literal["openai", "anthropic", "google", "vertexai", "azure"]

#: 「要不要思考」这个开关在请求体里的形状。
#:
#: 每个供应商的形状都不一样（llama.cpp 是 ``chat_template_kwargs.enable_thinking``，
#: minimax 是 ``thinking.type`` 但取值是 ``adaptive``），那是**供应商的知识**，
#: 所以写在这里的数据里，而不是让渠道自己去猜请求体。
#:
#: ``thinking_type`` = ``{"thinking": {"type": "enabled" | "disabled"}}``
#: （DeepSeek / MiMo 用这个形状；minimax 因为只认 ``adaptive`` 而单列）。
ThinkingStyle = Literal["none", "minimax", "thinking_type", "chat_template_kwargs"]

#: 结构化输出走哪条路。
#:
#: * ``tool`` —— 把 schema 当工具绑上去，模型调工具，我们校验参数。
#: * ``json_schema`` —— 原生 ``response_format=json_schema``（语法级约束）。
#:   llama.cpp 用 GBNF 硬约束，实测 5/5 解析成功；工具那条路在 0.8B 上经常
#:   「不调工具、直接写散文」。
StructuredOutputStyle = Literal["tool", "json_schema"]

#: 单次生成的 token 上限策略。
#:
#: ``llamacpp`` 特殊：llama.cpp 的 ``n_predict`` 默认 -1 = 不限长，而开着思考链的
#: 模型会一直续写思考 —— 实测不限长时请求直接挂死。所以它必须**有界**，而且
#: 思考开/关要两档预算。
MaxTokensPolicy = Literal["none", "llamacpp"]

#: 「思考深度」参数的位置。
#:
#: * ``output_config`` —— MiniMax 的 ``output_config.effort``。
#: * ``reasoning_effort`` —— DeepSeek（Chat Completions）的顶层 ``reasoning_effort``。
EffortStyle = Literal["none", "output_config", "reasoning_effort"]

#: OpenAI 兼容协议下的**客户端变体**。
#:
#: 同一个协议，不同供应商对「结构化输出」「工具 schema」「思考链」的处理不一样，
#: 所以客户端分几个变体。选哪个是**供应商的知识** → 写在目录里。
#:
#: * ``surgery`` —— 通用 OpenAI 兼容端点。带工具 schema 结构手术（删 title、
#:   折 anyOf），实测省 10.7% token。
#: * ``minimax`` —— MiniMax 专用：``with_structured_output`` 走「schema 当工具绑」
#:   那条路（MiniMax 不认 ``response_format``）。
#: * ``thinking_aware`` —— **思考开着就不能强制 tool_choice** 的那类端点
#:   （DeepSeek / MiMo）。结构化输出走「schema 当工具绑」，且每次请求按
#:   「这次思考开着没有」决定要不要强制 tool_choice。
#: * ``siliconflow`` —— 硅基流动：结构化输出先走原生 json_schema，失败再降级到工具。
#: * ``llamacpp`` —— 本地 llama-server：json_schema 是 GBNF 语法级约束，
#:   且思考链默认开着会把 token 预算烧光。
ClientKind = Literal["surgery", "minimax", "thinking_aware", "siliconflow", "llamacpp"]


@dataclass(frozen=True)
class ChannelSpec:
    """一条渠道 = 一个供应商的接入方式。

    这是一个**纯数据**对象：不 import 任何客户端 SDK，不含行为。这样它可以被
    导出成 JSON 给 manager 侧（添加渠道的表单要靠它给出「选哪个渠道 / 选哪个
    模型 / 默认地址是什么」）。
    """

    #: 渠道 id。它就是以前那个 ``provider`` 字段的值 —— 字段名没改（数据库、
    #: API、runner 载荷都靠它），改的是「这个值的取值从哪来、附带什么信息」。
    id: str

    #: 界面上显示的名字。
    label: str

    #: 客户端家族（见 :data:`LLMProtocol`）。
    protocol: LLMProtocol

    #: 默认访问端点。None = 用该协议客户端自带的默认地址。
    base_url: str | None = None

    #: 常用模型名。空 = 不给推荐列表（模型路由平台，或用户自己知道型号）。
    models: tuple[str, ...] = ()

    #: 密钥在 ``.env`` 里的变量名。老路（配置文件 + .env）靠它兜底。
    api_key_env: str | None = None

    #: 端点在 ``.env`` 里的变量名。老路靠它兜底。
    base_url_env: str | None = None

    #: 这条渠道是不是必须有一把密钥。
    #:
    #: 判据只有一条：能不能用服务账号凭据代替（azure / vertexai 能），
    #: 或者端点根本不鉴权（llamacpp → 本地 llama-server 默认不校验 Authorization 头）。
    requires_api_key: bool = True

    #: 客户端接不接受自定义端点。google / vertexai 的 SDK 没有这个旋钮。
    supports_endpoint: bool = True

    #: 思考开关在请求体里的形状。``none`` = 这个渠道没有思考开关。
    thinking_style: ThinkingStyle = "none"

    #: 这个渠道的模型**默认**是不是开着思考。
    #:
    #: DeepSeek 的推理模型默认开思考（CoT 在 ``reasoning_content``），MiniMax 的
    #: M3 默认开。开着思考时**不能**强制指定 tool_choice（DeepSeek 直接 400）。
    thinking_default_on: bool = False

    #: 「要不要思考」这个开关在 ``.env`` 里的变量名。
    #:
    #: 老路（配置文件 + .env）靠它兜底；渠道下发的值优先于它（见
    #: :meth:`~minitap.mobile_use.services.llm.ProviderOverride.thinking_or`）。
    thinking_env: str | None = None

    #: 没人指定时，我们**发什么**。
    #:
    #: None = 不发这个字段（走供应商自己的默认）。
    #: llama.cpp 是 False —— 它的思考链默认开着，而开着时 ``content`` 恒为空，
    #: 结构化输出必然失败（实测：思考 ON + max_tokens=256 → JSON 没写完；
    #: 不限长 → 直接挂死）。所以它必须**显式**发一个关。
    thinking_fallback: bool | None = None

    #: 请求里带上「本次运行 id」的查询参数名。
    #:
    #: minitap 用它把一次运行里的多次调用串起来（``?sessionId=…``）。
    session_query_param: str | None = None

    #: 「能不能强制指定 tool_choice」**不是一个独立开关** —— 它由两样东西共同决定：
    #: ``client_kind``（客户端会不会问服务端这个问题）和 ``thinking_default_on``
    #: （没说思考开关时按开着还是关着算）。所以这里**没有**这个字段：留一个
    #: 没人读的字段，目录和代码就会对不上 —— 而那正是这套改造要消灭的东西。

    #: 思考深度参数的位置。
    effort_style: EffortStyle = "none"

    #: 渠道支持哪些思考强度档位。空 = 这条渠道没有这个旋钮。
    #:
    #: 值来自官方 ``GET /models`` 的 ``effort.supported_levels``（顺序也照它）。
    #: 它只用于**校验和界面选项**：真正决定发不发、发哪个的是渠道下发的
    #: ``reasoning_effort``（见 ``services/llm.py`` 的 ``_resolve_reasoning_effort``）。
    effort_levels: tuple[str, ...] = ()

    #: 没指定强度时服务端用的默认档位（官方 ``effort.default_level``）。
    #:
    #: 只用于界面提示（「默认」到底是多少）；**不**表示我们会显式发它 ——
    #: 没指定时我们什么都不发，把默认留给服务端，和 ``thinking`` 的语义一致。
    effort_default: str | None = None

    #: 温度的合法区间。超了要钳（MiniMax 只收 0..1）。
    temperature_range: tuple[float, float] | None = None

    #: 结构化输出走哪条路。
    structured_output: StructuredOutputStyle = "tool"

    #: token 上限策略。
    max_tokens_policy: MaxTokensPolicy = "none"

    #: OpenAI 兼容协议下的客户端变体（见 :data:`ClientKind`）。
    #: 非 OpenAI 协议用不到，填 ``surgery`` 占位。
    client_kind: ClientKind = "surgery"

    #: 这条渠道是不是**必须**由使用者自己给访问端点。
    #:
    #: 「自定义（OpenAI 兼容）」和 Azure 是：它们没有默认地址，不给端点的话
    #: 请求会打到别处 —— 「密钥是我自己那家的，怎么报的是 OpenAI 的错」。
    requires_base_url: bool = False

    #: 给**使用者**看的提示（界面会显示在这条渠道下面）。
    #:
    #: 只放「用户必须知道、否则会踩坑」的事：服务条款限制、密钥过期规则、
    #: 套餐额度怎么算。**不放**运行时旋钮（那些是 mobile-use 自己发请求时用的
    #: 知识，导出去只会多一份会过期的副本）。
    note: str | None = None

    #: 要不要开 ``stream_usage``。
    #:
    #: 不开的话流式调用的 ``usage_metadata`` 是 None，遥测里 token 消耗记 0、
    #: 缓存命中率无从计算（看起来像「这次调用免费」）。
    stream_usage: bool = False

    #: 单次请求超时（秒）。None = 用客户端默认（OpenAI SDK 默认 600s 且重试
    #: 两次 —— 一次挂掉的连接能把 agent 拖住 30 分钟，只打印一行
    #: "Waiting for LLM call response..."）。
    request_timeout: float | None = None

    #: 失败重试次数。
    max_retries: int | None = None

    #: 「模型名前缀 → 这个前缀用的重试次数」。
    #:
    #: minitap 上的 ``google/`` 系模型要多试一次（默认 2），其余照默认。
    retry_rules: tuple[tuple[str, int], ...] = ()

    def default_url(self) -> str | None:
        """这条渠道的默认端点。"""
        return self.base_url


# ---------------------------------------------------------------------------
# 目录本体
# ---------------------------------------------------------------------------

CHANNELS: dict[str, ChannelSpec] = {
    # ---- OpenAI 兼容协议 ----------------------------------------------------
    "openai": ChannelSpec(
        id="openai",
        label="OpenAI",
        protocol="openai",
        base_url="https://api.openai.com/v1",
        models=("gpt-5", "gpt-5-mini", "gpt-5-nano", "o4-mini"),
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        request_timeout=180.0,
        max_retries=1,
        stream_usage=True,
    ),
    "deepseek": ChannelSpec(
        id="deepseek",
        label="DeepSeek（深度求索）",
        protocol="openai",
        # 实测：国内站 https://api.deepseek.com，/v1 也能用但默认写不带 /v1 的。
        base_url="https://api.deepseek.com",
        # 实测（GET /models，2026-10-24）：deepseek-flash / deepseek-v4-pro。
        models=("deepseek-flash", "deepseek-v4-pro"),
        api_key_env="DEEPSEEK_API_KEY",
        base_url_env="DEEPSEEK_BASE_URL",
        # ⚠️ 这两个开关是 DeepSeek 能跑起来的关键：
        # 推理模型**默认开思考**，而思考模式下强制 tool_choice 是硬 400
        # （"Thinking mode does not support this tool_choice"）——
        # 2026-10-24 实测：deepseek-flash + 强制 tool_choice = 400，
        # 加上 {"thinking": {"type": "disabled"}} 后 = 200。
        thinking_style="thinking_type",
        thinking_default_on=True,
        thinking_env="DEEPSEEK_THINKING",
        # 思考强度（Chat Completions 的顶层 ``reasoning_effort``）。
        # 官方 GET /models（2026-10-26）两个模型都是 low/high/max、默认 high；
        # 文档另外注明 low / medium 目前会映射为 high、xhigh 映射为 max ——
        # 所以实际有区别的只有 high 与 max，界面要把这句说清楚。
        effort_style="reasoning_effort",
        effort_levels=("low", "high", "max"),
        effort_default="high",
        # 和 MiMo 同属「思考感知」那一类（见 ClientKind）—— DeepSeek 只是
        # 「默认开思考」，MiMo 是「按需开启」，差别在 thinking_default_on 里。
        client_kind="thinking_aware",
        request_timeout=180.0,
        max_retries=1,
    ),
    "minimax": ChannelSpec(
        id="minimax",
        label="MiniMax",
        protocol="openai",
        # 国内站。国际站是 https://api.minimax.io/v1 —— 两家的 key **不通用**，
        # 国内 key 发到国际站是 401「invalid api key (2049)」（2026-10-24 实测）。
        base_url="https://api.minimaxi.com/v1",
        # 实测（GET /v1/models，2026-10-24）。
        models=(
            "MiniMax-M3.1-Flash",
            "MiniMax-M3",
            "MiniMax-M2.7",
            "MiniMax-M2.7-highspeed",
            "MiniMax-M2.5",
            "MiniMax-M2.5-highspeed",
            "MiniMax-M2.1",
            "MiniMax-M2.1-highspeed",
            "MiniMax-M2",
        ),
        api_key_env="MINIMAX_API_KEY",
        base_url_env="MINIMAX_BASE_URL",
        thinking_style="minimax",
        thinking_default_on=True,
        thinking_env="MINIMAX_THINKING",
        effort_style="output_config",
        # MiniMax 只收 0..1（超了服务端 400）。
        temperature_range=(0.0, 1.0),
        client_kind="minimax",
        request_timeout=180.0,
        max_retries=1,
        stream_usage=True,
    ),
    "siliconflow": ChannelSpec(
        id="siliconflow",
        label="硅基流动 SiliconFlow",
        protocol="openai",
        base_url="https://api.siliconflow.cn/v1",
        # 实测（GET /v1/models，2026-10-24，全量 98 个）。这里挑的是
        # 「仓库配置里已经在用」+「能力对得上」的：
        #   - Qwen2.5-72B / 7B-Instruct：文本模型，工具调用 + json_schema 都行
        #   - Qwen3-VL-32B / 8B-Instruct：视觉 + json_schema 行，**不支持强制 tool_choice**
        #   - zai-org/GLM-4.5V：视觉 + 工具调用行，**不支持 json_schema**
        # ⚠️ 视觉模型和文本模型不能混用：cortex 要 json_schema，executor 要工具调用，
        # 指到同一个模型上必然有一边瞎掉。
        models=(
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen3-VL-32B-Instruct",
            "Qwen/Qwen3-VL-8B-Instruct",
            "zai-org/GLM-4.5V",
            "zai-org/GLM-5.3",
            "deepseek-ai/DeepSeek-V4-Pro",
            "deepseek-ai/DeepSeek-V4.1-Flash",
            "moonshotai/Kimi-K2.7-Code",
        ),
        api_key_env="SILICONFLOW_API_KEY",
        base_url_env="SILICONFLOW_BASE_URL",
        client_kind="siliconflow",
        request_timeout=180.0,
        max_retries=1,
        stream_usage=True,
    ),
    "openrouter": ChannelSpec(
        id="openrouter",
        label="OpenRouter",
        protocol="openai",
        base_url="https://openrouter.ai/api/v1",
        # 模型路由平台：模型列表由平台自己变，写死一份反而会过期 → 不给推荐。
        api_key_env="OPEN_ROUTER_API_KEY",
        base_url_env="OPEN_ROUTER_BASE_URL",
    ),
    "xai": ChannelSpec(
        id="xai",
        label="xAI（Grok）",
        protocol="openai",
        base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
        base_url_env="XAI_BASE_URL",
    ),
    "minitap": ChannelSpec(
        id="minitap",
        label="Minitap",
        protocol="openai",
        base_url="https://platform.minitap.ai/api/v1",
        # 仓库的 recommended preset 里在用的那些。
        models=(
            "google/gemini-3-pro-preview",
            "google/gemini-2.5-pro",
            "meta-llama/llama-4-scout",
            "meta-llama/llama-4-maverick",
            "meta-llama/llama-3.1-70b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
            "meta-llama/llama-3.3-70b-instruct",
            "openai/gpt-5",
            "openai/gpt-5-mini",
            "openai/gpt-5-nano",
        ),
        api_key_env="MINITAP_API_KEY",
        base_url_env="MINITAP_API_BASE_URL",
        # minitap 用 sessionId 把一次运行里的多次调用串起来。
        session_query_param="sessionId",
        # google/ 系模型要多试一次（以前的工厂就是这么写的）。
        retry_rules=(("google/", 2),),
    ),
    "llamacpp": ChannelSpec(
        id="llamacpp",
        label="llama.cpp（本地）",
        protocol="openai",
        base_url="http://127.0.0.1:9931/v1",
        models=("unsloth/Qwen3.5-0.8B-GGUF:Q6_K",),
        api_key_env="LLAMACPP_API_KEY",
        base_url_env="LLAMACPP_BASE_URL",
        # 本地 llama-server 默认不鉴权 → 没密钥也合法。
        requires_api_key=False,
        # 关思考是**必须**的：开着时 content 恒为空，结构化输出必然失败。
        thinking_style="chat_template_kwargs",
        thinking_default_on=True,
        thinking_env="LLAMACPP_ENABLE_THINKING",
        # 关思考是**必须**的：开着时 content 恒为空，结构化输出必然失败。
        thinking_fallback=False,
        # json_schema 是语法级约束（GBNF），实测 5/5 解析成功。
        structured_output="json_schema",
        max_tokens_policy="llamacpp",
        client_kind="llamacpp",
        stream_usage=True,
        request_timeout=180.0,
        max_retries=1,
    ),
    # ---- 小米 MiMo ------------------------------------------------------------
    "mimo-token-plan": ChannelSpec(
        id="mimo-token-plan",
        label="MiMo Token Plan（小米订阅套餐）",
        protocol="openai",
        # 中国集群。其它集群：新加坡 token-plan-sgp / 欧洲 token-plan-ams
        # （同一后缀，见小米官方文档「Token Plan → 快速接入」）。
        base_url="https://token-plan-cn.xiaomimimo.com/v1",
        # Token Plan 个人版覆盖的 6 款模型里能对话的是这两款（另有 asr / tts
        # 三款是语音，不在本目录里）。官方文档（2026-10-24）：
        # 「各套餐均支持 mimo-v2.6-pro、mimo-v2.6-flash …」
        # ⚠️ mimo-v2.5-pro / mimo-v2.5 已于 2026-10-21 下线，不列。
        models=("mimo-v2.6-pro", "mimo-v2.6-flash"),
        # 小米文档里这个变量就叫 MIMO_API_KEY。⚠️ 它和「按量付费」那把
        # （sk- 开头、https://api.xiaomimimo.com/v1）**相互独立、不可混用** ——
        # 两边要加两个渠道时，环境变量必须分开。
        api_key_env="MIMO_API_KEY",
        base_url_env=None,
        # 思考开关的形状是 ``thinking.type``（小米给 Kilo Code 的示例里就是
        # ``"thinking": {"type": "enabled"}``）。
        thinking_style="thinking_type",
        # 🔴 **服务端默认是开着思考的**（2026-10-24 实测，token-plan-cn，
        # mimo-v2.6-pro）：不下发 ``thinking`` 时 usage 里 ``reasoning_tokens``
        # 是 13~17、message 里有 ``reasoning_content``。
        #
        # 以前这里写的是 False（依据是「MiMo 的思考是按需开启的」）—— 与实测
        # 相反。后果：框架以为「没说 = 没开思考」⇒ 允许强制 tool_choice，
        # 而实际思考是开着的 ⇒ 小米文档警告的那件事就发生了：
        # 「调 tool 的时候开启 thinking 导致输出不完整」。
        #
        # 实测症状（2026-10-24）：max_completion_tokens=64 时推理链把预算吃光
        # ⇒ finish_reason=length、**没有 tool_calls**、静默失败（不是 DeepSeek
        # 那种硬 400）。预算放大到 2048 后 6 种组合全部 3/3 命中工具调用 ——
        # 所以它不是「必然坏」，是「预算一紧就坏」。
        #
        # ⇒ 这一项必须如实地反映**服务端默认**：True。
        thinking_default_on=True,
        # 思考开关的旋钮。没配 MIMO_THINKING 时不下发这个字段，走服务端默认
        # （= 开着思考）。配 false 则显式发 {"thinking": {"type": "disabled"}}：
        # reasoning_tokens 归 0、completion_tokens 从 ~35 降到 17，工具调用稳定。
        thinking_env="MIMO_THINKING",
        client_kind="thinking_aware",
        request_timeout=180.0,
        max_retries=1,
        stream_usage=True,
        note=(
            "Token Plan 的 key（tp- 开头）按小米条款**只允许用于编程工具**"
            "（Claude Code / OpenCode / OpenClaw 等），「禁止以 API 调用的形式用于"
            "自动化脚本、自定义应用程序后端等明显非 Coding 场景」—— 平台可暂停服务、"
            "封禁 key。把它用在设备自动化这类后端任务上有违约风险。"
            "要长期跑自动化，请用按量付费那把 sk- key（地址 https://api.xiaomimimo.com/v1）。"
        ),
    ),
    # ---- Anthropic 协议 -----------------------------------------------------
    "anthropic": ChannelSpec(
        id="anthropic",
        label="Anthropic（Claude）",
        protocol="anthropic",
        # ChatAnthropic 的端点字段是 ``anthropic_api_url``，默认就是这个。
        base_url="https://api.anthropic.com",
        models=("claude-sonnet-4-6", "claude-haiku-4-5-20251001"),
        api_key_env="ANTHROPIC_API_KEY",
        request_timeout=180.0,
        max_retries=2,
    ),
    # ---- 自定义：任何 OpenAI 兼容的第三方服务 --------------------------------
    "custom": ChannelSpec(
        id="custom",
        label="自定义（OpenAI 兼容）",
        protocol="openai",
        # 没有默认地址：必须自己填。填错的话请求会打到别处，而错误信息完全
        # 指不到真正的原因。
        base_url=None,
        requires_base_url=True,
        api_key_env=None,
        base_url_env=None,
    ),
    # ---- 保留的原生通道（SDK 没有 OpenAI 兼容等价物）------------------------
    "google": ChannelSpec(
        id="google",
        label="Google Gemini",
        protocol="google",
        base_url=None,  # SDK 没有「访问端点」这个旋钮
        models=("gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-2.5-pro"),
        api_key_env="GOOGLE_API_KEY",
        supports_endpoint=False,
        max_retries=2,
    ),
    "vertexai": ChannelSpec(
        id="vertexai",
        label="Google Vertex AI",
        protocol="vertexai",
        base_url=None,
        models=("gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-2.5-pro"),
        # 走服务账号凭据（ADC），没有「API Key」这个概念。
        requires_api_key=False,
        supports_endpoint=False,
        max_retries=2,
    ),
    "azure": ChannelSpec(
        id="azure",
        label="Azure OpenAI",
        protocol="azure",
        base_url=None,  # 必填，但由调用方给（部署自己的端点）
        api_key_env="AZURE_API_KEY",
        base_url_env="AZURE_BASE_URL",
        # azure 没有默认端点：每个部署的 URL 都不一样，必须自己给。
        requires_base_url=True,
        # azure 可以完全不带 key（走 DefaultAzureCredential）。
        requires_api_key=False,
        max_retries=2,
    ),
}


# ---------------------------------------------------------------------------
# 查询helper
# ---------------------------------------------------------------------------


def channel_ids() -> tuple[str, ...]:
    """所有渠道 id，顺序即目录里的顺序。"""
    return tuple(CHANNELS.keys())


def get_channel(channel_id: str | None) -> ChannelSpec | None:
    """按 id 取渠道。认不出来返回 None（**不抛** —— 调用点常要区分「不认识」）。"""
    if not channel_id:
        return None
    return CHANNELS.get(channel_id.strip().lower())


def require_channel(channel_id: str) -> ChannelSpec:
    """按 id 取渠道，不认识就报错 —— 报错里带上有哪些渠道可选。

    这条路径用于「渠道 id 写错了」：写错一个字母要到建 agent 时才炸，那时候人
    已经跑起来了。在这里挡住，并把可选清单给出来。
    """
    spec = get_channel(channel_id)
    if spec is None:
        raise ValueError(
            f"unknown LLM channel {channel_id!r}; known channels are: " + ", ".join(channel_ids())
        )
    return spec


def channel_requires_api_key(channel_id: str) -> bool:
    """这条渠道是不是必须有一把密钥。

    判据只有一条：能不能用服务账号凭据代替，或者端点根本不鉴权。
    """
    spec = get_channel(channel_id)
    if spec is None:
        # 不认识的渠道按「要密钥」处理 —— 宁可多要一把（报错），也别少要
        # （跑到一半才 401）。
        return True
    return spec.requires_api_key


def channel_supports_endpoint(channel_id: str) -> bool:
    """这条渠道的客户端接不接受自定义端点。"""
    spec = get_channel(channel_id)
    if spec is None:
        return True
    return spec.supports_endpoint


def default_base_url(channel_id: str) -> str | None:
    """这条渠道的默认端点。"""
    spec = get_channel(channel_id)
    return spec.base_url if spec else None


def protocol_of(channel_id: str) -> LLMProtocol:
    """这条渠道用哪个协议客户端。

    不认识的渠道按 ``openai`` 处理：绝大多数第三方服务都是 OpenAI 兼容的，
    而且用户自己填了端点 —— 按 openai 协议发才对。
    """
    spec = get_channel(channel_id)
    return spec.protocol if spec else "openai"


def catalog_payload() -> list[dict[str, Any]]:
    """导出成给 manager 侧的 JSON 载荷。

    只带**界面需要**的字段：选渠道、选模型、默认地址是什么、要不要密钥，
    以及「思考强度能选哪几档」。

    思考开关的**请求体形状**、结构化输出策略这类运行时知识**不导出** ——
    manager 侧用不到，导出只会多一份会过期的副本。但强度档位是表单要渲染的
    选项，所以和模型列表一样属于「界面需要」的那部分。
    """
    return [
        {
            "id": spec.id,
            "label": spec.label,
            "protocol": spec.protocol,
            "baseUrl": spec.base_url,
            "models": list(spec.models),
            "requiresApiKey": spec.requires_api_key,
            "supportsEndpoint": spec.supports_endpoint,
            "thinkingDefaultOn": spec.thinking_default_on,
            "reasoningEffortLevels": list(spec.effort_levels),
            "reasoningEffortDefault": spec.effort_default,
            "note": spec.note,
        }
        for spec in CHANNELS.values()
    ]
