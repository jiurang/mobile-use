"""端点归一化：base_url 是**基址**，不是端点。

这条规则是被真机的 404 逼出来的，不是设计出来的。

**症状**（2026-10-26，Redmi Turbo 4，manager 执行 885）：在 manager-web 上把
SCNet 的渠道填成 ``https://api.scnet.cn/api/llm/v1/embeddings``（照着供应商
文档粘的完整端点），任务一跑，每次调用都是::

    Error code: 404 - {'path': '/acx/llm/v1/embeddings/chat/completions',
                        'status': 404, 'error': 'Not Found'}

**病因**：OpenAI 兼容客户端会在 base_url 后面**自己拼** ``/chat/completions``，
而用户粘进来的是**已经带端点**的地址 ⇒ 拼成了 ``.../embeddings/chat/completions``。

**为什么这个错特别难查**：地址看着像模像样（域名对、路径是供应商给的），服务端
只回一句 ``Not Found``，没有任何一处提示「你的 base_url 多带了一段路径」。
于是人会去怀疑模型、怀疑 key、怀疑网络 —— 而真正的原因在**填表那一步**。

⇒ 规则：粘进来的地址如果已经带着客户端会拼的那段路径，就把它剥掉，
**两种填法都要能用**：基址（``.../v1``）和完整端点（``.../v1/chat/completions``）。
"""

from __future__ import annotations

from types import SimpleNamespace

from minitap.mobile_use.sdk.utils import llm_config_for_single_model
from minitap.mobile_use.services.channels import normalize_base_url
from minitap.mobile_use.services.llm import get_llm


def _ctx(llm_config):
    """get_llm 只读 ctx.llm_config 和 ctx.trace_id。"""
    return SimpleNamespace(llm_config=llm_config, trace_id="t-1")


# ---------------------------------------------------------------------------
# 1. OpenAI 兼容协议（绝大多数第三方服务走这一条）
# ---------------------------------------------------------------------------


def test_full_endpoint_is_reduced_to_the_base():
    """完整端点 → 基址。执行 885 的那个地址。"""
    assert (
        normalize_base_url("https://api.scnet.cn/api/llm/v1/embeddings", "openai")
        == "https://api.scnet.cn/api/llm/v1"
    )


def test_chat_completions_suffix_is_stripped():
    assert (
        normalize_base_url("https://x.example.com/v1/chat/completions", "openai")
        == "https://x.example.com/v1"
    )


def test_plain_base_is_left_alone():
    """基址本来就对，不能被乱改 —— 尤其 ``/v1`` 必须留着。"""
    assert normalize_base_url("https://api.openai.com/v1", "openai") == "https://api.openai.com/v1"
    assert (
        normalize_base_url("https://api.scnet.cn/api/llm/v1", "openai")
        == "https://api.scnet.cn/api/llm/v1"
    )


def test_trailing_slash_is_removed():
    assert normalize_base_url("https://x.example.com/v1/", "openai") == "https://x.example.com/v1"


def test_none_and_blank_stay_empty():
    assert normalize_base_url(None, "openai") is None
    assert normalize_base_url("   ", "openai") is None


def test_lookalike_path_is_not_stripped():
    """只是**结尾**像端点才剥。中间带 chat/completions 的路径不能动。

    ``https://x.com/chat/completions-proxy/v1`` 的结尾是 ``/v1``，不受影响；
    而 ``https://x.com/proxy/chat/completions`` 结尾确实是端点 —— 那种服务
    本身就把端点放在路径里，剥掉反而对。
    """
    assert (
        normalize_base_url("https://x.example.com/proxy/v1", "openai")
        == "https://x.example.com/proxy/v1"
    )


# ---------------------------------------------------------------------------
# 2. Anthropic 协议：客户端拼的是 /v1/messages，基址要停在根上
# ---------------------------------------------------------------------------


def test_anthropic_messages_endpoint_is_reduced_to_the_root():
    assert (
        normalize_base_url("https://api.anthropic.com/v1/messages", "anthropic")
        == "https://api.anthropic.com"
    )


def test_anthropic_v1_is_reduced_to_the_root():
    """留着 /v1 会拼成 /v1/v1/messages。"""
    assert normalize_base_url("https://api.anthropic.com/v1", "anthropic") == "https://api.anthropic.com"


def test_anthropic_root_is_left_alone():
    assert (
        normalize_base_url("https://api.anthropic.com", "anthropic")
        == "https://api.anthropic.com"
    )


def test_protocols_without_endpoint_knob_are_never_stripped():
    """google / vertexai 没有端点旋钮；azure 的 endpoint 是部署根地址。"""
    assert normalize_base_url("https://x.example.com/v1/embeddings", "google") == (
        "https://x.example.com/v1/embeddings"
    )
    assert normalize_base_url("https://x.example.com/v1/embeddings", "azure") == (
        "https://x.example.com/v1/embeddings"
    )


# ---------------------------------------------------------------------------
# 3. 落到客户端上：拼出来的请求地址必须是对的
# ---------------------------------------------------------------------------


def test_client_base_url_is_normalized():
    """渠道下发的完整端点 → 客户端拿到的是基址。

    这条是整个修复的落点：客户端的 ``openai_api_base`` 决定了它最终请求
    ``{base}/chat/completions``，基址不对就是 404。
    """
    cfg = llm_config_for_single_model(
        "custom",
        "Qwen3-Embedding-8B",
        api_key="sk-test",
        base_url="https://api.scnet.cn/api/llm/v1/embeddings",
    )
    client = get_llm(_ctx(cfg), name="cortex")

    assert client.openai_api_base == "https://api.scnet.cn/api/llm/v1"
    # 拼上客户端自己会拼的那段，就是供应商文档里的端点。
    assert client.openai_api_base + "/chat/completions" == (
        "https://api.scnet.cn/api/llm/v1/chat/completions"
    )


def test_client_base_url_keeps_a_plain_base_unchanged():
    cfg = llm_config_for_single_model(
        "custom",
        "MiniMax-M3.1-Flash",
        api_key="sk-test",
        base_url="https://api.scnet.cn/api/llm/v1",
    )
    client = get_llm(_ctx(cfg), name="cortex")

    assert client.openai_api_base == "https://api.scnet.cn/api/llm/v1"


def test_catalog_default_endpoints_are_not_broken():
    """目录里的默认端点走一遍归一化，必须一个都不变。

    归一化是在客户端构造时无条件跑的 —— 如果它把某个渠道的默认地址改坏了，
    那就是「修一个渠道、弄坏另外七个」。
    """
    from minitap.mobile_use.services.channels import CHANNELS

    for spec in CHANNELS.values():
        if spec.base_url is None:
            continue
        assert normalize_base_url(spec.base_url, spec.protocol) == spec.base_url, spec.id
