"""Tests for the MiniMax LLM provider."""

import logging
import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from minitap.mobile_use.services.llm import (
    minimax_model_family,
    resolve_minimax_thinking,
)


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("MiniMax-M3", "m3"),
        ("MiniMax-M3.1-Flash", "m3.1"),
        ("MiniMax-M3.1-Flash-Preview", "m3.1"),
        ("MiniMax-M2.7", "m2.7"),
        ("MiniMax-M2.7-highspeed", "m2.7"),
        ("MiniMax-M2", "m2"),
        ("minimax-m3", "m3"),
        ("", ""),
    ],
)
def test_minimax_model_family(model_name, expected):
    """家族号决定「能不能关思考」，所以取家族号的规则不能含糊。"""
    assert minimax_model_family(model_name) == expected


@pytest.mark.parametrize(
    ("model_name", "requested", "expected"),
    [
        # 没指定 = 不发这个字段 = 走服务端默认。
        ("MiniMax-M3", None, None),
        ("MiniMax-M3.1-Flash", None, None),
        # 开：adaptive 是三个家族都认的写法（M3.1 只认 adaptive）。
        ("MiniMax-M3", True, {"type": "adaptive"}),
        ("MiniMax-M3.1-Flash", True, {"type": "adaptive"}),
        ("MiniMax-M2.7", True, {"type": "adaptive"}),
        # 关：只有 M3 真能关。
        ("MiniMax-M3", False, {"type": "disabled"}),
        # 关不掉的三种：M3.1（发了 400）、M2.x（发了无效）、未知家族（不敢发）。
        ("MiniMax-M3.1-Flash", False, None),
        ("MiniMax-M2.7", False, None),
        ("MiniMax-M9-Ultra", False, None),
    ],
)
def test_resolve_minimax_thinking(model_name, requested, expected):
    """能不能关思考**看模型家族** —— 发 disabled 到 M3.1 上是硬 400。"""
    assert resolve_minimax_thinking(model_name, requested) == expected


def test_resolve_minimax_thinking_warns_when_it_cannot_turn_thinking_off(caplog):
    """关不掉的时候要**报出来**：静默忽略会变成「我明明拨了开关，它还在思考」。"""
    with caplog.at_level(logging.WARNING, logger="minitap.mobile_use.services.llm"):
        result = resolve_minimax_thinking("MiniMax-M3.1-Flash", False)

    assert result is None
    assert any("关不掉" in record.message for record in caplog.records)


def test_resolve_minimax_thinking_does_not_warn_when_turning_thinking_on(caplog):
    """「开」对所有家族都成立，不该报噪音。"""
    with caplog.at_level(logging.WARNING, logger="minitap.mobile_use.services.llm"):
        resolve_minimax_thinking("MiniMax-M3.1-Flash", True)

    assert not [r for r in caplog.records if "关不掉" in r.message]


# 打桩要打在**工厂真正构造的那个类**上：get_minimax_llm 建的是
# ``MiniMaxChatOpenAI``（``ChatOpenAI`` 的子类），不是 ``ChatOpenAI`` 本身。
# 子类在 import 期就已经把基类绑死了，之后再 patch ``llm.ChatOpenAI``
# 拦不住已经建好的子类 —— 症状是「patch 生效了但 call_args 是 None」。
@patch("minitap.mobile_use.services.llm.MiniMaxChatOpenAI")
@patch("minitap.mobile_use.services.llm.settings")
def test_get_minimax_llm_sends_thinking_switch(mock_settings, mock_chat_openai):
    """渠道的 thinking 开关必须真的进请求体 —— 不然它只是个界面上的摆设。"""
    from minitap.mobile_use.services.llm import ProviderOverride, get_minimax_llm

    mock_settings.MINIMAX_API_KEY = SecretStr("test-key")

    get_minimax_llm(model_name="MiniMax-M3", override=ProviderOverride(thinking=False))
    assert mock_chat_openai.call_args.kwargs["extra_body"]["thinking"] == {"type": "disabled"}

    get_minimax_llm(model_name="MiniMax-M3", override=ProviderOverride(thinking=True))
    assert mock_chat_openai.call_args.kwargs["extra_body"]["thinking"] == {"type": "adaptive"}

    # 没指定 = 不发这个字段（走服务端默认）。
    get_minimax_llm(model_name="MiniMax-M3", override=ProviderOverride(thinking=None))
    assert "thinking" not in (mock_chat_openai.call_args.kwargs["extra_body"] or {})


@patch("minitap.mobile_use.services.llm.MiniMaxChatOpenAI")
@patch("minitap.mobile_use.services.llm.settings")
def test_get_minimax_llm_does_not_send_thinking_to_m31(mock_settings, mock_chat_openai):
    """M3.1 发 disabled 是硬 400（每个请求都 400，agent 直接死）—— 不能发。"""
    from minitap.mobile_use.services.llm import ProviderOverride, get_minimax_llm

    mock_settings.MINIMAX_API_KEY = SecretStr("test-key")

    get_minimax_llm(model_name="MiniMax-M3.1-Flash", override=ProviderOverride(thinking=False))
    extra_body = mock_chat_openai.call_args.kwargs["extra_body"] or {}
    assert "thinking" not in extra_body


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(False, {"type": "disabled"}), (True, {"type": "adaptive"}), (None, None)],
)
@patch("minitap.mobile_use.services.llm.MiniMaxChatOpenAI")
@patch("minitap.mobile_use.services.llm.settings")
def test_get_minimax_llm_falls_back_to_settings(mock_settings, mock_chat_openai, requested, expected):
    """渠道没下发时用 settings.MINIMAX_THINKING（老路：.env 里那个旋钮）。"""
    from minitap.mobile_use.services.llm import ProviderOverride, get_minimax_llm

    mock_settings.MINIMAX_API_KEY = SecretStr("test-key")
    mock_settings.MINIMAX_THINKING = requested

    get_minimax_llm(model_name="MiniMax-M3", override=ProviderOverride(thinking=None))
    extra_body = mock_chat_openai.call_args.kwargs["extra_body"] or {}
    if expected is None:
        assert "thinking" not in extra_body
    else:
        assert extra_body["thinking"] == expected


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(-0.5, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (1.5, 1.0)],
)
@patch("minitap.mobile_use.services.llm.MiniMaxChatOpenAI")
@patch("minitap.mobile_use.services.llm.settings")
def test_get_minimax_llm_clamps_temperature(mock_settings, mock_chat_openai, requested, expected):
    """MiniMax rejects temperatures outside (0, 1]; the provider must clamp them."""
    from minitap.mobile_use.services.llm import get_minimax_llm

    mock_settings.MINIMAX_API_KEY = SecretStr("test-key")
    get_minimax_llm(temperature=requested)
    assert mock_chat_openai.call_args.kwargs["temperature"] == expected


@pytest.mark.integration
@patch("minitap.mobile_use.services.llm.settings")
def test_minimax_m27_llm_invoke(mock_settings):
    from minitap.mobile_use.services.llm import get_minimax_llm

    key = os.environ.get("MINIMAX_API_KEY")
    if not key or key in {"...", "your-minimax-api-key"}:
        pytest.skip("MINIMAX_API_KEY not set, skipping integration test")

    mock_settings.MINIMAX_API_KEY = SecretStr(key)
    # settings 被打桩后，**所有**属性都变成 MagicMock。端点和 effort 必须给
    # 真实值：base_url 会进 pydantic 校验（MagicMock 不是字符串 → 直接报错），
    # effort 则会被当成未知取值。
    #
    # base_url 取**应用自己配置的那个端点**（国内 api.minimaxi.com / 国际
    # api.minimax.io），不能写死 —— key 和端点是配对的，写死成另一个端点
    # 就是拿国内 key 打国际站，直接 401。
    from minitap.mobile_use.config import settings as real_settings

    mock_settings.MINIMAX_BASE_URL = real_settings.MINIMAX_BASE_URL
    mock_settings.MINIMAX_EFFORT = real_settings.MINIMAX_EFFORT or "max"
    client = get_minimax_llm(model_name="MiniMax-M2.7", temperature=0.7)
    response = client.invoke("Say 'hello' and nothing else.")
    assert len(response.content) > 0
