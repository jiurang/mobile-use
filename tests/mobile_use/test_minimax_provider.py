"""Tests for the MiniMax LLM provider."""

import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(-0.5, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (1.5, 1.0)],
)
# 打桩要打在**工厂真正构造的那个类**上：get_minimax_llm 建的是
# ``MiniMaxChatOpenAI``（``ChatOpenAI`` 的子类），不是 ``ChatOpenAI`` 本身。
# 子类在 import 期就已经把基类绑死了，之后再 patch ``llm.ChatOpenAI``
# 拦不住已经建好的子类 —— 症状是「patch 生效了但 call_args 是 None」。
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
