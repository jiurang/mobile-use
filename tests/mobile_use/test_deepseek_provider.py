"""Tests for the DeepSeek LLM provider."""

import asyncio
import os
from unittest.mock import patch

import pytest
from langchain_openai.chat_models import base as lc_base
from pydantic import SecretStr


def _mock_settings(mock_settings, **overrides):
    """Fill in every setting get_deepseek_llm reads.

    ``mock_settings`` is a MagicMock, so any attribute we don't set is *not*
    ``None`` — that would silently take the DEEPSEEK_THINKING branch. Always
    be explicit here.
    """
    mock_settings.DEEPSEEK_API_KEY = SecretStr("test-key")
    mock_settings.DEEPSEEK_BASE_URL = None
    mock_settings.DEEPSEEK_THINKING = None
    for key, value in overrides.items():
        setattr(mock_settings, key, value)


@patch("minitap.mobile_use.services.llm.ThinkingAwareChatOpenAI")
@patch("minitap.mobile_use.services.llm.settings")
def test_get_deepseek_llm_defaults_to_domestic_endpoint(mock_settings, mock_chat_openai):
    from minitap.mobile_use.services.llm import get_deepseek_llm

    _mock_settings(mock_settings)
    get_deepseek_llm()

    kwargs = mock_chat_openai.call_args.kwargs
    assert kwargs["base_url"] == "https://api.deepseek.com"
    assert kwargs["model"] == "deepseek-flash"
    # Thinking mode is DeepSeek's own default, so we send nothing.
    #
    # 实现改成协议层之后，这个参数**始终出现**（空的时候是 None）——
    # 「不发这个字段」和「发一个 None」在请求体上一回事。
    assert kwargs.get("extra_body") is None


@patch("minitap.mobile_use.services.llm.ThinkingAwareChatOpenAI")
@patch("minitap.mobile_use.services.llm.settings")
def test_get_deepseek_llm_honours_base_url_override(mock_settings, mock_chat_openai):
    from minitap.mobile_use.services.llm import get_deepseek_llm

    _mock_settings(mock_settings, DEEPSEEK_BASE_URL="https://api.deepseek.com/v1")
    get_deepseek_llm()

    assert mock_chat_openai.call_args.kwargs["base_url"] == "https://api.deepseek.com/v1"


@pytest.mark.parametrize(
    ("thinking", "expected"),
    [(None, "<absent>"), (True, "enabled"), (False, "disabled")],
)
@patch("minitap.mobile_use.services.llm.ThinkingAwareChatOpenAI")
@patch("minitap.mobile_use.services.llm.settings")
def test_get_deepseek_llm_thinking_toggle(mock_settings, mock_chat_openai, thinking, expected):
    from minitap.mobile_use.services.llm import get_deepseek_llm

    _mock_settings(mock_settings, DEEPSEEK_THINKING=thinking)
    get_deepseek_llm()

    extra_body = mock_chat_openai.call_args.kwargs.get("extra_body")
    if expected == "<absent>":
        assert extra_body is None
    else:
        assert extra_body == {"thinking": {"type": expected}}


@patch("minitap.mobile_use.services.llm.ThinkingAwareChatOpenAI")
@patch("minitap.mobile_use.services.llm.settings")
def test_get_deepseek_llm_does_not_clamp_temperature(mock_settings, mock_chat_openai):
    """Unlike MiniMax, DeepSeek accepts the whole (0, 2] range — pass it through."""
    from minitap.mobile_use.services.llm import get_deepseek_llm

    _mock_settings(mock_settings)
    get_deepseek_llm(temperature=1.5)

    assert mock_chat_openai.call_args.kwargs["temperature"] == 1.5


@pytest.mark.parametrize("thinking", [None, True, False])
@patch("minitap.mobile_use.services.llm.settings")
def test_deepseek_never_sends_response_format(mock_settings, thinking):
    """Regression guard for the 400 that killed every agent node.

    langchain-openai 1.1.x's ``ChatOpenAI.with_structured_output`` defaults to
    ``method="json_schema"`` (the *BaseChatOpenAI* default is still
    ``"function_calling"``), which sends ``response_format=<PydanticClass>``.
    DeepSeek answers::

        400 invalid_request_error: This response_format type is unavailable now

    So the schema must go out as a ``tools`` entry, never as ``response_format``.
    """
    from minitap.mobile_use.agents.planner.types import PlannerOutput
    from minitap.mobile_use.services.llm import get_deepseek_llm

    _mock_settings(mock_settings, DEEPSEEK_THINKING=thinking)

    captured: dict = {}
    original = lc_base.BaseChatOpenAI._get_request_payload

    class _Stop(Exception):
        pass

    def spy(self, input_, **kwargs):
        captured.update(original(self, input_, **kwargs))
        raise _Stop

    lc_base.BaseChatOpenAI._get_request_payload = spy
    try:
        get_deepseek_llm(model_name="deepseek-flash").with_structured_output(
            PlannerOutput
        ).invoke("hi")
    except _Stop:
        pass
    finally:
        lc_base.BaseChatOpenAI._get_request_payload = original

    assert "response_format" not in captured
    assert [t["function"]["name"] for t in captured["tools"]] == ["PlannerOutput"]


@pytest.mark.parametrize(("thinking", "forced"), [(None, False), (True, False), (False, True)])
@patch("minitap.mobile_use.services.llm.settings")
def test_deepseek_forces_tool_choice_only_when_thinking_off(mock_settings, thinking, forced):
    """Thinking mode 400s on a forced tool_choice; unset/true must stay unforced.

    Live error::

        400 invalid_request_error: Thinking mode does not support this tool_choice
    """
    from minitap.mobile_use.agents.planner.types import PlannerOutput
    from minitap.mobile_use.services.llm import get_deepseek_llm

    _mock_settings(mock_settings, DEEPSEEK_THINKING=thinking)
    runnable = get_deepseek_llm(model_name="deepseek-flash").with_structured_output(PlannerOutput)

    assert ("tool_choice" in runnable._bound.kwargs) is forced


@pytest.mark.integration
@pytest.mark.parametrize("thinking", [None, False])
@patch("minitap.mobile_use.services.llm.settings")
def test_deepseek_structured_output(mock_settings, thinking):
    """The real risk for DeepSeek is structured output, not chat.

    Assert the parsed object, not just a non-empty string: the failure mode is
    ``parsed=None`` or a 400 before any token is emitted.

    ⚠️ **必须用生产真实的 schema**，不要用本文件里另造一个。
    原因（2026-10-24 实测，deepseek-flash，三组各 20 轮）：

    ==========================  =========  ========  ==================
    工具名                        描述        结果      失败时喊的名字
    ==========================  =========  ========  ==================
    ``_PlannerOutput``（下划线）   空          18/20    ``false`` / ``invoke``
    ``PlannerOutput``            空          **20/20**  —
    ``PlannerOutputDoc``         有          **20/20**  —
    ==========================  =========  ========  ==================

    ⇒ 元凶是**工具名带下划线前缀**（模型把 ``_PlannerOutput`` 记成
    ``false`` / ``invoke`` 这种编出来的名字），**不是**描述为空
    （加不加 docstring 测不出差别）。

    ⇒ 生产里的 agent schema 全都不带下划线（``PlannerOutput`` /
    ``ExecutorOutput`` / ``CortexOutput`` …）⇒ **生产不受影响**，
    是这个 fixture 自己踩的坑。用真实 schema 才能测到真实行为。

    另外：真实模型偶尔不按要求调工具（哪怕工具名正常），所以这里**重试几次**
    —— 这是 live-LLM integration 测试的常态，不是掩盖 bug：重试 3 次仍失败
    就把每次的诊断一起抛出来。
    """
    from minitap.mobile_use.agents.planner.types import PlannerOutput
    from minitap.mobile_use.services.llm import get_deepseek_llm

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key or key in {"...", "your-deepseek-api-key"}:
        pytest.skip("DEEPSEEK_API_KEY not set, skipping integration test")

    mock_settings.DEEPSEEK_API_KEY = SecretStr(key)
    mock_settings.DEEPSEEK_BASE_URL = None
    mock_settings.DEEPSEEK_THINKING = thinking

    client = get_deepseek_llm(model_name="deepseek-flash")
    structured = client.with_structured_output(PlannerOutput)

    attempts: list[str] = []
    for attempt in range(1, 4):
        try:
            result = asyncio.run(structured.ainvoke("Split 'buy milk' into subgoals."))
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"#{attempt}: {type(exc).__name__}: {exc}")
            continue
        assert isinstance(result, PlannerOutput)
        assert result.subgoals
        return

    pytest.fail(
        "DeepSeek 结构化输出连续 3 次失败（每次的诊断）：\n" + "\n".join(attempts)
    )

