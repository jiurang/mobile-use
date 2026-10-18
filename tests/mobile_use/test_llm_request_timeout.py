"""Every model client must bound its own request timeout.

``timeout=None`` / an unset ``request_timeout`` does not mean "generous", it
means "wait 600s per attempt and retry twice": the OpenAI SDK's default is 10
minutes and langchain retries, so one hung TCP connection parks an agent for
half an hour. The only symptom is the log line ``Waiting for LLM call
response...`` from ``invoke_llm_with_timeout_message``, which prints a *hint*
after 10 seconds and then awaits the call forever.

This was found the hard way: on a phone over USB a MiniMax call started at
22:23:59 and had produced nothing by 22:36, with the run unrecoverable.

An unbounded timeout is therefore a defect whether or not the network is
currently misbehaving, so it is pinned here rather than left to review.
"""

from __future__ import annotations

import math

import pytest

from minitap.mobile_use.services import llm as llm_mod


@pytest.fixture
def fake_minimax_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_mod.settings, "MINIMAX_API_KEY", "sk-test-not-a-real-key", raising=False
    )


def test_the_constant_is_actually_bounded() -> None:
    value = llm_mod.LLM_REQUEST_TIMEOUT_SECONDS
    assert isinstance(value, (int, float))
    assert math.isfinite(value), "an infinite timeout is the bug this file exists for"
    assert 0 < value <= 600, (
        f"{value}s is not a sane ceiling: the SDK default alone is 600s, so this "
        "must be tighter than the default to add any protection at all"
    )


def test_minimax_client_bounds_timeout_and_retries(fake_minimax_key: None) -> None:
    client = llm_mod.get_minimax_llm()

    assert client.request_timeout == llm_mod.LLM_REQUEST_TIMEOUT_SECONDS
    # Unbounded x unbounded retries is how one dropped packet costs 30 minutes.
    assert client.request_timeout is not None
    assert client.max_retries == 1


def test_anthropic_client_does_not_opt_out_of_a_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_mod.settings, "ANTHROPIC_API_KEY", "sk-ant-test", raising=False
    )
    client = llm_mod.get_anthropic_llm()

    # Not `request_timeout`: ChatAnthropic names the field
    # `default_request_timeout` (alias "timeout"), and an unrecognised keyword is
    # dropped without complaint — so this assertion is the only thing standing
    # between "edited" and "actually bounded".
    assert client.default_request_timeout == llm_mod.LLM_REQUEST_TIMEOUT_SECONDS
