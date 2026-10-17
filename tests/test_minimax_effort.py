"""`MINIMAX_EFFORT`（output_config.effort）的生效性测试。

背景（2026-10-17 真机实测）
--------------------------
M3.1-Flash-Preview **关不掉**思考（官方文档：always thinks），只能调深度：

    "MiniMax-M3.1-Flash-Preview additionally always thinks and supports tunable
     thinking depth via output_config.effort"

取值 low / medium / high / xhigh / max，**默认 max**。默认行为不在我们的配置里、
也没法被审计，所以显式写出来。

这里守四件事：

1. 归一化：大小写 / 空格写错不能静默变默认 —— 默认是 max（最贵那档），
   静默回落等于悄悄多花钱。
2. 真的下发：``extra_body`` 必须出现在请求里。「字段声明了但没传到运行时」和
   「写进了缓存 key 但没人用」是同一类最难查的 bug，必须用可观测差异判断。
3. **不下发 ``thinking``**：M3.1 官方明确 "If type is sent it must be adaptive;
   disabled returns HTTP 400"。留一个能设成 disabled 的旋钮就是地雷 ——
   谁设了就是每个请求 400，而且症状是「所有调用全挂」，很难联想到配置。
4. 原有行为（模型名、temperature 夹取）不能被这次改动带坏。
"""

from __future__ import annotations

import pytest

from minitap.mobile_use.services import llm as llm_mod


class TestResolveEffort:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, None),
            ("", None),
            ("   ", None),
            ("max", "max"),
            ("MAX", "max"),
            ("  Max  ", "max"),
            ("low", "low"),
            ("MEDIUM", "medium"),
            ("high", "high"),
            ("xhigh", "xhigh"),
        ],
    )
    def test_normalization(self, monkeypatch, raw, expected):
        monkeypatch.setattr(llm_mod.settings, "MINIMAX_EFFORT", raw, raising=False)
        assert llm_mod.resolve_minimax_effort() == expected

    def test_unknown_value_warns_instead_of_silently_using_max(self, monkeypatch, caplog):
        """不认识的值必须 warning。

        静默回落在这里特别危险：服务端默认是 **max**（最贵），写错一个字符
        就变成「按最贵那档跑」，没有任何症状。
        """
        monkeypatch.setattr(llm_mod.settings, "MINIMAX_EFFORT", "yes-please", raising=False)
        with caplog.at_level("WARNING"):
            assert llm_mod.resolve_minimax_effort() is None
        assert "MINIMAX_EFFORT" in caplog.text
        assert "yes-please" in caplog.text

    def test_all_documented_levels_are_accepted(self, monkeypatch):
        """官方文档列的五个档位必须全认，漏一个就是静默降级成 max。"""
        for level in ("low", "medium", "high", "xhigh", "max"):
            monkeypatch.setattr(llm_mod.settings, "MINIMAX_EFFORT", level, raising=False)
            assert llm_mod.resolve_minimax_effort() == level


class TestRequestPayload:
    """判据是**可观测的请求体**，不是「代码里出现过这个字段」。"""

    def _client(self, monkeypatch, effort):
        monkeypatch.setattr(llm_mod.settings, "MINIMAX_API_KEY", "test-key", raising=False)
        monkeypatch.setattr(llm_mod.settings, "MINIMAX_EFFORT", effort, raising=False)
        return llm_mod.get_minimax_llm("MiniMax-M3.1-Flash-Preview", temperature=1)

    def test_max_actually_reaches_the_request(self, monkeypatch):
        client = self._client(monkeypatch, "max")
        assert client.extra_body == {"output_config": {"effort": "max"}}

    def test_low_actually_reaches_the_request(self, monkeypatch):
        client = self._client(monkeypatch, "low")
        assert client.extra_body == {"output_config": {"effort": "low"}}

    def test_unset_sends_nothing(self, monkeypatch):
        """不配置时必须**不发** —— 发一个空对象可能被服务端当成 max。"""
        client = self._client(monkeypatch, None)
        assert not client.extra_body

    def test_never_sends_thinking_param(self, monkeypatch):
        """M3.1 上 thinking 只能发 adaptive，发 disabled 直接 400。

        这是本文件最该守住的一条：一个「关掉思考」的旋钮在 M3.1 上不是无效，
        而是**主动破坏**。任何把它加回来的改动都会让每个请求 400。
        """
        for effort in ("low", "medium", "high", "xhigh", "max", None):
            client = self._client(monkeypatch, effort)
            assert "thinking" not in (client.extra_body or {}), f"effort={effort} 时不该带 thinking"

    def test_model_name_is_preserved(self, monkeypatch):
        client = self._client(monkeypatch, "max")
        assert client.model_name == "MiniMax-M3.1-Flash-Preview"


class TestTemperatureClamp:
    """原有行为不能被这次改动带坏。"""

    @pytest.mark.parametrize(
        ("t", "expected"), [(-0.5, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (1.5, 1.0)]
    )
    def test_clamps(self, monkeypatch, t, expected):
        monkeypatch.setattr(llm_mod.settings, "MINIMAX_API_KEY", "test-key", raising=False)
        monkeypatch.setattr(llm_mod.settings, "MINIMAX_EFFORT", "max", raising=False)
        client = llm_mod.get_minimax_llm("MiniMax-M3.1-Flash-Preview", temperature=t)
        assert client.temperature == expected
