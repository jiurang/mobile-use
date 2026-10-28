"""提示词语言切换（``en`` / ``zh``）。

这个文件锁住三件事：

1. **语言怎么定**：任务级 > Agent 级 > 环境变量 > 默认英文。
   写错一个环境变量**不许**让整轮任务挂掉 —— 回落 + 上报即可。
2. **中文版文件怎么找**：``xxx.md`` → ``xxx-zh.md``（``cortex.zh.md`` 是
   历史命名，也得认）。找不到中文版时**回落英文 + 明确 warning** ——
   「以为在跑中文、其实跑的是英文」会把 A/B 结论带歪，必须出声。
3. **每个 agent 真的接上了开关**：prompt 里出现的是中文，不是英文。

⚠️ 第 3 条是这份测试存在的理由。语言开关最容易坏的方式不是「解析出错」，
而是**某个 agent 忘了接** —— 那份提示词永远是英文，而表面上一切正常。
所以下面每个 agent 都跑一遍它自己的提示词拼装路径，抓真正发给 LLM 的消息。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "minitap"))

import pytest  # noqa: E402
from pytest import MonkeyPatch  # noqa: E402

from minitap.mobile_use.agents.contextor.types import ContextorOutput  # noqa: E402
from minitap.mobile_use.agents.orchestrator.types import OrchestratorOutput  # noqa: E402
from minitap.mobile_use.agents.planner.types import Subgoal, SubgoalStatus  # noqa: E402
from minitap.mobile_use.config import LLM, LLMConfig, LLMConfigUtils, LLMWithFallback  # noqa: E402
from minitap.mobile_use.context import (  # noqa: E402
    DeviceContext,
    DevicePlatform,
    MobileUseContext,
    PerceptionConfig,
)
from minitap.mobile_use.prompt_language import (  # noqa: E402
    DEFAULT_PROMPT_LANGUAGE,
    PROMPT_LANGUAGE_ENV,
    PromptLanguage,
    normalize_language,
    pick,
    read_prompt,
    resolve_prompt_language,
    resolve_prompt_path,
)

_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENTS_DIR = _ROOT / "minitap" / "mobile_use" / "agents"

#: 本次交付翻译的 6 份中文提示词（文件名必须是 ``<stem>-zh.md``）。
TRANSLATED_PROMPTS = [
    "contextor/contextor-zh.md",
    "hopper/hopper-zh.md",
    "orchestrator/orchestrator-zh.md",
    "orchestrator/human-zh.md",
    "outputter/human-zh.md",
    "reporter/reporter-zh.md",
]

#: 开关覆盖到的全部 agent 提示词（含本次之前就有的中英两版）。
#: 每一个都必须能在 zh 下找到中文版 —— 否则开关就是半拉子的。
SWITCHED_PROMPTS = [
    "contextor/contextor.md",
    "hopper/hopper.md",
    "orchestrator/orchestrator.md",
    "orchestrator/human.md",
    "outputter/human.md",
    "reporter/reporter.md",
    "executor/executor.md",
    "cortex/cortex.md",
]


def _ctx(language: PromptLanguage = PromptLanguage.EN) -> MobileUseContext:
    def llm() -> LLMWithFallback:
        return LLMWithFallback(
            provider="openai", model="gpt-5", fallback=LLM(provider="openai", model="o4-mini")
        )

    return MobileUseContext(
        trace_id="test",
        device=DeviceContext(
            host_platform="WINDOWS",
            mobile_platform=DevicePlatform.ANDROID,
            device_id="test",
            device_width=1220,
            device_height=2712,
        ),
        llm_config=LLMConfig(
            planner=llm(),
            orchestrator=llm(),
            contextor=llm(),
            cortex=llm(),
            executor=llm(),
            utils=LLMConfigUtils(outputter=llm(), hopper=llm()),
        ),
        perception=PerceptionConfig(),
        prompt_language=language,
    )


class _FakeLLM:
    """只记录「发给它的是什么」的假 LLM。"""

    def __init__(self, calls: list, output=None):
        self.calls = calls
        self.output = output

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self.output


def _system_text(messages) -> str:
    """从消息列表里取出 system 那一条的正文。"""
    return "\n".join(str(m.content) for m in messages if m.type == "system")


# ---------------------------------------------------------------------------
# 1. 语言归一化
# ---------------------------------------------------------------------------


def test_normalize_language_accepts_common_spellings():
    assert normalize_language("zh") == PromptLanguage.ZH
    assert normalize_language("ZH-CN") == PromptLanguage.ZH
    assert normalize_language("中文") == PromptLanguage.ZH
    assert normalize_language("en") == PromptLanguage.EN
    assert normalize_language("English") == PromptLanguage.EN


def test_normalize_language_returns_none_when_it_cannot_tell():
    """认不出来 = **没指定**，不是「英文」。

    分清这两者很重要：前者要继续往下找（agent 级 → 环境变量），后者直接定案。
    """
    assert normalize_language("klingon") is None
    assert normalize_language(None) is None


# ---------------------------------------------------------------------------
# 2. 语言优先级
# ---------------------------------------------------------------------------


def test_task_level_wins_over_agent_level_and_env(monkeypatch: MonkeyPatch):
    monkeypatch.setenv(PROMPT_LANGUAGE_ENV, "en")
    assert resolve_prompt_language(task="zh", agent="en") == PromptLanguage.ZH
    assert resolve_prompt_language(task="en", agent="zh") == PromptLanguage.EN


def test_agent_level_wins_over_env(monkeypatch: MonkeyPatch):
    monkeypatch.setenv(PROMPT_LANGUAGE_ENV, "en")
    assert resolve_prompt_language(agent="zh") == PromptLanguage.ZH


def test_env_is_used_when_nothing_else_is_set(monkeypatch: MonkeyPatch):
    monkeypatch.setenv(PROMPT_LANGUAGE_ENV, "zh")
    assert resolve_prompt_language() == PromptLanguage.ZH


def test_default_is_english(monkeypatch: MonkeyPatch):
    """什么都不设 = 今天的逐字节行为（英文），不能悄悄变。"""
    monkeypatch.delenv(PROMPT_LANGUAGE_ENV, raising=False)
    assert resolve_prompt_language() == PromptLanguage.EN
    assert DEFAULT_PROMPT_LANGUAGE == PromptLanguage.EN


def test_illegal_value_falls_back_instead_of_raising(monkeypatch: MonkeyPatch):
    """写错的环境变量**不许**让整轮任务挂掉 —— 回落 + 上报即可。"""
    monkeypatch.setenv(PROMPT_LANGUAGE_ENV, "klingon")
    assert resolve_prompt_language() == PromptLanguage.EN


def test_illegal_task_level_falls_through_to_agent_level(monkeypatch: MonkeyPatch):
    monkeypatch.delenv(PROMPT_LANGUAGE_ENV, raising=False)
    assert resolve_prompt_language(task="klingon", agent="zh") == PromptLanguage.ZH


# ---------------------------------------------------------------------------
# 3. 中文版文件怎么找
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel", TRANSLATED_PROMPTS)
def test_translated_prompts_exist(rel: str):
    assert (_AGENTS_DIR / rel).is_file(), f"中文版提示词缺失：{rel}"


@pytest.mark.parametrize("rel", SWITCHED_PROMPTS)
def test_every_switched_prompt_has_a_chinese_version(rel: str):
    """开关覆盖到的每个提示词都必须有中文版 —— 否则开关是半拉子的。"""
    base = _AGENTS_DIR / rel
    zh = resolve_prompt_path(base, PromptLanguage.ZH)
    assert zh.is_file()
    assert zh != base, f"{rel} 在 zh 下没找到中文版"


def test_english_always_stays_on_the_original_file():
    base = _AGENTS_DIR / "contextor/contextor.md"
    assert resolve_prompt_path(base, PromptLanguage.EN) == base


def test_cortex_zh_uses_the_legacy_dot_suffix():
    """``cortex.zh.md`` 是历史命名（点号），也得认。

    不认它 ⇒ 那个文件在 zh 下会被静默回落成英文 ⇒ A/B 结论直接作废。
    """
    zh = resolve_prompt_path(_AGENTS_DIR / "cortex/cortex.md", PromptLanguage.ZH)
    assert zh.name == "cortex.zh.md"


def test_missing_chinese_version_falls_back_to_english_with_a_warning(caplog):
    """缺中文版 ⇒ **回落英文 + 明确 warning**。

    「以为在跑中文、其实跑的是英文」在输出里完全看不出来，
    只有这条 warning 能揭穿它。
    """
    base = _AGENTS_DIR / "planner/planner.md"
    with caplog.at_level("WARNING"):
        assert resolve_prompt_path(base, PromptLanguage.ZH) == base
    assert "没有中文版" in caplog.text


def test_read_prompt_returns_the_chinese_body():
    body = read_prompt(_AGENTS_DIR / "reporter/reporter.md", PromptLanguage.ZH)
    assert "你是 **Reporter**" in body
    # 字面量不许翻：字段名是模型要原样吐出的
    assert "`summary`" in body and "`evidence`" in body


def test_pick_switches_between_the_two_wordings():
    assert pick("english", "中文", language=PromptLanguage.ZH) == "中文"
    assert pick("english", "中文", language=PromptLanguage.EN) == "english"


# ---------------------------------------------------------------------------
# 4. 每个 agent 真的接上了开关
# ---------------------------------------------------------------------------


def test_executor_prompt_follows_the_switch():
    """Executor：system prompt 走 ``executor.md`` / ``executor-zh.md``。"""
    from minitap.mobile_use.agents.executor.prompt import render_executor_system

    en = render_executor_system(_ctx(PromptLanguage.EN))
    zh = render_executor_system(_ctx(PromptLanguage.ZH))
    assert "You are the **Executor**" in en
    assert "你是 **Executor**" in zh
    # 索引字段名不许翻 —— 模板里讲的是 `index`。
    zh_template = (_AGENTS_DIR / "executor/executor-zh.md").read_text(encoding="utf-8")
    assert "`index`" in zh_template


def test_cortex_prompt_follows_the_switch(monkeypatch: MonkeyPatch):
    """Cortex：环境变量显式指定路径时优先于语言开关（那是「精确用这个文件」）。"""
    from minitap.mobile_use.agents.cortex.cortex import load_cortex_template

    monkeypatch.delenv("MOBILE_USE_CORTEX_PROMPT", raising=False)
    assert "你是 Cortex" in load_cortex_template(PromptLanguage.ZH).render()
    assert "You are the Cortex" in load_cortex_template(PromptLanguage.EN).render()


def test_cortex_env_override_beats_the_language_switch(monkeypatch: MonkeyPatch):
    """环境变量显式指定文件 ⇒ 盖过语言开关（那是「我要精确用这个文件」）。"""
    from minitap.mobile_use.agents.cortex.cortex import load_cortex_template

    override = _AGENTS_DIR / "cortex/cortex.md"
    monkeypatch.setenv("MOBILE_USE_CORTEX_PROMPT", str(override))
    rendered = load_cortex_template(PromptLanguage.ZH, str(override)).render()
    assert "You are the Cortex" in rendered
    assert "你是 Cortex" not in rendered


def test_reporter_prompt_follows_the_switch():
    """Reporter：system prompt 静态，两种语言各自缓存。"""
    from minitap.mobile_use.agents.reporter.reporter import load_reporter_prompt

    assert "你是 **Reporter**" in load_reporter_prompt(PromptLanguage.ZH)
    assert "You are the **Reporter**" in load_reporter_prompt(PromptLanguage.EN)


def test_reporter_human_message_labels_follow_the_switch():
    from minitap.mobile_use.agents.reporter.reporter import build_reporter_messages

    zh = build_reporter_messages(
        goal="读出 Wi-Fi 名称",
        outcome="failed",
        log_text="tap(3)",
        failed_targets=[],
        language=PromptLanguage.ZH,
    )
    assert "尝试的子目标：" in _system_text(zh) + str(zh[1].content)
    en = build_reporter_messages(
        goal="read the wifi name",
        outcome="failed",
        log_text="tap(3)",
        failed_targets=[],
        language=PromptLanguage.EN,
    )
    assert "Subgoal attempted:" in str(en[1].content)


@pytest.mark.asyncio
async def test_hopper_prompt_follows_the_switch(monkeypatch: MonkeyPatch):
    """Hopper：抓它真正发给 LLM 的那条 system 消息。"""
    import minitap.mobile_use.agents.hopper.hopper as hopper_mod
    from minitap.mobile_use.agents.hopper.hopper import HopperOutput

    calls: list = []
    monkeypatch.setattr(
        hopper_mod,
        "get_llm",
        lambda **kwargs: _FakeLLM(calls, HopperOutput(found=False, output=None, reason="no match")),
    )

    await hopper_mod.hopper(_ctx(PromptLanguage.ZH), "找包名", "com.a / com.b")
    assert "## Hopper" in _system_text(calls[0])
    assert "从批量数据里" in _system_text(calls[0])

    calls.clear()
    await hopper_mod.hopper(_ctx(PromptLanguage.EN), "find package", "com.a / com.b")
    assert "Extract relevant information" in _system_text(calls[0])


@pytest.mark.asyncio
async def test_outputter_prompt_follows_the_switch(monkeypatch: MonkeyPatch):
    """Outputter：system 提示 + human 模板都要跟着切。"""
    import minitap.mobile_use.agents.outputter.outputter as outputter_mod
    from minitap.mobile_use.config import OutputConfig
    from minitap.mobile_use.graph.state import State

    state = State(
        messages=[],
        initial_goal="读出 Wi-Fi 名称",
        subgoal_plan=[],
        latest_ui_hierarchy=None,
        latest_ocr_results=None,
        focused_app_info=None,
        device_date=None,
        screen_state="on",
        latest_screen_elements=None,
        last_action_at=None,
        structured_decisions=None,
        complete_subgoals_by_ids=[],
        executor_messages=[],
        cortex_last_thought=None,
        agents_thoughts=["想了一下"],
    )

    calls: list = []
    monkeypatch.setattr(outputter_mod, "get_llm", lambda **kwargs: _FakeLLM(calls, {"ok": True}))

    await outputter_mod.outputter(
        _ctx(PromptLanguage.ZH),
        OutputConfig(structured_output=None, output_description=None),
        state,
    )
    zh_text = _system_text(calls[0]) + str(calls[0][1].content)
    assert "把多智能体推理过程的最终结果整理成结构化输出" in zh_text
    assert "最初的目标是" in zh_text

    calls.clear()
    await outputter_mod.outputter(
        _ctx(PromptLanguage.EN),
        OutputConfig(structured_output=None, output_description=None),
        state,
    )
    en_text = _system_text(calls[0]) + str(calls[0][1].content)
    assert "The original goal was:" in en_text


@pytest.mark.asyncio
async def test_contextor_prompt_follows_the_switch(monkeypatch: MonkeyPatch):
    """Contextor：抓它真正发给 LLM 的那条 system 消息。"""
    import minitap.mobile_use.agents.contextor.contextor as contextor_mod

    calls: list = []
    monkeypatch.setattr(
        contextor_mod,
        "get_llm",
        lambda **kwargs: _FakeLLM(
            calls, ContextorOutput(should_relaunch_app=False, reasoning="ok")
        ),
    )

    node = contextor_mod.ContextorNode(_ctx(PromptLanguage.ZH))
    await node._invoke_contextor_llm(
        initial_goal="读出 Wi-Fi 名称",
        subgoal_plan=[],
        agents_thoughts=["想了一下"],
        locked_app_package="com.a",
        current_app_package="com.b",
    )
    assert "你是 **Contextor Agent**" in _system_text(calls[0])

    calls.clear()
    node = contextor_mod.ContextorNode(_ctx(PromptLanguage.EN))
    await node._invoke_contextor_llm(
        initial_goal="read the wifi name",
        subgoal_plan=[],
        agents_thoughts=["thought"],
        locked_app_package="com.a",
        current_app_package="com.b",
    )
    assert "You are the **Contextor Agent**" in _system_text(calls[0])


@pytest.mark.asyncio
async def test_orchestrator_prompt_follows_the_switch(monkeypatch: MonkeyPatch):
    """Orchestrator：system + human 两块都要跟着切。"""
    import minitap.mobile_use.agents.orchestrator.orchestrator as orchestrator_mod
    from minitap.mobile_use.graph.state import State

    def _state():
        return State(
            messages=[],
            initial_goal="读出 Wi-Fi 名称",
            subgoal_plan=[
                Subgoal(id="sg1", description="打开 Wi-Fi 页", status=SubgoalStatus.PENDING)
            ],
            latest_ui_hierarchy=None,
            latest_ocr_results=None,
            focused_app_info=None,
            device_date=None,
            screen_state="on",
            latest_screen_elements=None,
            last_action_at=None,
            structured_decisions=None,
            complete_subgoals_by_ids=["sg1"],
            executor_messages=[],
            cortex_last_thought=None,
            agents_thoughts=["想了一下"],
        )

    calls: list = []
    monkeypatch.setattr(
        orchestrator_mod,
        "get_llm",
        lambda **kwargs: _FakeLLM(
            calls, OrchestratorOutput(completed_subgoal_ids=[], needs_replaning=False, reason="ok")
        ),
    )

    await orchestrator_mod.OrchestratorNode(_ctx(PromptLanguage.ZH))(_state())
    zh_text = _system_text(calls[0]) + str(calls[0][1].content)
    assert "你是 **Orchestrator**" in zh_text
    assert "以下是供你分析的输入" in zh_text

    calls.clear()
    await orchestrator_mod.OrchestratorNode(_ctx(PromptLanguage.EN))(_state())
    en_text = _system_text(calls[0]) + str(calls[0][1].content)
    assert "You are the **Orchestrator**" in en_text


# ---------------------------------------------------------------------------
# 5. 开关怎么落到一次运行上
# ---------------------------------------------------------------------------


def test_task_request_carries_the_language():
    """任务级开关：SDK 的 ``task.with_prompt_language(...)`` 要真的带进请求里。"""
    from minitap.mobile_use.sdk.builders.task_request_builder import TaskRequestBuilder

    request = (
        TaskRequestBuilder[None](goal="读出 Wi-Fi 名称")
        .with_prompt_language(PromptLanguage.ZH)
        .build()
    )
    assert request.prompt_language == PromptLanguage.ZH


def test_agent_config_carries_the_language():
    """Agent 级开关：``config.with_prompt_language(...)`` 要真的进 AgentConfig。"""
    from minitap.mobile_use.sdk.builders.agent_config_builder import AgentConfigBuilder
    from minitap.mobile_use.sdk.types.agent import AgentProfile

    config = (
        AgentConfigBuilder()
        .add_profile(
            AgentProfile(
                name="default",
                llm_config=LLMConfig(
                    planner=LLMWithFallback(
                        provider="openai",
                        model="gpt-5",
                        fallback=LLM(provider="openai", model="o4-mini"),
                    ),
                    orchestrator=LLMWithFallback(
                        provider="openai",
                        model="gpt-5",
                        fallback=LLM(provider="openai", model="o4-mini"),
                    ),
                    contextor=LLMWithFallback(
                        provider="openai",
                        model="gpt-5",
                        fallback=LLM(provider="openai", model="o4-mini"),
                    ),
                    cortex=LLMWithFallback(
                        provider="openai",
                        model="gpt-5",
                        fallback=LLM(provider="openai", model="o4-mini"),
                    ),
                    executor=LLMWithFallback(
                        provider="openai",
                        model="gpt-5",
                        fallback=LLM(provider="openai", model="o4-mini"),
                    ),
                    utils=LLMConfigUtils(
                        outputter=LLMWithFallback(
                            provider="openai",
                            model="gpt-5",
                            fallback=LLM(provider="openai", model="o4-mini"),
                        ),
                        hopper=LLMWithFallback(
                            provider="openai",
                            model="gpt-5",
                            fallback=LLM(provider="openai", model="o4-mini"),
                        ),
                    ),
                ),
            ),
            validate=False,
        )
        .with_prompt_language(PromptLanguage.ZH)
        .build()
    )
    assert config.prompt_language == PromptLanguage.ZH


def test_cli_exposes_the_prompt_language_option():
    """CLI 上的开关：``--prompt-language`` 要真的挂在 Typer 上。"""
    from typer.testing import CliRunner

    from minitap.mobile_use.main import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--prompt-language" in result.stdout
