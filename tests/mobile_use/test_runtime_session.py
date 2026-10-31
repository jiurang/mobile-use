"""会话运行时（Session / Turn / AgentSession / Step）的行为测试。

这一层只做观测，所以测试也只断言**边界与事件**，不碰任何 agent 逻辑。
"""

import json
from pathlib import Path

from minitap.mobile_use.config import LLM, LLMConfig, LLMConfigUtils, LLMWithFallback
from minitap.mobile_use.context import (
    DeviceContext,
    DevicePlatform,
    MobileUseContext,
    PerceptionConfig,
)
from minitap.mobile_use.runtime.session import Session


def _types(session: Session) -> list[str]:
    return [event.type for event in session.log.events]


def _make_session(tmp_path: Path, *, enabled: bool = True) -> Session:
    return Session(log_dir=tmp_path, enabled=enabled)


def test_session_lifecycle_emits_bordered_events(tmp_path: Path):
    session = _make_session(tmp_path)
    turn = session.begin_turn(goal="open the settings app", max_steps=40, device_id="emulator-5554")
    executor = session.open_agent_session("executor", scope="subgoal:sg_1", seed="previous attempt")

    step = executor.begin_step(index=1, goal="tap settings")
    step.end("action", {"action": "tap"})

    executor.close("completed", summary="settings opened")
    turn.end("completed")
    session.end()

    assert _types(session) == [
        "session/start",
        "turn/start",
        "agent_session/start",
        "step/start",
        "step/end",
        "agent_session/close",
        "turn/end",
        "session/end",
    ]
    seqs = [event.seq for event in session.log.events]
    assert seqs == list(range(1, len(seqs) + 1))


def test_events_are_persisted_as_jsonl(tmp_path: Path):
    session = _make_session(tmp_path)
    turn = session.begin_turn(goal="check the weather")
    step = session.begin_step("cortex", goal="decide next subgoal")
    step.end("completed")
    turn.end("completed")
    session.end()

    lines = session.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(session.log.events)
    parsed = [json.loads(line) for line in lines]
    assert [item["type"] for item in parsed] == _types(session)
    assert all(item["session_id"] == session.id for item in parsed)


def test_step_end_is_idempotent(tmp_path: Path):
    session = _make_session(tmp_path)
    turn = session.begin_turn(goal="do one thing")
    step = session.begin_step("executor", goal="step goal")

    step.end("action")
    step.end("done")

    assert _types(session).count("step/end") == 1
    assert session.log.events[-1].payload["outcome"] == "action"
    turn.end("completed")


def test_turn_end_closes_dangling_steps_and_sessions(tmp_path: Path):
    session = _make_session(tmp_path)
    turn = session.begin_turn(goal="abandoned attempt")
    session.begin_step("executor", goal="never finishes")

    turn.end("cancelled")

    types = _types(session)
    assert "step/end" in types
    assert session.log.events[types.index("step/end")].payload["outcome"] == "aborted"
    assert "agent_session/close" in types
    assert session.log.events[types.index("agent_session/close")].payload["reason"] == "turn_end"


def test_new_goal_opens_a_new_executor_session(tmp_path: Path):
    session = _make_session(tmp_path)
    turn = session.begin_turn(goal="two attempts")

    first = session.open_agent_session("executor", scope="subgoal:sg_1")
    first.close("handoff", summary="first attempt failed")
    second = session.open_agent_session("executor", scope="subgoal:sg_1")

    assert first.id != second.id
    assert [record.close_reason for record in session.agent_sessions("executor")] == [
        "handoff",
        None,
    ]
    turn.end("completed")


def test_no_active_turn_is_silent(tmp_path: Path):
    session = _make_session(tmp_path)

    step = session.begin_step("executor", goal="orphan step")
    step.end("action")

    assert session.log.events == []
    assert not session.path.exists()


def test_disabled_log_writes_nothing(tmp_path: Path):
    session = _make_session(tmp_path, enabled=False)
    turn = session.begin_turn(goal="no log")
    session.begin_step("executor").end("action")
    turn.end("completed")
    session.end()

    assert session.log.events == []
    assert not session.path.exists()


def test_unwritable_log_directory_degrades(tmp_path: Path):
    # 把日志根目录指向一个**文件**，mkdir 必然失败：整层应降级而不是抛。
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    session = Session(log_dir=blocked, enabled=True)

    turn = session.begin_turn(goal="degrade")
    session.begin_step("executor").end("action")
    turn.end("completed")

    assert session.log.path.exists() is False


def _llm():
    return LLMWithFallback(
        provider="openai", model="gpt-5", fallback=LLM(provider="openai", model="o4-mini")
    )


def _ctx_with(session: Session) -> MobileUseContext:
    llm = _llm()
    return MobileUseContext(
        trace_id="task-1",
        device=DeviceContext(
            host_platform="WINDOWS",
            mobile_platform=DevicePlatform.ANDROID,
            device_id="emulator-5554",
            device_width=1080,
            device_height=2400,
        ),
        llm_config=LLMConfig(
            planner=llm,
            orchestrator=llm,
            contextor=llm,
            cortex=llm,
            executor=llm,
            utils=LLMConfigUtils(outputter=llm, hopper=llm),
        ),
        perception=PerceptionConfig(),
        runtime_session=session,
    )


def _node_state(**kw):
    from minitap.mobile_use.graph.state import State

    base = {
        "initial_goal": "open the settings app",
        "subgoal_plan": [],
        "latest_ui_hierarchy": None,
        "latest_ocr_results": None,
        "latest_screen_elements": None,
        "focused_app_info": None,
        "device_date": None,
        "complete_subgoals_by_ids": [],
        "agents_thoughts": [],
        "remaining_steps": 10,
        "executor_messages": [],
        "scratchpad": {},
    }
    base.update(kw)
    return State(**base)


def test_real_nodes_emit_the_turn_chain(tmp_path: Path, monkeypatch):
    """真 Cortex + 真 Executor（只换 LLM 层）跑出的完整事件链。"""
    import asyncio

    from minitap.mobile_use.agents.cortex import cortex as cortex_mod
    from minitap.mobile_use.agents.cortex.cortex import CortexNode
    from minitap.mobile_use.agents.executor.executor import ExecutorNode
    from minitap.mobile_use.agents.executor.types import ExecutorOutput

    session = _make_session(tmp_path)
    turn = session.begin_turn(goal="open the settings app", max_steps=10)
    ctx = _ctx_with(session)

    class _StubLLM:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, messages, *args, **kwargs):
            class _Out:
                next_subgoal = "tap into Settings"
                completion_reason = None
                subgoal_completed = False

            return _Out()

    monkeypatch.setattr(cortex_mod, "get_llm", lambda *a, **kw: _StubLLM())

    asyncio.run(CortexNode(ctx)(_node_state()))

    async def _decide(self, *, state, goal, stall_notice: str = ""):
        return ExecutorOutput(
            actions=[{"action": "tap", "target": {"index": 1}}],
            agent_thought="tap the settings row",
        )

    monkeypatch.setattr(ExecutorNode, "_decide", _decide)
    asyncio.run(
        ExecutorNode(ctx)(
            _node_state(
                executor_goal="tap into Settings",
                executor_steps=0,
            )
        )
    )

    turn.end("completed")
    session.end()

    types = _types(session)
    assert types[0] == "session/start"
    assert types[1] == "turn/start"
    assert "agent_session/start" in types
    assert "step/start" in types
    assert types.count("step/end") == 2  # cortex 一步 + executor 一步
    assert types[-2:] == ["turn/end", "session/end"]

    executor_steps = [
        event
        for event in session.log.events
        if event.type == "step/start" and event.payload.get("agent") == "executor"
    ]
    assert len(executor_steps) == 1
    assert executor_steps[0].payload["index"] == 1

    # 真节点也把段边界报上来：空历史 + 这一步写入的 1 条 AIMessage。
    executor_step_ends = [
        event
        for event in session.log.events
        if event.type == "step/end" and event.payload.get("agent") == "executor"
    ]
    assert executor_step_ends[0].payload["history"] == 1
    executor_session = session.agent_sessions("executor")[0]
    assert executor_steps[0].agent_session_id == executor_session.id
    assert executor_session.close_reason == "turn_end"


def test_two_turns_share_one_session(tmp_path: Path):
    """Session 跨 turn 存活：同一 id 下多个 turn 只开一次 session。"""
    session = _make_session(tmp_path)
    first = session.begin_turn(goal="turn one")
    first.end("completed")
    second = session.begin_turn(goal="turn two")
    second.end("completed")
    session.end()

    assert _types(session).count("session/start") == 1
    assert _types(session).count("turn/start") == 2
    assert _types(session).count("turn/end") == 2


def test_a_new_process_closes_the_interrupted_turn(tmp_path: Path):
    """进程被杀后再开同一 session id：悬空的 step / 会话 / turn 必须补上边界。"""
    session_id = "s_interrupted"
    crashed = Session(session_id=session_id, log_dir=tmp_path)
    crashed.begin_turn(goal="interrupted", device_id="dev")
    crashed.begin_step("executor", goal="never ends")
    crashed.log.close()  # 模拟进程退出：事件已落盘，但没有 turn/end

    resumed = Session(session_id=session_id, log_dir=tmp_path)
    resumed.start(device_id="dev")

    events = resumed.log.read_events()
    types = [event.type for event in events]
    assert types[-4:] == ["step/end", "agent_session/close", "turn/end", "session/start"]

    recovered_step = events[-4]
    assert recovered_step.payload["outcome"] == "aborted"
    assert recovered_step.payload["recovered"] is True
    assert events[-3].payload["reason"] == "interrupted"
    assert events[-2].payload == {
        "status": "cancelled",
        "error": "interrupted",
        "recovered": True,
    }
    assert events[-1].payload["resumed"] is True

    # seq 必须接着上一进程的最大值，不能从 1 重来。
    seqs = [event.seq for event in events]
    assert seqs == list(range(1, len(seqs) + 1))
    resumed.end()


def test_a_clean_log_recovers_nothing(tmp_path: Path):
    session_id = "s_clean"
    first = Session(session_id=session_id, log_dir=tmp_path)
    turn = first.begin_turn(goal="done cleanly")
    turn.end("completed")
    first.end()

    second = Session(session_id=session_id, log_dir=tmp_path)
    assert second.recover() == 0
    second.start()
    assert "resumed" not in second.log.events[-1].payload
    second.end()


def test_a_parse_retry_reuses_the_assembled_prompt(tmp_path: Path, monkeypatch):
    """解析失败重试不二次组装：一个 Step 只拼一次，重试只在末尾加一条 nudge。"""
    import asyncio

    from minitap.mobile_use.agents.executor import executor as executor_mod
    from minitap.mobile_use.agents.executor.executor import ExecutorNode

    session = _make_session(tmp_path)
    turn = session.begin_turn(goal="retry once")
    ctx = _ctx_with(session)

    assembled: list[dict] = []
    real = executor_mod.build_executor_messages

    def _counting(**kwargs):
        assembled.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(executor_mod, "build_executor_messages", _counting)
    monkeypatch.setattr(executor_mod, "get_structured_llm", lambda **kwargs: object())

    async def _invoke(self, runner, messages, schema):
        return None, "no parseable output"

    monkeypatch.setattr(ExecutorNode, "_invoke", _invoke)

    output = asyncio.run(ExecutorNode(ctx)._decide(state=_node_state(), goal="tap something"))

    assert len(assembled) == 1, "重试不该重新组装 prompt"
    assert output.actions == []
    turn.end("completed")


def test_history_is_monotonic_within_a_session(tmp_path: Path):
    """段边界只增不减：同一会话内 history 随步数增长。"""
    session = _make_session(tmp_path)
    turn = session.begin_turn(goal="three steps")
    executor = session.open_agent_session("executor", scope="subgoal:sg_1")

    for index, history in enumerate((2, 5, 9), start=1):
        executor.begin_step(index=index).end("action", {"history": history})

    executor.close("completed")

    ends = [e for e in session.log.events if e.type == "step/end"]
    assert [e.payload["history"] for e in ends] == [2, 5, 9]
    assert session.agent_sessions("executor")[0].history == 9
    close = next(e for e in session.log.events if e.type == "agent_session/close")
    assert close.payload["history"] == 9
    turn.end("completed")


def test_close_omits_history_when_no_step_reported_it(tmp_path: Path):
    """没有 Step 报过 history 时，关闭事件不带这个键（观测值缺失不能算 0）。"""
    session = _make_session(tmp_path)
    turn = session.begin_turn(goal="no history")

    empty = session.open_agent_session("planner", scope="turn")
    empty.close("turn_end")

    with_history = session.open_agent_session("cortex", scope="turn")
    with_history.begin_step().end("completed", {"history": 4})
    with_history.close("turn_end")

    closes = [e.payload for e in session.log.events if e.type == "agent_session/close"]
    assert "history" not in closes[0]
    assert closes[1]["history"] == 4
    turn.end("completed")


def test_a_new_session_starts_from_the_reset_history(tmp_path: Path):
    """换取后新会话从清空后的条数开始，不继承旧会话的末值。"""
    session = _make_session(tmp_path)
    turn = session.begin_turn(goal="two attempts")

    first = session.open_agent_session("executor", scope="subgoal:sg_1")
    first.begin_step(index=1).end("action", {"history": 12})
    first.close("handoff", summary="first attempt failed")

    second = session.open_agent_session("executor", scope="subgoal:sg_1")
    second.begin_step(index=1).end("action", {"history": 1})
    second.close("completed")

    records = session.agent_sessions("executor")
    assert [record.history for record in records] == [12, 1]
    turn.end("completed")


def test_context_excludes_runtime_session(tmp_path: Path):
    session = _make_session(tmp_path)
    llm = LLMWithFallback(
        provider="openai", model="gpt-5", fallback=LLM(provider="openai", model="o4-mini")
    )
    ctx = MobileUseContext(
        trace_id="task-1",
        device=DeviceContext(
            host_platform="WINDOWS",
            mobile_platform=DevicePlatform.ANDROID,
            device_id="emulator-5554",
            device_width=1080,
            device_height=2400,
        ),
        llm_config=LLMConfig(
            planner=llm,
            orchestrator=llm,
            contextor=llm,
            cortex=llm,
            executor=llm,
            utils=LLMConfigUtils(outputter=llm, hopper=llm),
        ),
        perception=PerceptionConfig(),
        runtime_session=session,
    )

    dumped = ctx.model_dump()
    assert "runtime_session" not in dumped
    assert ctx.runtime_session is session
