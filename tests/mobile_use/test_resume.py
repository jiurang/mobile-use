"""P3 续跑的两块地基：Turn 的接续与 checkpoint 的跨进程恢复。

这里不跑真机，也不碰 LLM：用一张最小的图证明「进程被杀后，第二次连接能接着
上一次的 checkpoint 往下跑」，以及 ``Session.resume_turn`` 不会新开一个 Turn。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from minitap.mobile_use.runtime.session import Session


class _Counter(TypedDict):
    n: int


def _build_graph(flag: Path, calls: Path):
    """A 永远成功；B 第一次必然抛错（用 flag 模拟「进程被杀」）。"""

    def step_a(state: _Counter) -> _Counter:
        calls.write_text(calls.read_text() + "a" if calls.exists() else "a", encoding="utf-8")
        return {"n": state["n"] + 1}

    def step_b(state: _Counter) -> _Counter:
        calls.write_text(calls.read_text() + "b", encoding="utf-8")
        if not flag.exists():
            flag.write_text("raised once", encoding="utf-8")
            raise RuntimeError("simulated crash")
        return {"n": state["n"] + 1}

    graph = StateGraph(_Counter)
    graph.add_node("a", step_a)
    graph.add_node("b", step_b)
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    return graph


def _thread() -> RunnableConfig:
    return {"configurable": {"thread_id": "s_demo:t_001"}}


def test_a_new_connection_resumes_from_the_last_checkpoint(tmp_path: Path):
    flag = tmp_path / "flag"
    calls = tmp_path / "calls"
    db = tmp_path / "checkpoints.db"
    graph = _build_graph(flag, calls)

    async def first_run() -> None:
        async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
            app = graph.compile(checkpointer=saver)
            try:
                await app.ainvoke({"n": 0}, _thread())
            except RuntimeError:
                pass  # 模拟进程在 b 之后被杀

    async def second_run() -> _Counter:
        # 新连接 = 新进程：只能读到磁盘上的 checkpoint。
        async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
            app = graph.compile(checkpointer=saver)
            return cast(_Counter, await app.ainvoke(None, _thread()))

    asyncio.run(first_run())
    result = asyncio.run(second_run())

    # a 只跑过一次：续跑从 checkpoint 之后开始，不重放已经完成的节点。
    assert calls.read_text(encoding="utf-8").startswith("ab")
    assert result == {"n": 2}


def test_resume_turn_continues_the_interrupted_turn(tmp_path: Path):
    session_id = "s_resume"
    crashed = Session(session_id=session_id, log_dir=tmp_path)
    turn = crashed.begin_turn(goal="open settings", max_steps=40, device_id="dev")
    crashed.begin_step("executor", goal="never ends")
    crashed.log.close()

    resumed = Session(session_id=session_id, log_dir=tmp_path)
    turn = resumed.resume_turn()

    assert turn is not None
    assert turn.id == "t_001", "续跑必须接着原来的 turn id，不能新开一个"
    assert turn.goal == "open settings"
    assert turn.max_steps == 40

    turn.end("completed")
    resumed.end()

    events = resumed.log.read_events()
    types = [event.type for event in events]
    assert types[-7:] == [
        "step/end",
        "agent_session/close",
        "turn/end",
        "session/start",
        "turn/resume",
        "turn/end",
        "session/end",
    ]
    recovered = [event for event in events if event.payload.get("recovered") is True]
    assert [event.type for event in recovered] == [
        "step/end",
        "agent_session/close",
        "turn/end",
    ]
    assert recovered[0].payload["outcome"] == "aborted"
    assert events[-3].payload == {"goal": "open settings"}


def test_adelete_thread_removes_the_checkpoint(tmp_path: Path):
    """正常收尾时 Agent 会调 ``adelete_thread``：快照不能无限堆积。"""
    db = tmp_path / "checkpoints.db"
    graph = _build_graph(tmp_path / "flag", tmp_path / "calls")
    (tmp_path / "flag").write_text("done", encoding="utf-8")

    async def run() -> bool:
        async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
            app = graph.compile(checkpointer=saver)
            await app.ainvoke({"n": 0}, _thread())
            await saver.adelete_thread("s_demo:t_001")
            return await saver.aget_tuple(_thread()) is None

    assert asyncio.run(run()) is True


def test_resume_turn_returns_none_for_a_clean_session(tmp_path: Path):
    session = Session(session_id="s_clean_resume", log_dir=tmp_path)
    turn = session.begin_turn(goal="already finished")
    turn.end("completed")
    session.end()

    reopened = Session(session_id="s_clean_resume", log_dir=tmp_path)
    assert reopened.resume_turn() is None
    reopened.end()
