"""四层会话对象：Session / Turn / AgentSession / Step。

这里只维护**边界与 id**，不持有任何 agent 的上下文。上下文仍在原来该在的地方
（``State`` / ``prompt``），所以接入这一层不改变任何行为。

生命周期规则（见 ``docs/会话架构方案v1.md`` 章节5）：

* 一个 Session 同一时刻只有一个 running Turn；Turn 串行是设备独占决定的。
* 每个 Step 必须属于一个 open 的 AgentSession；关闭后只读。
* Step 的结束是幂等的：重复调用 ``end`` 只保留第一次的结果。
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from minitap.mobile_use.runtime.events import Event, SessionLog
from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


def _new_session_id() -> str:
    return f"s_{uuid.uuid4().hex[:12]}"


def _session_id_from_env() -> str | None:
    """``MOBILE_USE_SESSION_ID`` 固定会话 id：进程重启后接着写同一份事件日志。"""
    value = (os.environ.get("MOBILE_USE_SESSION_ID") or "").strip()
    return value or None


@dataclass
class AgentSessionRecord:
    """一个 agent 在一次 turn 内的局部上下文域。"""

    id: str
    turn_id: str
    agent: str
    scope: str
    seed: str | None = None
    status: str = "open"
    close_reason: str | None = None
    steps: int = 0
    last_step_id: str | None = None
    # 局部历史的条数（由 Step 的 ``extra["history"]`` 上报，取最大值）。
    # 只回答「这段历史有多大」，不持有消息本身 —— 数据仍在 LangGraph state。
    history: int = 0
    started_at: datetime = field(default_factory=_now)
    ended_at: datetime | None = None


class StepHandle:
    """一次模型请求 + 工具批次。``end`` 幂等。"""

    def __init__(
        self,
        session: Session,
        *,
        step_id: str,
        agent: str,
        index: int,
        agent_session_id: str,
        record: AgentSessionRecord | None = None,
        meta: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> None:
        self._session = session
        self._enabled = enabled
        self._record = record
        self.id = step_id
        self.agent = agent
        self.index = index
        self.agent_session_id = agent_session_id
        self._meta = meta or {}
        self._started = time.perf_counter()
        self._ended = False

    @property
    def ended(self) -> bool:
        return self._ended

    def end(self, outcome: str, extra: dict[str, Any] | None = None) -> None:
        if self._ended:
            return
        self._ended = True
        if not self._enabled:
            return
        if self in self._session._open_steps:
            self._session._open_steps.remove(self)
        payload: dict[str, Any] = {
            "agent": self.agent,
            "outcome": outcome,
            "index": self.index,
            "duration_ms": round((time.perf_counter() - self._started) * 1000),
            **self._meta,
            **(extra or {}),
        }
        history = payload.get("history")
        if self._record is not None and isinstance(history, int):
            # 段边界只增不减：同一会话内历史只追加。
            self._record.history = max(self._record.history, history)
        self._session.log.append(
            "step/end",
            turn_id=self._session.current_turn_id,
            agent_session_id=self.agent_session_id,
            step_id=self.id,
            payload=payload,
        )


class AgentSessionHandle:
    """一个 agent 在一次 turn 内的会话句柄。"""

    def __init__(self, session: Session, record: AgentSessionRecord) -> None:
        self._session = session
        self.record = record

    @property
    def id(self) -> str:
        return self.record.id

    @property
    def agent(self) -> str:
        return self.record.agent

    def begin_step(
        self,
        *,
        index: int | None = None,
        goal: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> StepHandle:
        return self._session._begin_step(record=self.record, index=index, goal=goal, extra=extra)

    def close(self, reason: str, summary: str | None = None) -> None:
        self._session._close_agent_session(self.record, reason=reason, summary=summary)


class TurnHandle:
    """一次用户目标。``end`` 幂等。"""

    def __init__(self, session: Session, turn_id: str, goal: str, max_steps: int | None) -> None:
        self._session = session
        self.id = turn_id
        self.goal = goal
        self.max_steps = max_steps
        self.status = "running"
        self._ended = False

    @property
    def ended(self) -> bool:
        return self._ended

    def end(self, status: str, error: str | None = None) -> None:
        if self._ended:
            return
        self._ended = True
        self.status = status
        # 先收尾悬空的 Step（异常/取消路径），再关 AgentSession。
        self._session._close_open_steps(outcome="aborted")
        self._session._close_all_agent_sessions(reason="turn_end")
        self._session.log.append(
            "turn/end",
            turn_id=self.id,
            payload={"status": status, **({"error": error} if error else {})},
        )
        if self._session._current_turn is self:
            self._session._current_turn = None


class Session:
    """一次设备会话：跨多个 Turn 存活。

    ``enabled=False`` 或 ``log_dir`` 不可写时，整层自动降级成 no-op。
    """

    def __init__(
        self,
        *,
        session_id: str | None = None,
        log_dir: Path | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.id = session_id or _session_id_from_env() or _new_session_id()
        self.log = SessionLog(session_id=self.id, directory=log_dir, enabled=enabled)
        self._started = False
        self._ended = False
        self._turns = 0
        self._agent_sessions = 0
        self._current_turn: TurnHandle | None = None
        self._open_sessions: dict[str, AgentSessionRecord] = {}
        self._records: list[AgentSessionRecord] = []
        self._open_steps: list[StepHandle] = []

    # ---- session -----------------------------------------------------
    @property
    def path(self) -> Path:
        return self.log.path

    @property
    def ended(self) -> bool:
        return self._ended

    @property
    def current_turn_id(self) -> str | None:
        return self._current_turn.id if self._current_turn else None

    def start(self, device_id: str | None = None) -> None:
        if self._started:
            return
        self._started = True
        self.log.open()
        # 续用同一个 session id 时，先把上一进程留下的悬空边界补上，再开新的
        # session/start —— 顺序反了会让「turn/end」出现在一个没有 turn/start 的
        # session 里。
        recovered = self.recover()
        payload: dict[str, Any] = {}
        if device_id:
            payload["device"] = device_id
        if recovered:
            payload["resumed"] = True
        self.log.append("session/start", payload=payload)
        if self.log.enabled:
            logger.info(f"[runtime] session {self.id} -> {self.path}")

    def end(self, reason: str = "closed") -> None:
        if self._ended or not self._started:
            return
        if self._current_turn is not None:
            self._current_turn.end("cancelled", error=reason)
        self._ended = True
        self.log.append("session/end", payload={"reason": reason})
        self.log.close()

    # ---- 恢复 --------------------------------------------------------
    def recover(self) -> int:
        """补齐上一进程中断留下的悬空边界；返回补写的事件条数。

        只处理「有 start 无 end」的 turn / agent_session / step：不重跑任何逻辑，
        也不恢复屏幕。续跑以这些边界为起点，这里只保证事件流自洽。
        """
        events = self.log.read_events()
        if not events:
            return 0

        open_turns: list[str] = []
        open_steps: dict[str, Event] = {}
        open_sessions: dict[str, Event] = {}
        steps_of_session: dict[str, int] = {}
        for event in events:
            if event.type == "turn/start" and event.turn_id:
                open_turns.append(event.turn_id)
            elif event.type == "turn/end" and event.turn_id in open_turns:
                open_turns.remove(event.turn_id)
            elif event.type == "agent_session/start" and event.agent_session_id:
                open_sessions[event.agent_session_id] = event
            elif event.type == "agent_session/close":
                open_sessions.pop(event.agent_session_id or "", None)
            elif event.type == "step/start" and event.step_id:
                open_steps[event.step_id] = event
                if event.agent_session_id:
                    steps_of_session[event.agent_session_id] = (
                        steps_of_session.get(event.agent_session_id, 0) + 1
                    )
            elif event.type == "step/end":
                open_steps.pop(event.step_id or "", None)

        written = 0
        for turn_id in open_turns:
            for step in [e for e in open_steps.values() if e.turn_id == turn_id]:
                self.log.append(
                    "step/end",
                    turn_id=turn_id,
                    agent_session_id=step.agent_session_id,
                    step_id=step.step_id,
                    payload={
                        "agent": step.payload.get("agent"),
                        "outcome": "aborted",
                        "index": step.payload.get("index"),
                        "recovered": True,
                    },
                )
                written += 1
            for session in [e for e in open_sessions.values() if e.turn_id == turn_id]:
                self.log.append(
                    "agent_session/close",
                    turn_id=turn_id,
                    agent_session_id=session.agent_session_id,
                    payload={
                        "agent": session.payload.get("agent"),
                        "scope": session.payload.get("scope"),
                        "reason": "interrupted",
                        "steps": steps_of_session.get(session.agent_session_id or "", 0),
                        "recovered": True,
                    },
                )
                written += 1
            self.log.append(
                "turn/end",
                turn_id=turn_id,
                payload={"status": "cancelled", "error": "interrupted", "recovered": True},
            )
            written += 1
        return written

    # ---- turn --------------------------------------------------------
    def begin_turn(
        self,
        *,
        goal: str,
        max_steps: int | None = None,
        device_id: str | None = None,
    ) -> TurnHandle:
        self.start(device_id=device_id)
        if self._current_turn is not None and not self._current_turn.ended:
            # 正常路径由 _task_lock 保证串行；这里只是兜底，不让上一轮悬空。
            self._current_turn.end("cancelled", error="superseded by a new turn")
        self._turns += 1
        turn = TurnHandle(self, f"t_{self._turns:03d}", goal, max_steps)
        self._current_turn = turn
        self.log.append(
            "turn/start",
            turn_id=turn.id,
            payload={"goal": goal, **({"max_steps": max_steps} if max_steps else {})},
        )
        return turn

    # ---- agent session ----------------------------------------------
    def open_agent_session(
        self,
        agent: str,
        *,
        scope: str = "turn",
        seed: str | None = None,
    ) -> AgentSessionHandle:
        if self._current_turn is None:
            # 观测层不允许把任务弄挂：没有 turn 时返回一个不发事件的游离句柄。
            logger.warning(f"[runtime] opening {agent} session without an active turn")
            return AgentSessionHandle(
                self,
                AgentSessionRecord(
                    id=f"as_{agent}_detached",
                    turn_id="",
                    agent=agent,
                    scope=scope,
                    seed=seed,
                ),
            )
        existing = self._open_sessions.pop(agent, None)
        if existing is not None:
            self._finalize_agent_session(existing, reason="superseded", summary=None)
        self._agent_sessions += 1
        record = AgentSessionRecord(
            id=f"as_{agent}_{self._agent_sessions:03d}",
            turn_id=self._current_turn.id,
            agent=agent,
            scope=scope,
            seed=seed,
        )
        self._records.append(record)
        self._open_sessions[agent] = record
        self.log.append(
            "agent_session/start",
            turn_id=record.turn_id,
            agent_session_id=record.id,
            payload={"agent": agent, "scope": scope, **({"seed": seed} if seed else {})},
        )
        return AgentSessionHandle(self, record)

    def ensure_agent_session(self, agent: str, *, scope: str = "turn") -> AgentSessionHandle:
        record = self._open_sessions.get(agent)
        if record is None:
            return self.open_agent_session(agent, scope=scope)
        return AgentSessionHandle(self, record)

    def close_agent_session(
        self,
        agent: str,
        *,
        reason: str,
        summary: str | None = None,
    ) -> None:
        record = self._open_sessions.pop(agent, None)
        if record is not None:
            self._finalize_agent_session(record, reason=reason, summary=summary)

    def agent_sessions(self, agent: str | None = None) -> list[AgentSessionRecord]:
        if agent is None:
            return list(self._records)
        return [record for record in self._records if record.agent == agent]

    # ---- step -------------------------------------------------------
    def begin_step(
        self,
        agent: str,
        *,
        index: int | None = None,
        goal: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> StepHandle:
        session = self.ensure_agent_session(agent)
        return self._begin_step(record=session.record, index=index, goal=goal, extra=extra)

    def note_attempt(
        self,
        agent: str,
        *,
        reason: str | None = None,
        retried: bool = True,
        recovered: bool = False,
    ) -> None:
        """Step 内的 provider 重试。只记事实，不改 Step 状态。"""
        record = self._open_sessions.get(agent)
        if record is None:
            return
        self.log.append(
            "attempt/failed",
            turn_id=record.turn_id,
            agent_session_id=record.id,
            step_id=record.last_step_id,
            payload={
                "agent": agent,
                "retried": retried,
                "recovered": recovered,
                **({"reason": reason} if reason else {}),
            },
        )

    # ---- internals --------------------------------------------------
    def _begin_step(
        self,
        *,
        record: AgentSessionRecord,
        index: int | None,
        goal: str | None,
        extra: dict[str, Any] | None,
    ) -> StepHandle:
        step_index = index if index is not None else record.steps + 1
        step_id = f"{record.id}:st_{step_index:03d}"
        if not record.turn_id:
            # 游离会话（没有 active turn）：整条 Step 静默。
            return StepHandle(
                self,
                step_id=step_id,
                agent=record.agent,
                index=step_index,
                agent_session_id=record.id,
                enabled=False,
            )
        record.steps = max(record.steps, step_index)
        record.last_step_id = step_id
        payload: dict[str, Any] = {"agent": record.agent, "index": step_index}
        if goal:
            payload["goal"] = goal
        payload.update(extra or {})
        self.log.append(
            "step/start",
            turn_id=record.turn_id,
            agent_session_id=record.id,
            step_id=step_id,
            payload=payload,
        )
        handle = StepHandle(
            self,
            step_id=step_id,
            agent=record.agent,
            index=step_index,
            agent_session_id=record.id,
            record=record,
            meta=extra,
        )
        self._open_steps.append(handle)
        return handle

    def _close_open_steps(self, *, outcome: str) -> None:
        for step in list(self._open_steps):
            step.end(outcome)

    def _close_all_agent_sessions(self, *, reason: str) -> None:
        for record in list(self._open_sessions.values()):
            self._finalize_agent_session(record, reason=reason, summary=None)
        self._open_sessions.clear()

    def _close_agent_session(
        self, record: AgentSessionRecord, *, reason: str, summary: str | None
    ) -> None:
        if self._open_sessions.get(record.agent) is record:
            self._open_sessions.pop(record.agent, None)
        self._finalize_agent_session(record, reason=reason, summary=summary)

    def _finalize_agent_session(
        self,
        record: AgentSessionRecord,
        *,
        reason: str,
        summary: str | None,
    ) -> None:
        if record.status != "open":
            return
        record.status = "closed"
        record.close_reason = reason
        record.ended_at = _now()
        self.log.append(
            "agent_session/close",
            turn_id=record.turn_id,
            agent_session_id=record.id,
            payload={
                "agent": record.agent,
                "scope": record.scope,
                "reason": reason,
                "steps": record.steps,
                **({"summary": summary} if summary else {}),
                **({"history": record.history} if record.history else {}),
            },
        )
