"""事件与 JSONL 日志。

一条事件 = 一次边界。事件只追加、不修改；``seq`` 在 Session 内单调。

日志是**尽力而为**的：磁盘不可写、序列化失败都只 warning 一次，然后把这一层
降级成 no-op。会话记录是观测手段，不能因为写日志把任务弄挂。
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from minitap.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

_DISABLED_VALUES = frozenset({"0", "false", "off", "no"})


class Event(BaseModel):
    """一条会话事件。"""

    seq: int
    ts: datetime
    type: str
    session_id: str
    turn_id: str | None = None
    agent_session_id: str | None = None
    step_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


def session_log_enabled() -> bool:
    """``MOBILE_USE_SESSION_LOG`` 显式关闭时才不写；默认开。"""
    value = (os.environ.get("MOBILE_USE_SESSION_LOG") or "").strip().lower()
    return value not in _DISABLED_VALUES


def session_log_root() -> Path:
    """日志根目录：默认当前工作目录下的 ``.mobile-use/sessions``。"""
    override = (os.environ.get("MOBILE_USE_SESSION_LOG_DIR") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path.cwd() / ".mobile-use" / "sessions"


class SessionLog:
    """Session 的事件日志：内存一份、磁盘一份。"""

    def __init__(
        self,
        *,
        session_id: str,
        directory: Path | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.session_id = session_id
        self.enabled = session_log_enabled() if enabled is None else enabled
        self._directory = directory or session_log_root()
        self._path = self._directory / session_id / "events.jsonl"
        # RLock：追加路径在持锁时可能触发降级关闭，降级又要拿同一把锁。
        self._lock = threading.RLock()
        self._seq = 0
        self._events: list[Event] = []
        self._handle = None
        self._broken = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def open(self) -> None:
        if not self.enabled or self._broken or self._handle is not None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("a", encoding="utf-8")
        except OSError as exc:
            self._degrade(f"cannot open session log {self._path}: {exc}")
            return
        # 续用同一个 session id 时（进程重启 / 多轮），seq 必须接着上次的最大值，
        # 否则重启后会出现两个 seq=1，事件流不再单调。
        self._seq = max(self._seq, self._last_seq_on_disk())

    def _last_seq_on_disk(self) -> int:
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return self._seq
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                return int(json.loads(line)["seq"])
            except (ValueError, KeyError, TypeError):
                continue
        return self._seq

    def read_events(self) -> list[Event]:
        """读回磁盘上的全部事件（用于恢复上一进程留下的边界）。"""
        if not self.enabled:
            return []
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError:
            return []
        events: list[Event] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(Event.model_validate_json(line))
            except ValueError:
                continue
        return events

    def append(
        self,
        event_type: str,
        *,
        turn_id: str | None = None,
        agent_session_id: str | None = None,
        step_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Event | None:
        """追加一条事件。返回写入的事件；关闭或降级时返回 None。"""
        if not self.enabled or self._broken:
            return None
        with self._lock:
            self._seq += 1
            event = Event(
                seq=self._seq,
                ts=datetime.now(UTC),
                type=event_type,
                session_id=self.session_id,
                turn_id=turn_id,
                agent_session_id=agent_session_id,
                step_id=step_id,
                payload=payload or {},
            )
            self._events.append(event)
            if self._handle is None:
                self.open()
            if self._handle is not None:
                try:
                    self._handle.write(event.model_dump_json() + "\n")
                    self._handle.flush()
                except (OSError, ValueError, TypeError) as exc:
                    self._degrade(f"cannot write session log {self._path}: {exc}")
            return event

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.close()
                except OSError:
                    pass
                self._handle = None

    def _degrade(self, message: str) -> None:
        self._broken = True
        try:
            self.close()
        finally:
            logger.warning(f"[runtime] {message}; session log disabled for this run")

    @staticmethod
    def dump(event: Event) -> str:
        """测试与排查用的单条序列化。"""
        return json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
