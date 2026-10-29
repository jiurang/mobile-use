"""Bridge from LangGraph callbacks to the TUI event stream.

Why this exists
---------------
Every agent node calls its model with ``llm.ainvoke()`` -- never ``astream()``.
Token-level streaming is therefore **not** available, so a TUI cannot show
"tokens as they arrive". What *is* available through ``graph_config_callbacks``
(an existing, public field on ``AgentConfig``) is the chain lifecycle:

* ``on_chain_start`` / ``on_chain_end`` -- which graph node is running and how
  long it took. Node identity comes from ``metadata["langgraph_node"]``.
* the agent's own reasoning, carried on the node's state update as
  ``agents_thoughts`` (this is exactly what ``Agent._run_task`` reads off the
  ``updates`` stream and passes to ``log_agent_thought``).
* ``on_tool_start`` / ``on_tool_end`` -- the actual device actions.

That is enough for a live view, and it costs **zero changes to the agents or the
graph**.

Events are pushed onto a plain ``queue.SimpleQueue`` instead of touching
widgets: LangChain is free to dispatch callbacks from a worker thread, and this
keeps the handler free of any event-loop assumptions. The Textual side drains
the queue from its own worker.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler

from minitap.mobile_use.utils.llm_usage import extract_usage

# Graph node -> display label. Mirrors the `add_node` calls in graph/graph.py.
# Nodes outside this set (the graph itself, LangGraph internals, sub-chains
# inside a node such as structured-output parsers) are ignored on purpose.
#
# ⚠️ 这是一份**硬编码白名单**：`on_chain_start` 里 `node not in AGENT_LABELS`
# 就直接 return，所以名单里没有的节点在 TUI 里**完全不出现**（静默消失，
# 不是报错）。图中的节点必须都登记在这里，否则用户看不到那一步。
# （`reporter` 已于 2026-10-29 合并进 cortex，不再是图节点。）
AGENT_LABELS: dict[str, str] = {
    "planner": "planner",
    "plan_manager": "plan_manager",
    "contextor": "contextor",
    "cortex": "cortex",
    "executor": "executor",
    "executor_tools": "executor_tools",
}

# Nodes that actually drive the device; worth highlighting in the timeline.
ACTION_NODE = "executor_tools"


class EventKind(StrEnum):
    STATUS = "status"
    NODE_START = "node_start"
    NODE_END = "node_end"
    THOUGHT = "thought"
    ACTION_START = "action_start"
    ACTION_END = "action_end"
    RESULT = "result"
    ERROR = "error"
    LOG = "log"
    USAGE = "usage"


@dataclass(slots=True)
class TuiEvent:
    """One thing that happened, already reduced to what the UI needs."""

    kind: EventKind
    title: str
    detail: str = ""
    ok: bool = True
    duration_ms: int | None = None
    seq: int = 0
    payload: Any = None


def _short(value: Any, limit: int = 40) -> str:
    if isinstance(value, bool | int | float) or value is None:
        text = str(value)
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_tool_input(raw: Any) -> str:
    """Flatten tool arguments into a single compact line.

    Device tools carry coordinates, package names and selectors; showing the
    first few scalars is what makes an action trace legible at a glance.
    """
    if isinstance(raw, str):
        return _short(raw, 120)
    if not isinstance(raw, dict):
        return ""

    parts: list[str] = []
    for key, value in raw.items():
        rendered = _short(value)
        if rendered:
            parts.append(f"{key}={rendered}")
        if len(parts) >= 3:
            break
    return ", ".join(parts)


def extract_thought(outputs: Any) -> str | None:
    """Pull the agent's reasoning off a graph node's state update."""
    if not isinstance(outputs, dict):
        return None
    thoughts = outputs.get("agents_thoughts")
    if isinstance(thoughts, list) and thoughts:
        last = thoughts[-1]
        if isinstance(last, str) and last.strip():
            return last.strip()
    return None


def _extract_usage(response: Any) -> tuple[int, int] | None:
    """读取 LLM 响应的 token 用量。

    实现在 `utils/llm_usage.py`，与 telemetry 共用同一份 —— usage 的形状随
    langchain 版本和 provider 变化，是最容易漂移的地方，不能两处各写一遍。
    """
    return extract_usage(response)


class AgentEventBridge(BaseCallbackHandler):
    """LangChain callback handler that records agent activity for the TUI.

    Registered via ``AgentConfig.graph_config_callbacks``; see
    ``tui/session.py``. Failures inside the UI must never break a run, hence
    ``raise_error = False``.
    """

    raise_error = False

    def __init__(self) -> None:
        self._queue: queue.SimpleQueue[TuiEvent] = queue.SimpleQueue()
        self._seq = 0
        # run_id -> graph node, captured at start because metadata is not
        # repeated on the end event.
        self._nodes: dict[str, str] = {}
        # run_id -> monotonic start, for durations.
        self._started: dict[str, float] = {}
        # tool name per run_id, so the end event can name it too.
        self._tools: dict[str, str] = {}
        self._last_thought = ""
        self._tokens_in = 0
        self._tokens_out = 0
        self._usage_seen = False

    # ------------------------------------------------------------------
    # consumer side (Textual)
    # ------------------------------------------------------------------
    def drain(self) -> list[TuiEvent]:
        """Pop every event queued so far. Never blocks."""
        out: list[TuiEvent] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                return out

    def emit(self, kind: EventKind, title: str, **kwargs: Any) -> None:
        """Push an event onto the queue. Public so test/demo backends can
        synthesise the same stream the real agent produces."""
        self._seq += 1
        self._queue.put(TuiEvent(kind=kind, title=title, seq=self._seq, **kwargs))

    # Internal alias used by the callback methods below.
    _emit = emit

    # ------------------------------------------------------------------
    # token accounting
    # ------------------------------------------------------------------
    @property
    def total_tokens(self) -> tuple[int, int]:
        """Cumulative (input, output) tokens across every model call."""
        return self._tokens_in, self._tokens_out

    @property
    def usage_reported(self) -> bool:
        """False when the provider never reported usage.

        OpenAI-compatible endpoints may omit it entirely (the MiniMax endpoint
        this project defaults to does), so the UI must distinguish "zero
        tokens" from "unknown".
        """
        return self._usage_seen

    def reset_tokens(self) -> None:
        self._tokens_in = 0
        self._tokens_out = 0
        self._usage_seen = False

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        tokens = _extract_usage(response)
        if tokens is None:
            return
        prompt_tokens, completion_tokens = tokens
        if not (prompt_tokens or completion_tokens):
            return
        self._usage_seen = True
        self._tokens_in += prompt_tokens
        self._tokens_out += completion_tokens
        self._emit(EventKind.USAGE, "tokens", payload=(self._tokens_in, self._tokens_out))

    # ------------------------------------------------------------------
    # producer side (LangChain)
    # ------------------------------------------------------------------
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        node = (metadata or {}).get("langgraph_node")
        if node not in AGENT_LABELS:
            return
        key = str(run_id)
        self._nodes[key] = node
        self._started[key] = time.monotonic()
        self._emit(EventKind.NODE_START, AGENT_LABELS[node], payload=node)

    def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        node = self._nodes.pop(key, None)
        # `started` can legitimately be missing: the start event is dropped for
        # nodes we do not track, and LangGraph may end a run we never saw begin.
        started = self._started.pop(key, None)
        if node is None:
            return

        duration = int((time.monotonic() - started) * 1000) if started is not None else None
        self._emit(EventKind.NODE_END, AGENT_LABELS[node], duration_ms=duration, payload=node)

        thought = extract_thought(outputs)
        if thought and thought != self._last_thought:
            self._last_thought = thought
            self._emit(EventKind.THOUGHT, thought, payload=node)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: Any = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or "tool"
        key = str(run_id)
        self._started[key] = time.monotonic()
        self._tools[key] = name
        self._emit(EventKind.ACTION_START, name, detail=format_tool_input(inputs or input_str))

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        key = str(run_id)
        started = self._started.pop(key, None)
        name = self._tools.pop(key, "tool")
        duration = int((time.monotonic() - started) * 1000) if started is not None else None
        self._emit(
            EventKind.ACTION_END,
            name,
            detail=_short(output, 120),
            duration_ms=duration,
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        self._started.pop(key, None)
        name = self._tools.pop(key, "tool")
        self._emit(
            EventKind.ACTION_END,
            name,
            detail=_short(error, 120),
            ok=False,
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.emit(EventKind.ERROR, "llm error", detail=_short(error, 160), ok=False)
