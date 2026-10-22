"""独立复核：`llm_usage` 埋点在 LangGraph 图里到底通不通。

背景
----
仓库里到处写着一条结论：

    「在 LangGraph 的图里那个捕获槽根本收不到东西（ContextVar 跨不过图的
     任务上下文），图里的 usage 由 `services/usage_bridge.py` 的 graph-config
     callback 上报。」

`services/usage_bridge.py` **从未存在**（git 全历史无此文件，也没有任何
import）。这条结论是从 docstring 里抄来的，**没被实测验证过**。

而真机 graph 实测（`probe_out/executor_e2e.json`，2026-10-21）显示
executor 21 次调用全是真实非零 token —— 与那条结论矛盾。

本探针就是把这个矛盾一次性钉死：**不碰设备、不调真 LLM**，只用一个
最小 LangGraph 复刻「节点里调 LLM」这个形状，看 usage 能不能上报。

三种形态各测一遍：
  A. 节点内直接调（不是 graph）—— 基线，必须通
  B. 节点内经 LangGraph 调 —— 争议点
  C. 节点内经 LangGraph 调，且**只有原始响应**带 usage
     （模拟 with_structured_output：返回的是 pydantic 对象，usage 不在它身上）

C 是真正要证明的那一格：如果 C 也能拿到非零 token，说明
「捕获槽跨不过图」是假的。

跑法：uv run python v/probe_usage_bridge.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from minitap.mobile_use.services import telemetry as telemetry_mod  # noqa: E402
from minitap.mobile_use.services.llm import invoke_llm_with_timeout_message  # noqa: E402
from minitap.mobile_use.utils.llm_usage import record_raw_response  # noqa: E402

CAPTURED: list[dict[str, Any]] = []


class _State(TypedDict):
    value: int


def _install_probe() -> None:
    original = telemetry_mod.telemetry.capture_llm_usage

    def probe(**kwargs: Any) -> None:
        CAPTURED.append(dict(kwargs))
        return original(**kwargs)

    telemetry_mod.telemetry.capture_llm_usage = probe  # type: ignore[method-assign]


def _fake_raw_response() -> AIMessage:
    """一条带 usage 的原始响应 —— 模拟 provider 回来的 AIMessage。"""
    return AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 1234, "output_tokens": 56, "total_tokens": 1290},
    )


async def _llm_call_returns_pydantic_like() -> Any:
    """模拟 `with_structured_output`：返回体**身上没有** usage。

    usage 只在**原始响应**里（provider 的真实形状就是这样：
    结构化输出把原始 AIMessage 藏起来了）。
    """
    record_raw_response(_fake_raw_response())
    # 返回一个既没有 usage_metadata 也没有 llm_output 的对象 ——
    # 和 pydantic 结构化输出一样，读不到任何 usage。
    return {"parsed": "yes"}


async def _llm_call_with_usage_on_result() -> Any:
    """模拟直接返回带 usage_metadata 的 AIMessage。"""
    record_raw_response(_fake_raw_response())
    return _fake_raw_response()


async def _case_a() -> None:
    """A. 不经 graph，直接在当前任务里调。"""
    await invoke_llm_with_timeout_message(
        _llm_call_returns_pydantic_like(),
        agent_name="probe_direct",
        task_id="t",
    )


async def _case_b() -> None:
    """B/C. 经 LangGraph 的节点调。

    两种 LLM 形态各跑一遍（B：结果自带 usage；C：usage 只在原始响应里）。
    """

    async def node_b(state: _State) -> _State:
        await invoke_llm_with_timeout_message(
            _llm_call_with_usage_on_result(),
            agent_name="probe_graph",
            task_id="t",
        )
        return {"value": state["value"] + 1}

    async def node_c(state: _State) -> _State:
        await invoke_llm_with_timeout_message(
            _llm_call_returns_pydantic_like(),
            agent_name="probe_graph_structured",
            task_id="t",
        )
        return {"value": state["value"] + 1}

    for name, node in (("B", node_b), ("C", node_c)):
        g = StateGraph(_State)
        g.add_node("n", node)
        g.add_edge(START, "n")
        g.add_edge("n", END)
        app = g.compile()
        await app.ainvoke({"value": 0}, config={"recursion_limit": 10})


async def main() -> int:
    _install_probe()

    await _case_a()
    a_n = len(CAPTURED)
    await _case_b()
    b_n = len(CAPTURED)

    print("=" * 68)
    print("llm_usage 埋点在 LangGraph 图里通不通 —— 独立复核")
    print("=" * 68)
    for c in CAPTURED:
        print(
            f"  agent={c.get('agent'):<26} in={c.get('input_tokens'):>6} "
            f"out={c.get('output_tokens'):>5} cached={c.get('cached_tokens'):>6}"
        )
    print()

    ok = True
    if a_n != 1:
        print(f"❌ A 直接调用：预期 1 条上报，实到 {a_n} 条")
        ok = False
    else:
        print("✅ A 直接调用：上报 1 条")

    if b_n - a_n != 2:
        print(f"❌ B/C 图内调用：预期 2 条上报，实到 {b_n - a_n} 条")
        ok = False
    else:
        print("✅ B/C 图内调用：上报 2 条 —— 捕获槽跨得过 graph 边界")

    for c in CAPTURED:
        if c.get("input_tokens", 0) <= 0:
            print(f"❌ agent={c.get('agent')} 的 input_tokens 是 0 —— 埋点记了 0")
            ok = False

    if ok:
        print()
        print(
            "结论：那条「ContextVar 跨不过图 ⇒ 图内 usage 记 0」的结论是**假的**。\n"
            "      捕获槽里装的是同一个 list 对象，LangGraph 的节点任务继承父 context，\n"
            "      所以图里的原始响应外面读得到。**不要造 usage_bridge.py。**"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
