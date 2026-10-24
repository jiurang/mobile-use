"""探针 #3：把 DeepSeek 结构化输出的失败形态**分类**。

为什么这么探
------------
``test_deepseek_structured_output`` 是打真实 API 的 integration 测试，随机挂。
失败信息里有两种形态，但它们在**不同层**，修法完全不同：

* ``tool_calls=0, wrong_tools=[], finish_reason='tool_calls', content=''``
  —— 服务端说「调了工具」，langchain 却一个都没解析出来。
  ⇒ 工具调用**被 langchain 丢进了** ``msg.invalid_tool_calls``（**顶层属性**，
  不在 ``additional_kwargs`` 里 —— 我第一次就查错了地方，所以量出来是 0）。
* ``finish_reason='stop'`` + 没有 tool_calls
  —— 模型这次直接写了散文/空，没调工具。

本探针**一次 API 调用**同时拿到两样东西：
1. 原始 ``AIMessage``（``tool_calls`` / ``invalid_tool_calls`` / ``content`` / ``finish_reason``）；
2. 框架自己的 ``_parse`` 对同一条消息的结果（成功 or StructuredOutputError）。

⇒ 失败形态、频率、以及「invalid_tool_calls 里到底躺着什么」一次性对齐。

跑法：
    .\\.venv\\Scripts\\python.exe scripts/probe_deepseek_toolcall_shape.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

OUT_DIR = ROOT / "probe_out"
OUT_FILE = OUT_DIR / f"probe_deepseek_shape_{date.today():%Y%m%d}.txt"

ROUNDS = 20
PROMPT = "Split 'buy milk' into subgoals."


def emit(line: str) -> None:
    print(line, flush=True)
    with OUT_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


async def main() -> int:
    from pydantic import BaseModel, Field

    from minitap.mobile_use.services.llm import _ToolCallStructuredRunnable, get_deepseek_llm

    class _Subgoal(BaseModel):
        goal: str = Field(description="the subgoal")
        reasoning: str = Field(description="why")

    class _PlannerOutput(BaseModel):
        subgoals: list[_Subgoal] = Field(description="all subgoals")

    key = os.environ.get("DEEPSEEK_API_KEY") or ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("", encoding="utf-8")

    emit(f"DeepSeek 结构化输出失败形态分类  {date.today()}  rounds={ROUNDS}")
    if not key:
        emit("❌ 没有 DEEPSEEK_API_KEY")
        return 1

    client = get_deepseek_llm(model_name="deepseek-flash")
    # 复刻 executor/reporter 走的那条路：schema 当工具绑 + 自己解析。
    runner = _ToolCallStructuredRunnable(client, _PlannerOutput, force_tool_choice=False)
    bound = runner._bound

    taxonomy: Counter[str] = Counter()
    for i in range(ROUNDS):
        try:
            msg = await bound.ainvoke(PROMPT)
        except Exception as e:  # noqa: BLE001
            emit(f"--- #{i + 1} ❌ 请求失败 {type(e).__name__}: {str(e)[:200]}")
            taxonomy["request_error"] += 1
            continue

        meta = getattr(msg, "response_metadata", {}) or {}
        finish = meta.get("finish_reason")
        tcs = list(getattr(msg, "tool_calls", []) or [])
        # ⚠️ invalid_tool_calls 是 AIMessage 的**顶层属性**。
        invalid = list(getattr(msg, "invalid_tool_calls", []) or [])
        content = msg.content if isinstance(msg.content, str) else ""
        names = [
            (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)) for tc in tcs
        ]
        inv_names = [
            (itc.get("name") if isinstance(itc, dict) else getattr(itc, "name", None))
            for itc in invalid
        ]

        # 框架自己的解析结果（不额外发请求）。
        try:
            runner._parse(msg)
            outcome = "parse_ok"
            detail = ""
        except Exception as e:  # noqa: BLE001
            outcome = "parse_error"
            detail = str(e)[:220]

        if outcome == "parse_ok":
            taxonomy["parse_ok"] += 1
        elif invalid and not tcs:
            taxonomy["parse_error__only_invalid_tool_calls"] += 1
        elif not tcs and finish == "stop":
            taxonomy["parse_error__no_tool_calls_finish_stop"] += 1
        elif tcs and all(n != runner._tool_name for n in names):
            taxonomy["parse_error__wrong_tool_name"] += 1
        else:
            taxonomy["parse_error__other"] += 1

        emit("")
        emit(f"--- #{i + 1} finish_reason={finish!r} outcome={outcome}")
        emit(f"  tool_calls n={len(tcs)} names={names}")
        emit(f"  invalid_tool_calls n={len(invalid)} names={inv_names}")
        if invalid:
            first = invalid[0]
            raw = first.get("raw" if isinstance(first, dict) else "raw", None)
            emit(f"  invalid[0] = {json.dumps(first, ensure_ascii=False, default=str)[:500]}")
            if raw:
                emit(f"  invalid[0].raw = {json.dumps(raw, ensure_ascii=False)[:400]}")
        emit(f"  content[:200]={content[:200]!r}")
        if detail:
            emit(f"  parse 错误: {detail}")

    emit("")
    emit("=== 失败形态统计 ===")
    for k, v in taxonomy.most_common():
        emit(f"  {k}: {v}/{ROUNDS}")
    emit(f"证据落盘: {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
