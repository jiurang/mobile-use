"""探针：DeepSeek 结构化输出的**原始响应**长什么样。

为什么单独探
------------
``tests/mobile_use/test_deepseek_provider.py::test_deepseek_structured_output``
是打**真实 DeepSeek API** 的 integration 测试，重跑会随机挂（4 跑 3 挂），
两种失败形态完全不同：

A. ``tool_calls=0, wrong_tools=[], finish_reason='tool_calls', content=''``
   —— 服务端说「我调工具了」，langchain 却一个都没解析出来。
B. ``tool_calls=1, wrong_tools=['function_call'], finish_reason='tool_calls'``
   —— 模型喊了一个**叫 "function_call"** 的工具名，不是 schema 的名字。

A 和 B 的修法完全不同（A 像是解析层漏了字段，B 是模型行为/工具名问题），
所以必须先看**原始 JSON**，不能从 langchain 的 AIMessage 反推。

本探针做两件事：
1. 用 ``_get_request_payload`` 把框架**真正发出去的 body** 截下来；
2. 自己用 httpx 原样打给 DeepSeek，把**原始响应 JSON** 落盘。

跑法：
    .\\.venv\\Scripts\\python.exe scripts/probe_deepseek_structured_raw.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
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
OUT_FILE = OUT_DIR / f"probe_deepseek_raw_{date.today():%Y%m%d}.txt"

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
ROUNDS = 8
PROMPT = "Split 'buy milk' into subgoals."


def emit(line: str) -> None:
    print(line, flush=True)
    with OUT_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


class _Stop(Exception):
    pass


def capture_payload(client, schema) -> dict:
    """截下框架真正会发出去的请求体。"""
    from langchain_openai.chat_models import base as lc_base

    captured: dict = {}
    original = lc_base.BaseChatOpenAI._get_request_payload

    def spy(self, input_, **kwargs):
        captured.update(original(self, input_, **kwargs))
        raise _Stop

    bound = client.bind_tools([schema], parallel_tool_calls=False)
    lc_base.BaseChatOpenAI._get_request_payload = spy
    try:
        bound.invoke(PROMPT)
    except _Stop:
        pass
    finally:
        lc_base.BaseChatOpenAI._get_request_payload = original
    return captured


async def main() -> int:
    import httpx
    from pydantic import BaseModel, Field

    from minitap.mobile_use.services.llm import get_deepseek_llm

    class _Subgoal(BaseModel):
        goal: str = Field(description="the subgoal")
        reasoning: str = Field(description="why")

    class _PlannerOutput(BaseModel):
        subgoals: list[_Subgoal] = Field(description="all subgoals")

    key = os.environ.get("DEEPSEEK_API_KEY") or ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("", encoding="utf-8")

    emit(f"DeepSeek 结构化输出原始响应探针  {date.today()}  model={MODEL} rounds={ROUNDS}")
    if not key:
        emit("❌ 没有 DEEPSEEK_API_KEY")
        return 1

    client = get_deepseek_llm(model_name=MODEL)
    payload = capture_payload(client, _PlannerOutput)
    # 截到的 body 里没有 api_key，补上头即可。
    body = {k: v for k, v in payload.items() if k not in ("api_key", "api-key", "default_headers")}
    emit("")
    emit("=== 框架实际发出的请求体 ===")
    emit(json.dumps(body, ensure_ascii=False, indent=2)[:2000])

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    stats = {"tool_calls": 0, "empty": 0, "wrong_name": 0, "ok": 0, "err": 0}

    async with httpx.AsyncClient(timeout=120) as c:
        for i in range(ROUNDS):
            t0 = time.perf_counter()
            try:
                r = await c.post(DEEPSEEK_URL, json=body, headers=headers)
            except Exception as e:  # noqa: BLE001
                emit(f"\n--- #{i + 1} ❌ {type(e).__name__}: {e}")
                stats["err"] += 1
                continue
            ms = int((time.perf_counter() - t0) * 1000)
            try:
                resp = r.json()
            except Exception:  # noqa: BLE001
                emit(f"\n--- #{i + 1} HTTP {r.status_code} 非 JSON: {r.text[:300]}")
                stats["err"] += 1
                continue

            emit(f"\n--- #{i + 1} HTTP {r.status_code} {ms}ms ---")
            if r.status_code != 200:
                emit(f"  body: {json.dumps(resp, ensure_ascii=False)[:500]}")
                stats["err"] += 1
                continue

            choices = resp.get("choices") or [{}]
            first = choices[0]
            msg = first.get("message") or {}
            tcs = msg.get("tool_calls")
            usage = resp.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}

            emit(f"  finish_reason={first.get('finish_reason')!r}")
            emit(f"  usage={json.dumps(usage, ensure_ascii=False)}")
            emit(
                f"  completion_tokens_details={json.dumps(details, ensure_ascii=False)}"
            )
            emit(f"  message.content={json.dumps(msg.get('content'), ensure_ascii=False)[:300]}")
            emit(
                f"  message.reasoning_content="
                f"{json.dumps(msg.get('reasoning_content'), ensure_ascii=False)[:300]}"
            )
            emit(f"  message.tool_calls 原始 = {json.dumps(tcs, ensure_ascii=False)[:800]}")

            if tcs:
                stats["tool_calls"] += 1
                names = [tc.get("function", {}).get("name") for tc in tcs]
                emit(f"  ⇒ 工具名 = {names}")
                if any(n != "_PlannerOutput" for n in names):
                    stats["wrong_name"] += 1
                else:
                    stats["ok"] += 1
            else:
                stats["empty"] += 1
                emit("  ⇒ 没有 tool_calls（finish_reason 却可能是 tool_calls ⇒ 解析层要背锅）")

    emit("")
    emit(f"统计: {stats}  （ok=名字正确, wrong_name=喊错名字, empty=没有 tool_calls）")
    emit(f"证据落盘: {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
