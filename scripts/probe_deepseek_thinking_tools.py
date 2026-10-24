"""探针 #5：思考开关会不会影响「模型是否老实调用工具」。

为什么探这个
------------
DeepSeek 的思考模式**默认开着**，而思考模式下**不能强制 tool_choice**
（硬 400 "Thinking mode does not support this tool_choice"）⇒ 框架只能
「把工具绑上去、求模型自己调」。

而实测 2/20 的时候模型**自己编了个工具名**（``tool_call`` / ``Output`` /
``unknown``）⇒ 那一次调用就废了（解析层对不上名字 → StructuredOutputError）。

那问题就变成：**把思考关掉，模型会不会更老实？**
如果会，那这就是一条真修法（结构化输出这条路专门关思考）；
如果不会，就只能从「工具描述」和「框架容错」两头下手。

跑法：
    .\\.venv\\Scripts\\python.exe scripts/probe_deepseek_thinking_tools.py
"""

from __future__ import annotations

import asyncio
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
OUT_FILE = OUT_DIR / f"probe_deepseek_thinking_{date.today():%Y%m%d}.txt"

ROUNDS = 20
PROMPT = "Split 'buy milk' into subgoals."


def emit(line: str) -> None:
    print(line, flush=True)
    with OUT_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


async def run_arm(client, schema, expected: str, label: str) -> Counter:
    stats: Counter[str] = Counter()
    bound = client.bind_tools([schema], parallel_tool_calls=False)
    for i in range(ROUNDS):
        try:
            msg = await bound.ainvoke(PROMPT)
        except Exception as e:  # noqa: BLE001
            emit(f"  [{label}] #{i + 1} 请求失败 {type(e).__name__}: {str(e)[:160]}")
            stats["request_error"] += 1
            continue
        tcs = list(getattr(msg, "tool_calls", []) or [])
        names = [
            (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)) for tc in tcs
        ]
        usage = getattr(msg, "usage_metadata", None) or {}
        if not names:
            stats["no_tool_call"] += 1
            emit(
                f"  [{label}] #{i + 1} 没有 tool_calls "
                f"(finish={msg.response_metadata.get('finish_reason')!r})"
            )
            continue
        if all(n == expected for n in names):
            stats["ok"] += 1
        else:
            stats["wrong_name"] += 1
            emit(f"  [{label}] #{i + 1} ❌ 喊错名字 {names}")
        if usage:
            stats["out_tokens"] += int(usage.get("output_tokens") or 0)
    return stats


async def main() -> int:
    from pydantic import BaseModel, Field

    from minitap.mobile_use.services.llm import ProviderOverride, get_deepseek_llm

    class _Subgoal(BaseModel):
        goal: str = Field(description="the subgoal")
        reasoning: str = Field(description="why")

    class _PlannerOutput(BaseModel):
        subgoals: list[_Subgoal] = Field(description="all subgoals")

    key = os.environ.get("DEEPSEEK_API_KEY") or ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("", encoding="utf-8")

    emit(f"DeepSeek 思考开关 × 工具调用  {date.today()}  rounds/arm={ROUNDS}")
    if not key:
        emit("❌ 没有 DEEPSEEK_API_KEY")
        return 1

    arms = [
        ("思考=服务端默认(开)", ProviderOverride(), "_PlannerOutput"),
        ("思考=显式关闭", ProviderOverride(thinking=False), "_PlannerOutput"),
        ("思考=显式开启", ProviderOverride(thinking=True), "_PlannerOutput"),
    ]
    for label, override, expected in arms:
        client = get_deepseek_llm(model_name="deepseek-flash", override=override)
        emit("")
        emit(f"===== {label} =====")
        emit(f"  请求体 extra_body = {client.extra_body}")
        stats = await run_arm(client, _PlannerOutput, expected, label)
        n = stats.get("ok", 0) + stats.get("wrong_name", 0) + stats.get("no_tool_call", 0)
        avg = (stats.get("out_tokens", 0) / n) if n else 0
        emit(f"  ⇒ {label}: {dict(stats)}  平均输出 token≈{avg:.0f}")

    emit("")
    emit(f"证据落盘: {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
