"""探针 #2：MiMo 的「思考 × 工具调用」矩阵 —— 定死渠道默认该怎么配。

为什么单独探
------------
探针 #1 查出一件事：MiMo **不下发** ``thinking`` 时，服务端默认是**开着思考**的
（usage 里 ``reasoning_tokens`` 非零、message 里有 ``reasoning_content``）。

而渠道目录里写的是 ``thinking_default_on=False`` ⇒ 框架以为「没说 = 没开思考」
⇒ 允许强制 tool_choice ⇒ 实测 tool_calls **没有**、``finish_reason=length``。

但探针 #1 用的 ``max_completion_tokens=64`` 太小，推理链可能直接把预算吃光了
⇒ 「工具调用失败」到底是**必然**的，还是**只是预算不够**？

这个区别决定了修法：
* 必然失败 ⇒ 必须让渠道默认把思考**显式关掉**（thinking_fallback=False）。
* 只是预算不够 ⇒ 可以留思考开着，靠不强制 tool_choice 兜。

所以这一轮把预算放大到 2048，跑满 4 种组合，每种重复 3 次看稳定性。

跑法：
    .\\.venv\\Scripts\\python.exe scripts/probe_mimo_thinking.py
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
OUT_FILE = OUT_DIR / f"probe_mimo_thinking_{date.today():%Y%m%d}.txt"

BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MODEL = "mimo-v2.6-pro"
BUDGET = 2048
ROUNDS = 3

TOOL = {
    "type": "function",
    "function": {
        "name": "report",
        "description": "上报一个结果",
        "parameters": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}, "note": {"type": "string"}},
            "required": ["ok"],
        },
    },
}


def emit(line: str) -> None:
    print(line, flush=True)
    with OUT_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


async def one(c, key: str, thinking: str | None, force: bool) -> dict:
    payload: dict = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "调用 report 工具，ok 设为 true"}],
        "max_completion_tokens": BUDGET,
        "tools": [TOOL],
    }
    if thinking is not None:
        payload["thinking"] = {"type": thinking}
    if force:
        payload["tool_choice"] = {"type": "function", "function": {"name": "report"}}

    t0 = time.perf_counter()
    try:
        r = await c.post(
            f"{BASE_URL}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
    except Exception as e:  # noqa: BLE001
        return {"err": f"{type(e).__name__}: {e}", "ms": int((time.perf_counter() - t0) * 1000)}

    ms = int((time.perf_counter() - t0) * 1000)
    try:
        body = r.json()
    except Exception:  # noqa: BLE001
        return {"status": r.status_code, "err": r.text[:200], "ms": ms}
    if r.status_code != 200:
        return {
            "status": r.status_code,
            "err": json.dumps(body, ensure_ascii=False)[:200],
            "ms": ms,
        }

    first = (body.get("choices") or [{}])[0]
    msg = first.get("message") or {}
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "status": 200,
        "ms": ms,
        "tool_call": bool(msg.get("tool_calls")),
        "finish": first.get("finish_reason"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "content": (msg.get("content") or "")[:120],
    }


async def main() -> int:
    import httpx

    key = os.environ.get("MIMO_API_KEY") or ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("", encoding="utf-8")

    emit(f"MiMo 思考×工具矩阵  {date.today()}  model={MODEL} budget={BUDGET} rounds={ROUNDS}")
    if not key:
        emit("❌ 没有 MIMO_API_KEY")
        return 1

    matrix = [
        ("thinking=enabled  + 强制 tool_choice", "enabled", True),
        ("thinking=enabled  + 不强制 tool_choice", "enabled", False),
        ("thinking=disabled + 强制 tool_choice", "disabled", True),
        ("thinking=disabled + 不强制 tool_choice", "disabled", False),
        ("thinking 不下发  + 强制 tool_choice", None, True),
        ("thinking 不下发  + 不强制 tool_choice", None, False),
    ]

    async with httpx.AsyncClient(timeout=180, follow_redirects=True) as c:
        for label, thinking, force in matrix:
            emit("")
            emit(f"--- {label} ---")
            hits = 0
            for i in range(ROUNDS):
                res = await one(c, key, thinking, force)
                if res.get("err"):
                    emit(f"  #{i + 1} ❌ {res['err'][:180]}")
                    continue
                if res.get("status") != 200:
                    emit(f"  #{i + 1} ❌ HTTP {res['status']} {res.get('err', '')[:180]}")
                    continue
                tc = res["tool_call"]
                hits += 1 if tc else 0
                emit(
                    f"  #{i + 1} {'✅' if tc else '❌'} tool_call={'有' if tc else '无'} "
                    f"finish={res['finish']} {res['ms']}ms "
                    f"reasoning_tok={res['reasoning_tokens']} "
                    f"completion_tok={res['completion_tokens']}"
                )
                if not tc and res.get("content"):
                    emit(f"        content={res['content']!r}")
            emit(f"  ⇒ 工具调用命中 {hits}/{ROUNDS}")

    emit("")
    emit(f"证据落盘: {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
