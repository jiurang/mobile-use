"""探针 #4：工具名带下划线前缀 / 描述为空 —— 哪个才是模型喊错名字的元凶？

为什么探这个
------------
20 轮实测（``probe_deepseek_toolcall_shape.py``）：DeepSeek 有 2/20 喊了
**编出来的工具名**（``tool_call`` / ``Output`` / ``unknown``），而模型自己的
reasoning 里写的是「我该用 PlannerOutput 工具」—— 它知道该叫什么，只是**输出错了**。

两个可疑变量：

A. **工具名带下划线前缀**（测试 fixture ``_PlannerOutput``；生产里的
   ``PlannerOutput`` / ``ExecutorOutput`` / ``CortexOutput`` … 都没带）
B. **工具描述是空的**（pydantic 类的 docstring 才是描述；生产里 10 个 agent
   schema 有 6 个没有 docstring ⇒ 描述为空字符串）

本探针跑三组，每组 N 轮，**只变一个变量**：

* arm1 ``_PlannerOutput``：下划线 + 无描述   （= 现在的测试 fixture）
* arm2 ``PlannerOutput`` ：无下划线 + 无描述 （= 现在的生产 schema）
* arm3 ``PlannerOutputDoc``：无下划线 + 有描述 （= 加上 docstring 之后）

⇒ 直接得出「该改哪一处」。

跑法：
    .\\.venv\\Scripts\\python.exe scripts/probe_deepseek_toolname.py
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
OUT_FILE = OUT_DIR / f"probe_deepseek_toolname_{date.today():%Y%m%d}.txt"

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
            emit(f"  [{label}] #{i + 1} ❌ 喊错名字 {names} (期望 {expected})")
    return stats


async def main() -> int:
    from pydantic import BaseModel, Field

    from minitap.mobile_use.services.llm import get_deepseek_llm

    class _Subgoal(BaseModel):
        goal: str = Field(description="the subgoal")
        reasoning: str = Field(description="why")

    class _PlannerOutput(BaseModel):
        subgoals: list[_Subgoal] = Field(description="all subgoals")

    class PlannerOutput(BaseModel):
        subgoals: list[_Subgoal] = Field(description="all subgoals")

    class PlannerOutputDoc(BaseModel):
        """把用户的目标拆成若干个可执行的子目标。

        每个子目标写清楚「要做什么」以及「为什么这样拆」。
        """

        subgoals: list[_Subgoal] = Field(description="all subgoals")

    key = os.environ.get("DEEPSEEK_API_KEY") or ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("", encoding="utf-8")

    emit(f"DeepSeek 工具名/描述对照  {date.today()}  rounds/arm={ROUNDS}")
    if not key:
        emit("❌ 没有 DEEPSEEK_API_KEY")
        return 1

    client = get_deepseek_llm(model_name="deepseek-flash")
    arms = [
        ("arm1 下划线+无描述", _PlannerOutput, "_PlannerOutput"),
        ("arm2 无下划线+无描述", PlannerOutput, "PlannerOutput"),
        ("arm3 无下划线+有描述", PlannerOutputDoc, "PlannerOutputDoc"),
    ]
    for label, schema, expected in arms:
        emit("")
        emit(f"===== {label} =====")
        stats = await run_arm(client, schema, expected, label)
        emit(f"  ⇒ {label}: {dict(stats)}")

    emit("")
    emit(f"证据落盘: {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
