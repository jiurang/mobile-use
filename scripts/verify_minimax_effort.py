"""验证 `MINIMAX_EFFORT` 真的从 .env 读进来，并且真的进到请求体里。

为什么单独写这个
----------------
单元测试用 monkeypatch 直接改 `settings.MINIMAX_EFFORT`，绕过了 **.env 读取**。
配置项最典型的静默失效就是「代码里接上了、默认值也对，但 .env 里的值根本没被
读到」—— 症状是「改了配置没反应」，且没有任何报错。

所以这里走完整链路：真实 .env -> Settings -> get_minimax_llm -> extra_body。
不发真实请求（端到端由 bench_run.py 负责），只断言最终请求体。

跑法：
    uv run python scripts/verify_minimax_effort.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))

from minitap.mobile_use.config import settings  # noqa: E402
from minitap.mobile_use.services.llm import get_minimax_llm  # noqa: E402


def main() -> int:
    raw_env = os.environ.get("MINIMAX_EFFORT")
    print("=== 链路逐段核对 ===")
    print(f"1. os.environ['MINIMAX_EFFORT']      = {raw_env!r}   （进程环境变量）")
    print(f"2. settings.MINIMAX_EFFORT           = {settings.MINIMAX_EFFORT!r}   （.env 解析后）")
    print(f"3. Settings 字段声明存在              = "
          f"{'MINIMAX_EFFORT' in type(settings).model_fields}")

    if not settings.MINIMAX_API_KEY:
        print("FAIL 没有 MINIMAX_API_KEY，无法构造客户端")
        return 1

    client = get_minimax_llm("MiniMax-M3.1-Flash-Preview", temperature=1)
    print(f"4. client.extra_body                 = {client.extra_body!r}")
    print(f"5. client.model_name                 = {client.model_name!r}")

    problems = []
    if client.extra_body != {"output_config": {"effort": "max"}}:
        problems.append(f"extra_body 不是预期的 output_config.effort=max，实际 {client.extra_body!r}")
    if client.model_name != "MiniMax-M3.1-Flash-Preview":
        problems.append(f"模型名不对：{client.model_name!r}")
    if "thinking" in (client.extra_body or {}):
        problems.append("请求体里带 thinking —— M3.1 上发 disabled 会被官方 400 拒")

    print()
    if problems:
        print("FAIL：")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("PASS  MINIMAX_EFFORT=max 从 .env 一路走到请求体，且没有多余的 thinking 字段")
    print("      最终请求体片段："
          + json.dumps(client.extra_body, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
