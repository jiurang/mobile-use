"""验证：DeepSeek 走 executor 那条路（``get_structured_llm``）不再 400。

为什么单独探这个
----------------
「deepseek 一直在请求失败」的直接原因在这条路上，而且**只有**这条路上：

* planner / cortex / orchestrator / contextor / outputter / hopper 走
  ``llm.with_structured_output(...)`` —— 那条路会问客户端「这次思考开着没有」
  （旧版只读 .env 的 ``DEEPSEEK_THINKING``）。
* **executor / reporter 走 ``get_structured_llm``** —— 那条路以前**恒定**
  ``force_tool_choice=True``，而 DeepSeek 的推理模型默认开着思考，于是：

      400 invalid_request_error: Thinking mode does not support this tool_choice

* 而渠道下发的 ``thinking`` 又被**完全无视**（只读 .env），所以用户在 manager 里
  把思考关掉也不管用。

跑法：
    .\\.venv\\Scripts\\python.exe scripts/probe_deepseek_executor_path.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from types import SimpleNamespace  # noqa: E402


async def _main() -> int:
    from minitap.mobile_use.agents.executor.types import ExecutorOutput
    from minitap.mobile_use.sdk.utils import llm_config_for_single_model
    from minitap.mobile_use.services.llm import get_structured_llm

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("DEEPSEEK_API_KEY 没配，跳过。")
        return 1

    print("=" * 78)
    print("executor 那条路（get_structured_llm）打真实 DeepSeek")
    print("=" * 78)

    for thinking in (None, False):
        cfg = llm_config_for_single_model(
            "deepseek",
            "deepseek-flash",
            api_key=key,
            thinking=thinking,
        )
        ctx = SimpleNamespace(llm_config=cfg, trace_id="probe-1")
        runner = get_structured_llm(ctx=ctx, name="executor", schema=ExecutorOutput)
        forced = "tool_choice" in runner._bound.kwargs
        print(f"\n  渠道 thinking={thinking!r:<6} → 强制 tool_choice={forced}")

        try:
            result = await runner.ainvoke(
                "屏幕上有一个「搜索」按钮。点它，然后结束。",
            )
        except Exception as e:  # noqa: BLE001
            print(f"    ❌ {type(e).__name__}: {str(e)[:260]}")
            continue

        print(f"    ✅ 解析成功：{type(result).__name__}，动作数={len(result.actions or [])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
