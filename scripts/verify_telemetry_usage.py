"""验证 token 埋点在真实 LLM 调用链路上能跑通。

单测证明的是「给定一个假响应，能正确记录」。这里验证的是真实链路：
    invoke_llm_with_timeout_message -> 真实 provider -> telemetry.capture_llm_usage

不需要真机设备，只需要一个可用的 LLM 端点。跑法：
    .venv\\Scripts\\python.exe scripts\\verify_telemetry_usage.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minitap"))

from langchain_core.messages import HumanMessage  # noqa: E402

# 必须先 import config：settings 在这里做 .env 加载，晚于它取 key 会拿不到。
from minitap.mobile_use.config import settings  # noqa: E402,F401
from minitap.mobile_use.services import llm as llm_module  # noqa: E402
from minitap.mobile_use.services.telemetry import telemetry  # noqa: E402
from minitap.mobile_use.utils.logger import get_logger  # noqa: E402

logger = get_logger("verify_telemetry")


async def main() -> int:
    captured: list[dict] = []
    telemetry.capture_llm_usage = lambda **kw: captured.append(kw)  # type: ignore[method-assign]

    # 挑一个 .env 里真的有 key 的 provider。DeepSeek 是 OpenAI 兼容端点，
    # 且本仓库已接入（见 commit 5771430），最适合验证 usage 链路。
    from langchain_openai import ChatOpenAI  # noqa: E402

    from minitap.mobile_use.config import settings as _s  # noqa: E402

    model: Any
    name: str
    if _s.DEEPSEEK_API_KEY:
        model = ChatOpenAI(
            model="deepseek-chat",
            api_key=_s.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
            max_retries=1,
        )
        name = "deepseek/deepseek-chat"
    elif _s.GOOGLE_API_KEY:
        model = llm_module.get_google_llm(model="gemini-2.5-pro")
        name = "google/gemini-2.5-pro"
    else:
        print("没有可用的 API key。请在 .env 里设置 DEEPSEEK_API_KEY 或 GOOGLE_API_KEY。")
        return 2

    print(f"provider: {name}")
    print("发送一次最小请求（约 '1' 一词）...\n")

    messages = [HumanMessage(content="1")]
    result = await llm_module.invoke_llm_with_timeout_message(
        model.ainvoke(messages),
        agent_name="verify",
        task_id="verify-task",
    )

    print(f"响应类型: {type(result).__name__}")
    print(f"响应内容: {str(getattr(result, 'content', result))[:80]!r}")

    if not captured:
        print("\n未捕获到 usage 事件 —— 埋点没生效。")
        return 1

    ev = captured[0]
    print("\n捕获到的 usage 事件:")
    for k, v in ev.items():
        print(f"  {k:20s} {v}")

    if ev["input_tokens"] == 0 and ev["output_tokens"] == 0:
        print("\n注意：usage 为 0。可能是该 provider 不返回 usage，")
        print("或响应被 with_structured_output 包装过（那样就读不到了）。")
        print("这属于已知限制，不影响趋势分析。")
        return 1

    print(f"\nPASS: 真实调用记录到 input={ev['input_tokens']} output={ev['output_tokens']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
