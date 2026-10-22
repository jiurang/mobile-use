"""**Executor 一次调用的延迟分解** —— 时间到底花在哪个块上。

问题
----
真机实测：executor 每次决策中位 **4,143ms**（第七轮，27 次）。
提速要下手，先得知道这 4.1 秒花在哪：

* **输入侧**：元素表（真机 113 行 → 2,494 tok）？静态提示词 `executor.md`
  （2,109 tok）？还是局部历史？
* **输出侧**：executor 每次输出 ~659 tok（含 reasoning）。

怎么测
------
**不靠推断，直接用真机上抓下来的那一份 prompt 做对照实验**：
**同一个模型实例**（就是真机跑时用的那个 bound model）、**同一份 prompt**，
只改一个块，各打 N 次，比中位延迟。

变体：
  A  完整 prompt（基线 = 真机上发出去的那份）
  B  去掉元素表（保留其它一切）
  C  静态提示词换成一句话（保留元素表）
  D  只留目标 + 元素表 + 一句话提示（最短可用形态）

读法
----
* B 比 A 快很多 ⇒ **元素表是主要成本**，压它最划算。
* C 比 A 快很多 ⇒ **静态提示词是主要成本**（那它就不该每步都发）。
* B、C 都不明显 ⇒ 延迟主要在**输出 decode** 上，压输入没用，得压输出。

跑法（要真机在线）：
    uv run python scripts/audit_executor_latency.py
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("MOBILE_USE_TELEMETRY_ENABLED", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

# 抓到的 (bound_model, messages)
CAPTURED: list[tuple] = []


def _install_capture() -> None:
    """抓 executor 真正发出去的 messages + 那个 bound model 实例。"""
    from minitap.mobile_use.services import llm as llm_module

    orig = llm_module._ToolCallStructuredRunnable.ainvoke

    async def patched(self, input, config=None, **kwargs):  # type: ignore[no-untyped-def]
        try:
            # `self.schema` 才是绑定的 schema（`bound.kwargs` 里只有 tools /
            # tool_choice，没有 schema 键 —— 踩过）
            if getattr(getattr(self, "schema", None), "__name__", "") == "ExecutorOutput":
                CAPTURED.append((self._bound, list(input or [])))
        except Exception:  # noqa: BLE001
            pass
        return await orig(self, input, config, **kwargs)

    llm_module._ToolCallStructuredRunnable.ainvoke = patched


def _split_blocks(messages: list) -> dict:
    """把 executor prompt 拆成可单独摘除的块（按内容特征，不按位置）。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    system = next((m for m in messages if isinstance(m, SystemMessage)), None)
    target = None
    table = None
    for m in messages:
        if not isinstance(m, HumanMessage):
            continue
        text = str(m.content)
        if text.startswith("Your goal:"):
            target = m
        elif "[i]" in text or "resource-id" in text or "id/" in text:
            table = m
    return {"system": system, "target": target, "table": table}


async def _measure(bound, messages: list, *, n: int, label: str) -> dict:
    """用**同一个 bound model** 打 n 次，报中位延迟和平均 usage。"""
    times: list[float] = []
    ins: list[int] = []
    outs: list[int] = []
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            msg = await bound.ainvoke(messages)
            times.append((time.perf_counter() - t0) * 1000)
            u = getattr(msg, "usage_metadata", None) or {}
            ins.append(int(u.get("input_tokens", 0) or 0))
            outs.append(int(u.get("output_tokens", 0) or 0))
        except Exception as e:  # noqa: BLE001
            print(f"  [{label}] 失败: {type(e).__name__}: {str(e)[:140]}")
    med = round(statistics.median(times)) if times else 0
    avg_in = round(statistics.mean(ins)) if ins else 0
    avg_out = round(statistics.mean(outs)) if outs else 0
    print(
        f"  {label:<30} n={len(times)}  中位 {med:>6} ms   "
        f"输入 {avg_in:>6} tok   输出 {avg_out:>5} tok"
    )
    return {
        "label": label,
        "n": len(times),
        "median_ms": med,
        "avg_in": avg_in,
        "avg_out": avg_out,
    }


async def main() -> int:
    _install_capture()

    from minitap.mobile_use.config import initialize_llm_config
    from minitap.mobile_use.sdk import Agent
    from minitap.mobile_use.sdk.builders import Builders
    from minitap.mobile_use.sdk.types.agent import AgentProfile

    llm_config = initialize_llm_config()
    profile = AgentProfile(name="default", llm_config=llm_config)
    config = Builders.AgentConfig.with_default_profile(profile=profile)
    agent = Agent(config=config.build())
    await agent.init(retry_count=5, retry_wait_seconds=2)

    goal = "打开设置，进入 WLAN 页面"
    print(f"[probe] 真机跑一轮，抓 executor 的真实 prompt: {goal}")
    try:
        # 预算封顶：默认 80 个动作步（≈752 个 super-step）够一个跑飞的任务
        # 烧十几分钟（实测踩过）。测量探针不需要跑满，给 30 步足够抓到 prompt。
        await agent.run_task(
            request=agent.new_task(goal).with_max_steps(30).build()
        )
    except Exception as e:  # noqa: BLE001
        print(f"[probe] 任务异常（不影响后面的测量）: {type(e).__name__}: {str(e)[:200]}")
    finally:
        await agent.clean()

    if not CAPTURED:
        print("\n没抓到 executor 的 prompt —— 抓取钩子没匹配上。")
        return 1

    # 存一份 prompt 到磁盘：以后可以脱离设备重跑对照实验
    saved = ROOT / "probe_out" / "executor_prompt.json"
    try:
        saved.write_text(
            json.dumps(
                [
                    [{"type": type(m).__name__, "content": str(m.content)} for m in msgs]
                    for _, msgs in CAPTURED
                ],
                indent=1,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"   （prompt 已存到 {saved}）")
    except Exception as e:  # noqa: BLE001
        print(f"   （prompt 存盘失败：{e}）")

    # 取**最长**的那一份：元素表最全、局部历史最多，最接近真实负载
    bound, captured = max(
        CAPTURED, key=lambda pair: sum(len(str(m.content)) for m in pair[1])
    )
    blocks = _split_blocks(captured)
    print(f"\n抓到 {len(CAPTURED)} 份 executor prompt，用最长的一份做对照。")

    if blocks["system"] is None or blocks["table"] is None:
        print("⚠ 没拆出 system / 元素表块，变体对照会不准。")
        print(
            f"   块情况: system={blocks['system'] is not None} "
            f"target={blocks['target'] is not None} table={blocks['table'] is not None}"
        )
        return 1

    table_len = len(str(blocks["table"].content))
    system_len = len(str(blocks["system"].content))
    print(f"   静态提示词 {system_len:,} 字符   元素表 {table_len:,} 字符")

    from langchain_core.messages import SystemMessage

    base = list(captured)
    no_table = [m for m in base if m is not blocks["table"]]
    no_system = [
        SystemMessage(content="You are the Executor. Output one action as JSON.")
        if isinstance(m, SystemMessage)
        else m
        for m in base
    ]
    minimal = [
        SystemMessage(content="You are the Executor. Output one action as JSON.")
        if isinstance(m, SystemMessage)
        else (blocks["target"] if m is blocks["table"] else m)
        for m in base
    ]

    print("\n" + "=" * 84)
    print("变体对照（同一个 bound model 实例、同一份 prompt，只改一个块）")
    print("=" * 84)

    n = 4
    results = []
    results.append(await _measure(bound, base, n=n, label="A 完整 prompt（基线）"))
    results.append(await _measure(bound, no_table, n=n, label="B 去掉元素表"))
    results.append(await _measure(bound, no_system, n=n, label="C 静态提示词换一句话"))
    results.append(await _measure(bound, minimal, n=n, label="D 只留目标+元素表"))

    base_ms = results[0]["median_ms"]
    print()
    for r in results[1:]:
        delta = base_ms - r["median_ms"]
        share = (delta / base_ms * 100) if base_ms else 0
        print(
            f"  {r['label']:<30} 省 {delta:>5} ms（基线的 {share:>4.1f}%）"
            f"  输入少 {results[0]['avg_in'] - r['avg_in']:>5} tok"
        )

    out = ROOT / "probe_out" / "executor_latency.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
