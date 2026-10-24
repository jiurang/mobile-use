"""真机验证：「失败事实绑定到动作」这条新链路（2026-10-24）。

验什么
------
新链路是：**工具层**（失败时写下「哪个目标失败了」）→ **state**（瞬时字段）
→ **Contextor**（并进 `[上一步的结果]` 那一行）→ **Executor 的 prompt**。

本脚本在**真机**上跑完整图（真设备、真工具、真 Contextor、真 prompt 渲染），
只把**模型的第一个动作**换成一次注定失败的点击 —— 这样失败是**真的**
（真的 adb、真真的控制器、真真的报错），不需要靠运气等模型点错。

断言三件事：
1. prompt 里出现了那行失败事实；
2. **没有裸 index**（索引每轮重编号，带上会让模型点错元素）；
3. **没有结论**（「不要再用这个目标」这类判断是框架做不了的）。

跑法::

    uv run python scripts/probe_executor_failure_fact_e2e.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MOBILE_USE_TELEMETRY_ENABLED", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))
sys.stdout.reconfigure(encoding="utf-8")

from minitap.mobile_use.config import initialize_llm_config  # noqa: E402
from minitap.mobile_use.sdk import Agent  # noqa: E402
from minitap.mobile_use.sdk.builders import Builders  # noqa: E402
from minitap.mobile_use.sdk.types.agent import AgentProfile  # noqa: E402

DEVICE = "89R8ZXNBGQIRJRLF"

# 一次注定失败的点击：resource-id 和文字都不存在 ⇒ 真机上也找不到。
# ⚠️ 只**第一次**这么做；后面交给真模型自己跑完这个任务。
PROBE_TARGET = {
    "resource_id": "com.probe.nonexistent:id/definitely_not_here",
    "text": "探针不存在的元素",
}

PROMPTS: list[list] = []
CALLS: list[dict] = []

OUT_DIR = ROOT / "probe_out"
PROMPT_LOG = OUT_DIR / "executor_failure_fact_prompts.txt"
USAGE_LOG = OUT_DIR / "executor_failure_fact_usage.jsonl"


def _install_probes() -> None:
    """抓 prompt + usage。

    ⚠️ **边跑边落盘**：上一轮跑挂了（后台 900s 上限把进程杀了），断言没打印、
    证据全丢。所以这里每一轮立刻 flush 到文件 —— 进程被掐断也留得下证据。
    """
    from minitap.mobile_use.agents.executor import executor as executor_mod
    from minitap.mobile_use.services import telemetry as telemetry_mod

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_LOG.write_text("", encoding="utf-8")
    USAGE_LOG.write_text("", encoding="utf-8")

    # 1) 抓每一轮真正发给 Executor 的 messages
    original_build = executor_mod.build_executor_messages

    def build_spy(**kwargs):
        msgs = original_build(**kwargs)
        PROMPTS.append([str(m.content) for m in msgs])
        joined = "\n".join(str(m.content) for m in msgs)
        # 两处都要抓：
        #   ① 「这一步失败」= 并进局部历史的事实（**同子目标内**可见）
        #   ② 「已经失败过」= 跨子目标的失败索引块（**换子目标后**可见）
        # 只抓 ① 会漏掉「失败事实跨不过子目标边界」这类问题。
        hit = [ln for ln in joined.splitlines() if "这一步失败" in ln]
        ledger = [
            ln for ln in joined.splitlines() if "已经失败过" in ln or "already failed" in ln
        ]
        with PROMPT_LOG.open("a", encoding="utf-8") as fh:
            fh.write(
                f"=== 第 {len(PROMPTS)} 轮 | prompt {len(joined)} 字符 | "
                f"命中失败事实: {'是' if hit else '否'} | "
                f"跨子目标失败块: {'是' if ledger else '否'}\n"
            )
            for ln in hit:
                fh.write(f"    [历史] {ln.strip()}\n")
            for ln in ledger:
                fh.write(f"    [账本] {ln.strip()}\n")
            fh.flush()
        return msgs

    executor_mod.build_executor_messages = build_spy  # type: ignore[assignment]

    # 2) 抓 usage（含缓存命中）
    original_capture = telemetry_mod.telemetry.capture_llm_usage

    def capture_spy(
        task_id: str,
        agent: str,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        duration_seconds: float | None = None,
        used_fallback: bool = False,
    ) -> None:
        CALLS.append(
            {
                "agent": agent,
                "input": input_tokens,
                "cached": cached_tokens,
                "output": output_tokens,
            }
        )
        with USAGE_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(CALLS[-1], ensure_ascii=False) + "\n")
            fh.flush()
        return original_capture(
            task_id=task_id,
            agent=agent,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            duration_seconds=duration_seconds,
            used_fallback=used_fallback,
        )

    telemetry_mod.telemetry.capture_llm_usage = capture_spy  # type: ignore[method-assign]


def _script_first_action() -> None:
    """把 Executor 的**第一次**决策换成一次注定失败的点击。

    只换第一次 —— 之后交回真模型，这样后面几步是真实轨迹。
    """
    from minitap.mobile_use.agents.executor.executor import ExecutorNode
    from minitap.mobile_use.agents.executor.types import ExecutorOutput

    original_decide = ExecutorNode._decide
    counter = {"n": 0}

    async def decide(self, *, state, goal):  # type: ignore[no-untyped-def]
        counter["n"] += 1
        if counter["n"] == 1:
            print("  [探针] 第一次动作换成一次注定失败的点击（真机真失败）")
            return ExecutorOutput(
                actions=[{"action": "tap", "target": PROBE_TARGET}],
                agent_thought="（探针）先点一个不存在的元素，制造一次真实失败。",
                done=False,
            )
        return await original_decide(self, state=state, goal=goal)

    ExecutorNode._decide = decide  # type: ignore[method-assign]


def _prepare_device() -> None:
    """点亮 + 解锁 —— **不这么做 uiautomator dump 抓到的是锁屏**，元素表全是 keyguard 残留。

    还要 `svc power stayon true` —— 上一轮跑到一半屏幕自己熄了（`screen_off_timeout`
    很短），executor 被迫插入 wake_screen，白白多烧好几轮。收尾记得关掉
    （真机默认是 false）。
    """
    import subprocess
    import time

    for cmd in (
        ["shell", "input", "keyevent", "KEYCODE_WAKEUP"],
        ["shell", "wm", "dismiss-keyguard"],
        ["shell", "cmd", "statusbar", "collapse"],
        ["shell", "svc", "power", "stayon", "true"],
    ):
        subprocess.run(["adb", "-s", DEVICE, *cmd], capture_output=True, timeout=30)

    time.sleep(2.0)


def _release_device() -> None:
    """收尾：关掉 stayon（真机默认是 false，不关会一直亮）。"""
    import subprocess

    subprocess.run(
        ["adb", "-s", DEVICE, "shell", "svc", "power", "stayon", "false"],
        capture_output=True,
        timeout=30,
    )


def _check_prompts() -> tuple[bool, list[str]]:
    """在抓到的 prompt 里找那行失败事实，并检查它没有裸 index、没有结论。"""
    problems: list[str] = []
    found = False
    for i, msgs in enumerate(PROMPTS):
        joined = "\n".join(msgs)
        if "这一步失败" not in joined:
            continue
        found = True
        line = next(
            (ln for ln in joined.splitlines() if "这一步失败" in ln),
            "",
        )
        print(f"  第 {i + 1} 轮 prompt 里命中：{line.strip()[:160]}")
        # 1) 身份在
        if "探针不存在的元素" not in joined and "definitely_not_here" not in joined:
            problems.append(f"第 {i + 1} 轮：失败行里没有目标身份")
        # 2) 没有裸 index
        #
        # ⚠️ 只查**框架自己写的描述部分**：`[12] "点赞" like_button` 这种。
        # 工具回执原文里带 `(index=0)` 是**逐字保留的原始报错**，
        # 模型在上一行已经看过 —— 那不是框架下的结论，是事实。
        if re.search(r"\[\d+\]", line):
            problems.append(f"第 {i + 1} 轮：失败行里出现了裸 index：{line[:120]}")
        # 3) 没有结论
        for forbidden in ("不要再用", "换个", "不在屏幕", "无效", "没用", "重复"):
            if forbidden in line:
                problems.append(f"第 {i + 1} 轮：失败行里有结论词「{forbidden}」：{line[:120]}")

    # ---- 跨子目标的失败索引块：换子目标后失败事实还在不在 ----
    ledger_seen = 0
    for i, msgs in enumerate(PROMPTS):
        joined = "\n".join(msgs)
        for line in joined.splitlines():
            if "已经失败过" not in line and "already failed" not in line:
                continue
            ledger_seen += 1
            desc = line.split("(", 1)[0]  # 描述部分；括号里是逐字原因
            if re.search(r"\[\d+\]", desc):
                problems.append(f"第 {i + 1} 轮：跨子目标失败块里有裸 index：{line[:120]}")
            if "探针不存在的元素" not in desc and "definitely_not_here" not in desc:
                problems.append(f"第 {i + 1} 轮：跨子目标失败块里没有目标身份：{line[:120]}")
            print(f"  第 {i + 1} 轮 跨子目标失败块：{line.strip()[:160]}")
    if ledger_seen:
        print(f"  跨子目标失败块出现 {ledger_seen} 次（换子目标后失败事实仍在）")
    else:
        print("  ⚠ 跨子目标失败块一次都没出现（本次任务可能没换过子目标）")
    return found, problems


async def run(goal: str, max_steps: int) -> int:
    _install_probes()
    _script_first_action()
    _prepare_device()

    llm_config = initialize_llm_config()
    profile = AgentProfile(name="default", llm_config=llm_config)
    config = Builders.AgentConfig.with_default_profile(profile=profile)

    print("=" * 74)
    print(f"真机验证：失败事实绑定到动作（{goal}）")
    print("=" * 74)

    error = None
    agent = None
    try:
        agent = Agent(config=config.build())
        await agent.init(retry_count=5, retry_wait_seconds=2)
        request = agent.new_task(goal).build()
        # 步数预算调小 ⇒ **保证任务会自己收尾**，不会在「模型翻列表」里
        # 无限打转到后台超时（上轮就是这么被掐断的，证据全丢）。
        request.max_steps = max_steps
        await agent.run_task(request=request)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {str(e)[:400]}"
    finally:
        if agent is not None:
            await agent.clean()
        _release_device()

    print()
    print("=" * 74)
    print("断言")
    print("=" * 74)
    found, problems = _check_prompts()
    print(f"  prompt 抓到轮数: {len(PROMPTS)}")
    print(f"  失败事实进 prompt: {'✓ 是' if found else '✗ 没找到'}")

    executor_calls = [c for c in CALLS if c["agent"] == "executor"]
    in_tok = sum(c["input"] for c in executor_calls)
    ca_tok = sum(c["cached"] for c in executor_calls)
    print()
    print(f"  executor 调用 {len(executor_calls)} 次 | input={in_tok:,} cached={ca_tok:,} "
          f"命中率={(ca_tok / in_tok * 100 if in_tok else 0):.1f}%")
    all_in = sum(c["input"] for c in CALLS)
    all_ca = sum(c["cached"] for c in CALLS)
    print(f"  全部 agent 调用 {len(CALLS)} 次 | input={all_in:,} cached={all_ca:,} "
          f"命中率={(all_ca / all_in * 100 if all_in else 0):.1f}%")
    if error:
        print(f"  ⚠ 运行错误: {error}")

    out = ROOT / "probe_out" / "executor_failure_fact_e2e.json"
    out.write_text(
        json.dumps(
            {
                "goal": goal,
                "error": error,
                "prompts": len(PROMPTS),
                "failure_fact_in_prompt": found,
                "problems": problems,
                "calls": CALLS,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"  => {out}")

    if not found:
        print("\n  ✗ 没有在 prompt 里找到失败事实 —— 新链路没通")
        return 1
    if problems:
        print("\n  ✗ 发现问题：")
        for p in problems:
            print(f"    - {p}")
        return 1
    print("\n  ✓ 失败事实进了 prompt：没有裸 index，没有结论")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--goal",
        default="打开系统设置，进入 WLAN 页面，告诉我第一屏里显示的前 3 个 Wi-Fi 名称",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="步数预算：调小 ⇒ 任务保证自己收尾，不会无限打转到超时",
    )
    a = p.parse_args()
    return asyncio.run(run(a.goal, a.max_steps))


if __name__ == "__main__":
    raise SystemExit(main())
