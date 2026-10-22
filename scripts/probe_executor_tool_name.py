"""A/B：executor.md 里「你只有一个工具」这句，能不能把「喊错工具名」压下去。

背景（2026-10-22，执行 812 / 快手极速版 / Redmi Turbo 4）
--------------------------------------------------------
Executor 每轮只绑一个工具（``ExecutorOutput``），而且 ``tool_choice`` 是**强制**的。
但 MiniMax-M3 经常无视 ``tool_choice``，直接喊 prompt「工具表」里的**真实工具名**
（``tap`` / ``wait_for_delay`` / ``focus_and_input_text`` …）—— 66 次调用里 37 次。

解析层以前不查工具名，把那次调用当成 schema 校验「通过」→ 动作被吞成空动作 →
回灌一条无关的拒绝理由 → **连着 8 轮零动作**。

本探针
------
用**执行 812 的原始 prompt**（manager-server 的 ``llm_prompts`` 表）打真实 MiniMax
API，对比两版 system prompt：

* ``OLD`` = ``git HEAD`` 的 ``executor.md``（改动前）
* ``NEW`` = 工作区的 ``executor.md``（改动后）

只有 system 块不同，其余消息、tools、tool_choice、temperature 全都一模一样。

用法::

    # 从跑着的 manager-server 拉原始 prompt
    python scripts/probe_executor_tool_name.py --exec-id 812 --call-index 10 --n 8

    # 或用本地存的 promptJson
    python scripts/probe_executor_tool_name.py --prompt-file probe_out/p812/010.json --n 8

判据：``NEW`` 的「喊错工具名」次数必须显著少于 ``OLD``。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PROMPT_REL = "minitap/mobile_use/agents/executor/executor.md"
MANAGER = os.environ.get("MANAGER_URL", "http://127.0.0.1:9107")


# ---------------------------------------------------------------------------
# 取原始 prompt
# ---------------------------------------------------------------------------


def _load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    env_file = REPO / ".env"
    if not env_file.exists():
        return env
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def fetch_prompt(exec_id: int, call_index: int) -> dict:
    url = f"{MANAGER}/api/executions/{exec_id}/prompts/{call_index}"
    with urllib.request.urlopen(url, timeout=30) as r:
        dto = json.load(r)
    return json.loads(dto["promptJson"])


def load_prompt_file(path: str) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return json.loads(raw["promptJson"]) if "promptJson" in raw else raw


# ---------------------------------------------------------------------------
# 渲染两版 system prompt
# ---------------------------------------------------------------------------


def old_template() -> str:
    out = subprocess.run(
        ["git", "show", f"HEAD:{PROMPT_REL}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return out.stdout


def new_template() -> str:
    return (REPO / PROMPT_REL).read_text(encoding="utf-8")


def tools_list_from(system_text: str) -> str:
    m = re.search(r"^Available tools: (.+)$", system_text, re.MULTILINE)
    if not m:
        raise SystemExit("system prompt 里找不到 'Available tools:' 行")
    return m.group(1).strip()


def render(template: str, *, platform: str, tools: str) -> str:
    from jinja2 import Template

    return Template(template).render(platform=platform, executor_tools_list=tools, goal="")


def build_variants(saved_system: str) -> tuple[str, str]:
    """返回 (OLD system, NEW system)。

    先用 OLD 模板 + 候选 platform 渲染，**必须逐字节等于**存下来的 system 块 ——
    不等就说明变量没抽对，直接停手（否则 A/B 比的是两件不同的事）。
    """
    tools = tools_list_from(saved_system)
    old = old_template()
    for platform in ("android", "ios", "ANDROID", "IOS", "Android", "iOS"):
        if render(old, platform=platform, tools=tools) == saved_system:
            old_system = saved_system
            break
    else:
        raise SystemExit("渲染旧模板没能逐字节复现存下来的 system 块 —— 变量没抽对，A/B 无效。")
    new_system = render(new_template(), platform=platform, tools=tools)
    return old_system, new_system


# ---------------------------------------------------------------------------
# 打 MiniMax
# ---------------------------------------------------------------------------


def to_openai_messages(messages: list[dict], *, rewrite: bool = False) -> list[dict]:
    """runner 抓取层的格式 → OpenAI 请求格式。

    ⚠️ runner 的抓取层**丢掉了 tool_call id**（只留 name + args），但真实请求里
    id 是有的（langchain 生成）。这里按「assistant 带几个 call，就跟几个 tool
    结果配对」的顺序补 id —— 局部历史就是这个形状（一个动作一条结果）。

    ``rewrite=True`` 时把历史里的设备工具名改写成 ``ExecutorOutput``
    （镜像 ``agents/executor/prompt.py:rewrite_history_tool_calls``）。
    """
    out: list[dict] = []
    pending: list[str] = []
    counter = 0
    for m in messages:
        role = m.get("role", "user")
        text = " ".join(
            b.get("text", "") for b in (m.get("blocks") or []) if b.get("type") == "text"
        )
        calls = m.get("toolCalls") or []
        if role == "assistant" and calls:
            tool_calls = []
            for c in calls:
                counter += 1
                call_id = f"call_hist_{counter}"
                pending.append(call_id)
                name = c.get("name")
                args = c.get("args") or {}
                if rewrite and name and name != "ExecutorOutput":
                    thought = args.get("agent_thought") or ""
                    action = {
                        "action": name,
                        **{k: v for k, v in args.items() if k != "agent_thought"},
                    }
                    args = {"actions": [action], "agent_thought": thought}
                    name = "ExecutorOutput"
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    }
                )
            out.append({"role": "assistant", "content": text or None, "tool_calls": tool_calls})
            continue
        if role == "tool":
            call_id = pending.pop(0) if pending else "call_hist_unknown"
            msg = {"role": "tool", "tool_call_id": call_id, "content": text}
            if m.get("name"):
                # 改写后 assistant 那边的工具名是 ExecutorOutput —— 这里跟着对齐
                msg["name"] = "ExecutorOutput" if rewrite else m["name"]
            out.append(msg)
            continue
        out.append({"role": role, "content": text})
    return out


def call_minimax(env: dict, system: str, messages: list[dict], params: dict) -> dict:
    base = env.get("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1").rstrip("/")
    payload = {
        "model": params.get("model", "MiniMax-M3"),
        "temperature": params.get("temperature", 1.0),
        "messages": [{"role": "system", "content": system}, *messages],
        "tools": params.get("tools"),
        "tool_choice": params.get("tool_choice"),
        "parallel_tool_calls": params.get("parallel_tool_calls", False),
    }
    effort = env.get("MINIMAX_EFFORT")
    if effort:
        payload["output_config"] = {"effort": effort}
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {env.get('MINIMAX_API_KEY', '')}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def classify(resp: dict, expected: str) -> str:
    choices = resp.get("choices") or []
    if not choices:
        return "empty"
    msg = choices[0].get("message") or {}
    calls = msg.get("tool_calls") or []
    if not calls:
        return "prose" if (msg.get("content") or "").strip() else "empty"
    name = calls[0].get("function", {}).get("name")
    return "CORRECT" if name == expected else f"WRONG:{name}"


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec-id", type=int)
    ap.add_argument("--call-index", type=int)
    ap.add_argument("--prompt-file")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    if args.prompt_file:
        prompt = load_prompt_file(args.prompt_file)
    elif args.exec_id is not None and args.call_index is not None:
        prompt = fetch_prompt(args.exec_id, args.call_index)
    else:
        ap.error("要么给 --prompt-file，要么给 --exec-id + --call-index")

    env = _load_env()
    if not env.get("MINIMAX_API_KEY"):
        raise SystemExit("没有 MINIMAX_API_KEY（.env 里找）")

    saved_system = prompt["messages"][0]["blocks"][0]["text"]
    old_system, new_system = build_variants(saved_system)
    raw_messages = prompt["messages"][1:]
    params = prompt.get("params", {})
    expected = params.get("tool_choice", {}).get("function", {}).get("name", "ExecutorOutput")

    # 三个变体，只有这两处不同：
    #   OLD        = HEAD 的 executor.md + 原样历史（= 修复前的真机行为）
    #   NEW_PROMPT = 新的 executor.md + 原样历史（只测提示词这一半）
    #   NEW_ALL    = 新的 executor.md + 改写后的历史（= 完整修复）
    variants = [
        ("OLD", old_system, to_openai_messages(raw_messages)),
        ("NEW_PROMPT", new_system, to_openai_messages(raw_messages)),
        ("NEW_ALL", new_system, to_openai_messages(raw_messages, rewrite=True)),
    ]

    print(f"模型={params.get('model')}  tool_choice={expected}  每组 n={args.n}")
    print(f"system: OLD {len(old_system)} chars / NEW {len(new_system)} chars")
    diff_lines = sum(1 for a, b in zip(old_system.splitlines(), new_system.splitlines()) if a != b)
    print(f"system 差异行数: {diff_lines}")
    tainted = sum(
        1
        for m in raw_messages
        if m.get("role") == "assistant"
        for c in (m.get("toolCalls") or [])
        if c.get("name") and c.get("name") != "ExecutorOutput"
    )
    print(f"历史里被污染的设备工具名: {tainted} 个")
    print()

    results: dict[str, Counter] = {name: Counter() for name, _, _ in variants}
    latencies: dict[str, list[float]] = {name: [] for name, _, _ in variants}

    for variant, system, messages in variants:
        for i in range(args.n):
            t0 = time.perf_counter()
            try:
                resp = call_minimax(env, system, messages, params)
            except Exception as e:  # noqa: BLE001
                results[variant]["ERROR:" + type(e).__name__] += 1
                print(f"  {variant} #{i + 1} ERROR {type(e).__name__}: {str(e)[:160]}")
                continue
            latencies[variant].append(time.perf_counter() - t0)
            kind = classify(resp, expected)
            results[variant][kind] += 1
            print(f"  {variant} #{i + 1}: {kind}  ({latencies[variant][-1]:.1f}s)")

    print()
    print("=" * 70)
    for variant, _, _ in variants:
        c = results[variant]
        wrong = sum(v for k, v in c.items() if k.startswith("WRONG"))
        good = c.get("CORRECT", 0)
        lat = latencies[variant]
        med = sorted(lat)[len(lat) // 2] if lat else 0
        print(
            f"{variant:>11}: 正确 {good} / 喊错 {wrong} / "
            f"散文 {c.get('prose', 0)} / 中位 {med:.1f}s"
        )
        for k, v in c.most_common():
            print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
