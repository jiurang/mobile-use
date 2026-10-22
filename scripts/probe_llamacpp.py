"""Probe a local llama.cpp / llama-server endpoint with mobile-use's real schemas.

这个脚本回答的是一个具体问题：**「跑通」不等于「能用」**。
结构化输出能不能解析出来，只说明协议对得上；答得好不好是另一回事。

它对每个角色的**真实 pydantic schema** 打 N 次，统计：

* 解析成功率（能不能解析出对象 —— 不是「返回 200」）
* 延迟中位数
* 失败时的原始原因

⚠️ **每个请求都必须带 ``max_tokens`` 和客户端超时。**
llama.cpp 的 ``n_predict`` 默认 -1（不限长），而开着思考链的模型会一直续写 ——
实测不限长时请求直接挂死（20s / 300s 两次超时都没返回）。
探针自己把自己写的那个坑踩一遍，是最蠢的失败方式。

Run:
    python scripts/probe_llamacpp.py
    LLAMACPP_BASE_URL=http://127.0.0.1:9931/v1 python scripts/probe_llamacpp.py --rounds 10
    python scripts/probe_llamacpp.py --thinking     # 打开思考链，看它把预算烧成什么样
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("LLAMACPP_BASE_URL", "http://127.0.0.1:9931/v1")
MODEL = os.environ.get("LLAMACPP_MODEL", "unsloth/Qwen3.5-0.8B-GGUF:Q6_K")
TIMEOUT = 60

# Windows 控制台默认 GBK，输出中文会变成乱码。脚本本身是 UTF-8，这里只管 stdout。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):  # pragma: no cover - 老 Python / 非标准流
    pass


def _server_is_up(host: str = "127.0.0.1", port: int = 9931) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def _request(method: str, path: str, payload: dict | None = None) -> tuple[int, object]:
    url = BASE + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = resp.read().decode("utf-8", "replace")
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, body[:800]
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


def _load_schemas() -> list[tuple[str, type]]:
    """各角色真实用的 schema。

    只挑「形状不同」的那几个：json_schema 路径对扁平 schema 最容易，对嵌套
    schema（executor 的 actions[].target）才有考验。
    """
    from minitap.mobile_use.agents.contextor.types import ContextorOutput
    from minitap.mobile_use.agents.cortex.types import CortexOutput
    from minitap.mobile_use.agents.executor.types import ExecutorOutput
    from minitap.mobile_use.agents.orchestrator.types import OrchestratorOutput
    from minitap.mobile_use.agents.planner.types import PlannerOutput
    from minitap.mobile_use.agents.reporter.reporter import ReporterSummary

    return [
        ("planner", PlannerOutput),
        ("orchestrator", OrchestratorOutput),
        ("contextor", ContextorOutput),
        ("cortex", CortexOutput),
        ("executor", ExecutorOutput),
        ("reporter", ReporterSummary),
    ]


PROMPTS: dict[str, str] = {
    "planner": "Goal: read the current battery level. Produce the plan.",
    "orchestrator": "Subgoal 1 is done. Decide the next step.",
    "contextor": "Summarize what is on the current screen.",
    "cortex": "The last attempt reached the subgoal and the report shows the Wi-Fi name.",
    "executor": "The Settings icon is at index 3 on the screen. Decide the next action.",
    "reporter": "The subgoal was achieved. Write the summary.",
}


def _to_response_format(schema: type) -> dict:
    """把 pydantic schema 转成 OpenAI 的 json_schema 响应格式。

    用 openai SDK 自己的转换函数（和 langchain 走的是同一条路），避免手抄
    一份「差不多」的 JSON Schema —— 差一个字段，测出来的就不是同一条路径了。
    """
    from openai.lib._parsing._completions import type_to_response_format_param

    return type_to_response_format_param(schema)


def probe_role(name: str, schema: type, rounds: int, thinking: bool) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPTS.get(name, "Respond.")}],
        "response_format": _to_response_format(schema),
        # 关思考是默认；开思考时用它对比「思考把预算烧光」是什么样。
        "chat_template_kwargs": {"enable_thinking": thinking},
        "temperature": 0,
        # 关键：不设上限会挂死（n_predict 默认 -1）。
        "max_tokens": 4096 if thinking else 1024,
    }

    latencies: list[float] = []
    failures: list[str] = []
    parsed = 0

    for _ in range(rounds):
        started = time.perf_counter()
        status, body = _request("POST", "/chat/completions", payload)
        latencies.append(time.perf_counter() - started)
        if status != 200:
            failures.append(f"HTTP {status}: {str(body)[:200]}")
            continue
        try:
            message = body["choices"][0]["message"]
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            if not content:
                # 思考链吃光预算的典型症状：content 空，只有 reasoning。
                failures.append(
                    f"content 为空（finish_reason={body['choices'][0].get('finish_reason')}, "
                    f"reasoning {len(reasoning)} 字符）"
                )
                continue
            json.loads(content)  # 语法约束下应当一定是合法 JSON
            parsed += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{type(exc).__name__}: {exc}")

    return {
        "role": name,
        "parsed": parsed,
        "rounds": rounds,
        "latency_median": statistics.median(latencies) if latencies else 0.0,
        "latency_max": max(latencies) if latencies else 0.0,
        "failures": failures[:3],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5, help="每个角色打几次")
    ap.add_argument("--thinking", action="store_true", help="打开思考链（对比用）")
    args = ap.parse_args()

    if not _server_is_up():
        print(f"没有 llama-server 在听 {BASE} —— 先把 server 起起来。")
        return 1

    status, body = _request("GET", "/models")
    if status != 200:
        print(f"GET /models 失败：HTTP {status} {body}")
        return 1
    names = [m.get("id") for m in body.get("data", [])]
    print(f"端点：{BASE}")
    print(f"模型：{names}")
    if MODEL not in names:
        print(f"⚠️  配置里的模型名 {MODEL!r} 不在列表里 —— 端点可能会拒。")
    print(f"思考链：{'开' if args.thinking else '关'}   每角色 {args.rounds} 次\n")

    results = []
    for name, schema in _load_schemas():
        result = probe_role(name, schema, args.rounds, args.thinking)
        results.append(result)
        flag = "OK " if result["parsed"] == result["rounds"] else "部分"
        print(
            f"[{flag}] {result['role']:<14} {result['parsed']}/{result['rounds']} 解析成功  "
            f"中位 {result['latency_median']:.2f}s  最长 {result['latency_max']:.2f}s"
        )
        for failure in result["failures"]:
            print(f"        - {failure}")

    total = sum(r["parsed"] for r in results)
    rounds_total = sum(r["rounds"] for r in results)
    print(f"\n合计 {total}/{rounds_total} 解析成功")
    return 0 if total == rounds_total else 1


if __name__ == "__main__":
    sys.exit(main())
