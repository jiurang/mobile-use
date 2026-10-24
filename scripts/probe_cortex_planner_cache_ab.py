"""真 API A/B：cortex 的「规划者视角 thoughts」到底省多少缓存（2026-10-24）。

测什么
------
拿**真实执行**（exec 861）的 12 轮 cortex prompt，两种渲染各跑一遍：

* **A 现状** —— thoughts 全量（含 executor 的逐步流水）
* **B 方案 1** —— 砍掉 executor 的成功流水，只留失败类

⚠️ 三条必须守住的纪律（都踩过坑）：

1. **命中率只来自真 API 的 usage** —— 离线算的只用来定位断点。
2. **每组序列用唯一 tag 冲刷缓存** —— provider 会把「用户输入末尾」固化成
   可复用单元，同一个请求第二次发整段都会命中 ⇒ 上一组会污染下一组的首轮。
3. **不拿命中率当结论** —— 砍掉的字节本来就是「能命中缓存」的那部分，
   分子分母同减 ⇒ **命中率百分比反而会降**。
   ⇒ 结论看 **input 总量** 和 **满价部分（input − cached）**。

跑法::

    .venv\\Scripts\\python.exe scripts\\probe_cortex_planner_cache_ab.py 861
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))
sys.stdout.reconfigure(encoding="utf-8")

from minitap.mobile_use.utils.thoughts_compaction import (  # noqa: E402
    is_planner_relevant,
)

MANAGER = "http://127.0.0.1:9107"
MODEL = "MiniMax-M3.1-Flash-Preview"
BASE = "https://api.minimaxi.com/v1"
THOUGHTS_HEADER = "Agent thoughts history (newest last):"


def _env() -> dict[str, str]:
    env: dict[str, str] = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


ENV = _env()
API_KEY = ENV.get("MINIMAX_API_KEY") or os.environ.get("MINIMAX_API_KEY", "")


# --------------------------------------------------------------------- 取真实 prompt


def _api(url: str):
    with urllib.request.urlopen(url, timeout=180) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8")) if raw else None


def _blocks(pj):
    if isinstance(pj, str):
        try:
            pj = json.loads(pj)
        except json.JSONDecodeError:
            return []
    out = []
    for m in (pj or {}).get("messages") or []:
        b = m.get("blocks")
        text = (
            "\n".join(str(x.get("text", "")) for x in b)
            if isinstance(b, list)
            else str(m.get("content", ""))
        )
        out.append((str(m.get("role", "user")), text))
    return out


def _split_entries(body: str) -> list[list[str]]:
    entries: list[list[str]] = []
    for line in body.splitlines():
        if line.startswith("[") and "]" in line[:20] and line[1 : line.index("]")].isalpha():
            entries.append([line])
        elif entries:
            entries[-1].append(line)
        else:
            entries.append([line])
    return entries


def _planner_body(body: str) -> str:
    kept = []
    for entry in _split_entries(body):
        joined = "\n".join(entry)
        if is_planner_relevant(joined):
            kept.append(joined)
    return "\n".join(kept)


def real_prompts(exec_id: int) -> list[list[tuple[str, str]]]:
    rows = _api(f"{MANAGER}/api/executions/{exec_id}/prompts") or []
    rows = rows.get("items") if isinstance(rows, dict) else rows
    out = []
    for r in rows or []:
        if (r.get("agent") or "") != "cortex":
            continue
        d = _api(f"{MANAGER}/api/executions/{exec_id}/prompts/{r['callIndex']}") or {}
        bs = _blocks(d.get("promptJson") or "")
        if bs:
            out.append(bs)
    out.sort(key=lambda b: sum(len(t) for _r, t in b))
    return out


def to_messages(blocks: list[tuple[str, str]], tag: str) -> list[dict]:
    msgs = []
    for role, text in blocks:
        if tag:
            # 唯一 tag 插在 **靠前且每组都不同** 的一块 ⇒ 冲刷掉上一组的缓存
            text = text.replace("Initial goal:", f"Initial goal: {tag} ", 1)
        msgs.append({"role": "system" if role == "system" else "user", "content": text})
    return msgs


# --------------------------------------------------------------------- HTTP


def call(messages: list[dict]) -> dict:
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": 1,
        "max_tokens": 16,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    last = ""
    for attempt in range(6):
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(f"{BASE}/chat/completions", json=body, headers=headers)
        if resp.status_code == 200:
            return resp.json()
        last = f"HTTP {resp.status_code}: {resp.text[:200]}"
        if resp.status_code not in (429, 500, 502, 503, 529):
            break
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(last)


def usage_of(resp: dict) -> tuple[int, int]:
    u = resp.get("usage") or {}
    total = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
    cached = 0
    for key in ("prompt_cache_hit_tokens", "cached_tokens", "cache_read_input_tokens"):
        if isinstance(u.get(key), int):
            cached = u[key]
            break
    else:
        details = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
        if isinstance(details, dict):
            for key in ("cached_tokens", "cache_read", "cache_hit_tokens"):
                if isinstance(details.get(key), int):
                    cached = details[key]
                    break
    return total, cached


# --------------------------------------------------------------------- 主流程


def run_group(blocks_seq: list[list[tuple[str, str]]], tag: str, mode: str) -> dict:
    total_in = total_ca = 0
    rows = []
    for i, blocks in enumerate(blocks_seq):
        if mode == "planner":
            blocks = [
                (role, text)
                for role, text in blocks
                if not text.startswith(THOUGHTS_HEADER)
            ] + [
                (
                    role,
                    text.split("\n", 1)[0]
                    + "\n"
                    + _planner_body(text.split("\n", 1)[1] if "\n" in text else ""),
                )
                for role, text in blocks
                if text.startswith(THOUGHTS_HEADER)
            ]
        msgs = to_messages(blocks, tag)
        resp = call(msgs)
        tin, tca = usage_of(resp)
        total_in += tin
        total_ca += tca
        rows.append((i + 1, tin, tca, tca / tin * 100 if tin else 0))
    return {"rows": rows, "input": total_in, "cached": total_ca}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exec", nargs="?", type=int, default=861)
    a = ap.parse_args()

    seq = real_prompts(a.exec)
    if not seq:
        print(f"没拿到 #{a.exec} 的 cortex prompt")
        return 1
    print("=" * 92)
    print(f"===== #{a.exec} cortex {len(seq)} 轮 · 方案 1 真 API A/B =====")
    print("=" * 92)

    stamp = str(int(time.time()))
    a_res = run_group(seq, f"[A {stamp}]", "full")
    b_res = run_group(seq, f"[B {stamp}]", "planner")

    for name, res in (("A 现状（全量 thoughts）", a_res), ("B 方案1（规划者视角）", b_res)):
        print(f"\n--- {name} ---")
        for n, tin, tca, pct in res["rows"]:
            print(f"  第 {n:>2} 轮  input={tin:>7,}  cached={tca:>7,}  ({pct:>5.1f}%)")
        print(
            f"  合计 input={res['input']:,}  cached={res['cached']:,}  "
            f"命中率={res['cached'] / res['input'] * 100:.1f}%  "
            f"满价部分={res['input'] - res['cached']:,}"
        )

    ai, ac = a_res["input"], a_res["cached"]
    bi, bc = b_res["input"], b_res["cached"]
    print("\n" + "=" * 92)
    print("结论（口径：input 总量 + 满价部分，不是命中率百分比）")
    print("=" * 92)
    print(f"  input 总量     {ai:,} → {bi:,}   ({(bi - ai) / ai * 100:+.1f}%)")
    print(f"  cached 总量    {ac:,} → {bc:,}   ({(bc - ac) / ac * 100:+.1f}%)")
    full_a, full_b = ai - ac, bi - bc
    print(
        f"  满价部分       {full_a:,} → {full_b:,}   "
        f"({(full_b - full_a) / full_a * 100:+.1f}%)"
    )
    print(
        f"  命中率百分比   {ac / ai * 100:.1f}% → {bc / bi * 100:.1f}%"
        f"   ← 砍掉的都是「本来能命中」的字节，分子分母同减 ⇒ 这个数会降，不是收益"
    )
    print(
        "\n  ⇒ 省的钱 = cached 减少量 × 折扣 + 满价部分减少量。"
        "折扣按你自己的账单算（这里不猜）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
