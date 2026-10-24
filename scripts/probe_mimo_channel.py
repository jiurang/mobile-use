"""探针：MiMo Token Plan 渠道（mimo-token-plan）的连通性，端到端。

为什么分层测
------------
「能不能连通」不是一个是非题，它有三层，每层坏掉的样子完全不同：

1. **裸 HTTP** —— 端点对不对、key 认不认、模型名对不对。
   失败症状：404 / 401 / "model not found"。
2. **thinking_aware 契约** —— MiMo 是「思考开着就不能强制 tool_choice」那一类端点。
   关掉 thinking 还强制 tool_choice 必须拿到 200，否则结构化输出这条路是死的。
   失败症状：HTTP 400 "Thinking mode does not support this tool_choice"。
3. **项目自己的客户端** —— 走 ``build_openai_protocol_client``，
   用渠道目录里的真实数据（base_url / api_key_env / client_kind）。
   这一层最容易坏，而且坏了不报错：渠道目录说 key 在 ``MIMO_API_KEY``，
   但 ``Settings`` 没收这个字段 ⇒ key 变 ``sk-local`` ⇒ 401 指向"你的 key 不对"，
   而真正的原因是"配置根本没接上"。

每条证据**立刻落盘**再继续（后台命令被掐断也不丢），输出文件：
``probe_out/probe_mimo_channel_<日期>.txt``

跑法：
    .\\.venv\\Scripts\\python.exe scripts/probe_mimo_channel.py
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

# Windows 控制台默认 GBK，打不出 ✅/❌ → 直接崩在 print 上，
# 证据全丢。强制 UTF-8，errors="replace" 兜底（落盘的文件始终是 UTF-8）。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

OUT_DIR = ROOT / "probe_out"
OUT_FILE = OUT_DIR / f"probe_mimo_channel_{date.today():%Y%m%d}.txt"

BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
CHANNEL_ID = "mimo-token-plan"
CANDIDATE_MODELS = ("mimo-v2.6-pro", "mimo-v2.6-flash")


def emit(line: str) -> None:
    """打印 + 立刻落盘。"""
    print(line, flush=True)
    with OUT_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


def section(title: str) -> None:
    emit("")
    emit(f"=== {title} ===")


async def probe_models(c: httpx.AsyncClient, key: str) -> list[str]:
    section("L1a  GET /v1/models —— 这把 key 能看到哪些模型")
    try:
        r = await c.get(
            f"{BASE_URL}/models",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
    except Exception as e:  # noqa: BLE001
        emit(f"  ❌ 连接失败 {type(e).__name__}: {e}")
        return []
    emit(f"  HTTP {r.status_code}")
    try:
        body = r.json()
        ids = [m.get("id") for m in body.get("data", []) if isinstance(m, dict)]
        emit(f"  模型列表 ({len(ids)}): {ids}")
        return [i for i in ids if isinstance(i, str)]
    except Exception:  # noqa: BLE001
        emit(f"  body: {r.text[:400]}")
        return []


async def probe_chat(
    c: httpx.AsyncClient,
    key: str,
    model: str,
    *,
    thinking: str | None = None,
    force_tool: bool = False,
) -> dict:
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": "只回两个字：你好"}],
        "max_completion_tokens": 64,
    }
    if thinking is not None:
        payload["thinking"] = {"type": thinking}
    if force_tool:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "report",
                    "description": "上报一个结果",
                    "parameters": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                    },
                },
            }
        ]
        payload["tool_choice"] = {"type": "function", "function": {"name": "report"}}

    t0 = time.perf_counter()
    try:
        r = await c.post(
            f"{BASE_URL}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "err": f"{type(e).__name__}: {e}",
            "ms": int((time.perf_counter() - t0) * 1000),
        }

    ms = int((time.perf_counter() - t0) * 1000)
    try:
        body = r.json()
    except Exception:  # noqa: BLE001
        return {"ok": False, "status": r.status_code, "err": r.text[:300], "ms": ms}

    if r.status_code != 200:
        return {
            "ok": False,
            "status": r.status_code,
            "err": json.dumps(body, ensure_ascii=False)[:300],
            "ms": ms,
        }

    choices = body.get("choices") or []
    first = choices[0] if choices else {}
    msg = first.get("message") or {}
    content = msg.get("content")
    tool_calls = msg.get("tool_calls")
    usage = body.get("usage") or {}
    return {
        "ok": True,
        "status": 200,
        "ms": ms,
        "content": content,
        "tool_calls": tool_calls,
        "reasoning": (msg.get("reasoning_content") or "")[:200],
        "finish_reason": first.get("finish_reason"),
        "usage": usage,
    }


async def probe_layer1(c: httpx.AsyncClient, key: str) -> None:
    section("L1b  POST /v1/chat/completions —— 裸 HTTP 文本对话")
    for model in CANDIDATE_MODELS:
        res = await probe_chat(c, key, model)
        if res["ok"]:
            emit(
                f"  ✅ {model:<18} {res['ms']:>6}ms  content={res['content']!r} "
                f"finish={res['finish_reason']} usage={res['usage']}"
            )
        else:
            emit(f"  ❌ {model:<18} {res.get('status', '-')} {res.get('err', '')[:240]}")


async def probe_layer2(c: httpx.AsyncClient, key: str, model: str) -> None:
    section("L2  thinking_aware 契约 —— 关掉 thinking 后能不能强制 tool_choice")
    cases = [
        ("thinking=disabled + 强制 tool_choice", {"thinking": "disabled", "force_tool": True}),
        ("thinking=enabled  + 强制 tool_choice", {"thinking": "enabled", "force_tool": True}),
        ("thinking 不下发  + 强制 tool_choice", {"thinking": None, "force_tool": True}),
        ("thinking=enabled  + 不带工具", {"thinking": "enabled", "force_tool": False}),
    ]
    for label, kw in cases:
        res = await probe_chat(c, key, model, **kw)
        if res["ok"]:
            tc = res.get("tool_calls")
            emit(
                f"  ✅ {label:<36} {res['ms']:>6}ms  tool_calls={'有' if tc else '无'} "
                f"finish={res['finish_reason']} reasoning={(res.get('reasoning') or '')[:60]!r}"
            )
            if tc:
                emit(f"       └─ tool_call: {json.dumps(tc[0], ensure_ascii=False)[:220]}")
        else:
            emit(f"  ❌ {label:<36} HTTP {res.get('status', '-')} {res.get('err', '')[:240]}")


async def probe_layer3(model: str) -> None:
    section("L3  项目自己的渠道客户端 —— build_openai_protocol_client")
    try:
        from minitap.mobile_use.services.channels import require_channel
        from minitap.mobile_use.services.llm import build_openai_protocol_client
    except Exception as e:  # noqa: BLE001
        emit(f"  ❌ 导入失败 {type(e).__name__}: {e}")
        return

    spec = require_channel(CHANNEL_ID)
    emit(f"  渠道: {spec.id} / {spec.protocol} / {spec.base_url}")
    emit(
        f"  密钥变量: {spec.api_key_env} / client_kind={spec.client_kind}"
        f" / thinking_default_on={spec.thinking_default_on}"
    )

    # ⚠️ 这一步就是「配置没接上」最容易藏的地方：key_or() 会回退到
    # settings.<API_KEY_ENV>，而 Settings 是 extra="ignore" —— 没声明的字段
    # 会被静默丢弃 ⇒ 密钥变 "sk-local" ⇒ 401 指向"你的 key 不对"。
    try:
        from minitap.mobile_use.config import settings

        raw = getattr(settings, spec.api_key_env, None)
        if raw is None:
            emit(
                f"  ❌ settings.{spec.api_key_env} = None"
                "  ← Settings 没声明这个字段，.env 里的值被丢弃了"
            )
        else:
            shown = raw.get_secret_value() if hasattr(raw, "get_secret_value") else str(raw)
            emit(f"  ✅ settings.{spec.api_key_env} 已读到 (前缀 {shown[:3]}…, 长度 {len(shown)})")
    except Exception as e:  # noqa: BLE001
        emit(f"  ❌ 读 settings 失败 {type(e).__name__}: {e}")

    try:
        client = build_openai_protocol_client(spec, model)
        emit(f"  客户端类型: {type(client).__name__}  base_url={client.openai_api_base}")
        res = await client.ainvoke([{"role": "user", "content": "只回两个字：你好"}])
        content = getattr(res, "content", res)
        emit(f"  ✅ 调用成功: content={str(content)[:200]!r}")
    except Exception as e:  # noqa: BLE001
        emit(f"  ❌ 调用失败 {type(e).__name__}: {str(e)[:400]}")


async def probe_layer3b(model: str) -> None:
    section("L3b  agent 真实结构化输出路径 —— get_structured_llm + ExecutorOutput")
    try:
        from types import SimpleNamespace

        from minitap.mobile_use.agents.executor.types import ExecutorOutput
        from minitap.mobile_use.config import LLM, LLMConfig, LLMConfigUtils, LLMWithFallback
        from minitap.mobile_use.services.llm import get_structured_llm
    except Exception as e:  # noqa: BLE001
        emit(f"  ❌ 导入失败 {type(e).__name__}: {e}")
        return

    # 走「老路」（allow_env_fallback=True）⇒ 密钥从 settings.MIMO_API_KEY 读。
    # 这正是 agent 实际跑的那条路：.env 里配好就起得来，不用在代码里传 key。
    entry = LLM(provider=CHANNEL_ID, model=model)
    wf = LLMWithFallback(**entry.model_dump(), fallback=entry.model_dump())
    cfg = LLMConfig(
        planner=wf,
        orchestrator=wf,
        contextor=wf,
        cortex=wf,
        executor=wf,
        utils=LLMConfigUtils(outputter=wf, hopper=wf),
    )
    ctx = SimpleNamespace(llm_config=cfg, trace_id="probe-mimo")

    try:
        runner = get_structured_llm(ctx=ctx, name="executor", schema=ExecutorOutput)
    except Exception as e:  # noqa: BLE001
        emit(f"  ❌ 建 runner 失败 {type(e).__name__}: {e}")
        return

    forced = "tool_choice" in getattr(runner, "_bound", None).kwargs
    emit(f"  绑定了 tool_choice: {forced}（thinking 没下发 ⇒ 服务端默认开着思考 ⇒ 预期 False）")

    prompt = (
        "你现在是 executor。屏幕上有一个按钮写着「打开」。"
        "请调用 ExecutorOutput 工具，actions 里放一个动作："
        '{"action": "click", "index": 0}'
    )

    # 先看**原始**消息：模型到底调了工具没、调的哪个名字、args 是什么。
    # 不看原始消息就无法区分「模型返回空列表」和「解析把内容吞了」——
    # 而后者会静默产出 actions=[]，agent 会因此原地打转。
    t0 = time.perf_counter()
    try:
        raw = await runner._bound.ainvoke(prompt)
    except Exception as e:  # noqa: BLE001
        emit(f"  ❌ 原始调用失败 {type(e).__name__}: {str(e)[:400]}")
        return
    ms = int((time.perf_counter() - t0) * 1000)

    emit(f"  --- 原始 AIMessage ({ms}ms) ---")
    emit(f"  finish_reason={getattr(raw, 'response_metadata', {}).get('finish_reason')!r}")
    tcs = raw.tool_calls or []
    emit(f"  tool_calls={len(tcs)}")
    for tc in tcs:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
        emit(f"    - name={name!r} args={json.dumps(args, ensure_ascii=False)[:300]}")
    content = raw.content if isinstance(raw.content, str) else ""
    emit(f"  content[:400]={content[:400]!r}")
    if not tcs:
        emit("  ⚠️ 模型没调工具 —— 框架会走 content 解析兜底（可能静默产出空 actions）")

    t0 = time.perf_counter()
    try:
        result = await runner.ainvoke(prompt)
    except Exception as e:  # noqa: BLE001
        emit(f"  ❌ 结构化解析失败 {type(e).__name__}: {str(e)[:400]}")
        return
    ms = int((time.perf_counter() - t0) * 1000)
    actions = getattr(result, "actions", None)
    emit(f"  --- 框架解析结果 ({ms}ms) ---")
    emit(f"  actions={actions}")
    emit(f"  {'✅ 有动作' if actions else '❌ 空动作 —— agent 会因此原地打转'}")


async def main() -> int:
    key = os.environ.get("MIMO_API_KEY") or ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("", encoding="utf-8")

    emit(f"MiMo 渠道连通性探针  {date.today()}")
    emit(f"端点: {BASE_URL}")
    emit(
        f"key: {(key[:3] + '…' + key[-4:]) if key else '（环境里没有 MIMO_API_KEY）'}"
        f" 长度={len(key)}"
    )

    if not key:
        emit("❌ 环境里没有 MIMO_API_KEY，无法测真实调用。")
        return 1

    async with httpx.AsyncClient(timeout=90, follow_redirects=True) as c:
        await probe_models(c, key)
        await probe_layer1(c, key)
        await probe_layer2(c, key, CANDIDATE_MODELS[0])
        await probe_layer3(CANDIDATE_MODELS[0])
        await probe_layer3b(CANDIDATE_MODELS[0])

    emit("")
    emit(f"证据落盘: {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
