"""验证「把可变块并成追加流」的设计（2026-10-24）。

提出的设计
----------
把三块易变内容整理成**只追加**的流，每轮只发新增的那一段：

    历史   a b c          ← 每轮新增一条
    失败账本 a1 c2        ← 失败挂在**产生它的那条历史**上
    第一轮: a + a1；第二轮: b；第三轮: c + c1 ……

本脚本量三件事
--------------
1. **A/B**：同样的失败事实，两种摆法 ——

   * **A 现状**：失败账本是**独立的块**，排在局部历史之后（⇒ 在缓存尾巴里，
     每轮满价重发，而且它只增不减 ⇒ 越跑越贵）。
   * **B 提案**：失败事实**绑到产生它的那条历史条目上**，随历史一起追加
     （⇒ 落在可缓存的前缀里）。

2. **尾巴大小对命中率的影响**（失败账本越大，现状越吃亏）。

3. 离线算的公共前缀比例 vs 真 API 的 cached —— 两边对得上才算数。

跑法::

    uv run python scripts/probe_executor_merged_append.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minitap"))
sys.stdout.reconfigure(encoding="utf-8")


# ⚠️ 这里原来有个 ``_import_channels_tolerantly()`` —— 声称给「channels.py 在途
# 重构、import 直接炸」兜个底。它**从来没生效过**：算出来的 ``src``（摘掉
# ``force_tool_choice=`` 之后的内容）被丢掉了，模块仍然从**原文件**加载 ⇒ 真
# 触发时照样抛同一个 TypeError，只是多打一句「兜住了继续」骗人。
#
# 而且它现在**根本不会触发**：``force_tool_choice`` 字段已经从
# ``ChannelSpec`` 里删干净（见 ``services/channels.py``），channels.py 导入是
# 通的。⇒ 整段删掉，别留着假装兜底。


from minitap.mobile_use.agents.executor.prompt import build_executor_messages  # noqa: E402
from minitap.mobile_use.agents.executor.types import ExecutorOutput  # noqa: E402
from minitap.mobile_use.config import LLM, LLMConfig, LLMConfigUtils, LLMWithFallback  # noqa: E402
from minitap.mobile_use.context import (  # noqa: E402
    DeviceContext,
    DevicePlatform,
    MobileUseContext,
)
from minitap.mobile_use.utils.screen_elements import ScreenElement  # noqa: E402
from minitap.mobile_use.utils.ui_hierarchy import ElementBounds  # noqa: E402

MODEL = "MiniMax-M3.1-Flash-Preview"
BASE = "https://api.minimaxi.com/v1"
GOAL = "进入 WLAN 页面并读出当前连接的 Wi-Fi 名称"

LABELS = ["首页", "搜索", "推荐", "关注", "我的", "消息", "视频", "直播", "设置",
          "返回", "更多", "收藏", "分享", "评论", "下载", "Wi-Fi", "蓝牙", "流量",
          "省电模式", "勿扰模式", "个性定制", "电池", "存储空间", "隐私保护"]
CLASSES = ["TextView", "ImageView", "LinearLayout", "FrameLayout", "android.widget.Button"]


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        env[k.strip()] = v
    return env


API_KEY = load_env()["MINIMAX_API_KEY"]


def _ctx() -> MobileUseContext:
    def llm() -> LLMWithFallback:
        return LLMWithFallback(
            provider="openai", model="gpt-5", fallback=LLM(provider="openai", model="o4-mini")
        )

    return MobileUseContext(
        trace_id="probe",
        device=DeviceContext(
            host_platform="WINDOWS",
            mobile_platform=DevicePlatform.ANDROID,
            device_id="probe",
            device_width=1220,
            device_height=2712,
        ),
        llm_config=LLMConfig(
            planner=llm(), orchestrator=llm(), contextor=llm(), cortex=llm(), executor=llm(),
            utils=LLMConfigUtils(outputter=llm(), hopper=llm()),
        ),
    )


def make_elements(turn: int, n_rows: int) -> list[ScreenElement]:
    out: list[ScreenElement] = []
    for i in range(n_rows):
        k = (i + turn * 3) % len(LABELS)
        label = LABELS[k] + (f"{i}" if i % 5 == 0 else "")
        out.append(
            ScreenElement(
                index=i + 1, kind="ui", label=label, text="",
                resource_id=f"com.example.app:id/{label}_{i}", display_id="",
                class_name=CLASSES[i % len(CLASSES)],
                bounds=ElementBounds(x=20 * (i % 6), y=120 * i, width=360, height=96),
                flags="CLFS" if i % 3 == 0 else "C",
            )
        )
    return out


def make_failures(n: int) -> list[dict]:
    """造 n 条跨子目标的失败事实（形状同 ``append_failure`` 的产物）。"""
    out = []
    for i in range(n):
        label = LABELS[(i * 5) % len(LABELS)]
        out.append(
            {
                "key": f"index={7 + i * 3}|id=com.example.app:id/{label}_{7 + i * 3}|text={label}",
                "index": 7 + i * 3,
                "resource_id": f"com.example.app:id/{label}_{7 + i * 3}",
                "label": label,
                "reason": "no element with resource_id on the current screen",
            }
        )
    return out


def make_history(steps: int, failures: list[dict], *, merge: bool) -> list:
    """局部历史。

    ``merge=True``（提案 B）：失败事实**绑到产生它的那条历史**上 ——
    第 i 步失败，第 i 条历史条目里就带着它。
    ``merge=False``（现状 A）：历史里不带失败，失败单独成块。
    """
    by_step: dict[int, list[dict]] = {}
    if merge:
        for f in failures:
            step = (f["index"] - 7) // 3  # 和 make_failures 的下标对应
            by_step.setdefault(min(step, max(steps - 1, 0)), []).append(f)

    msgs: list = []
    for i in range(steps):
        call_id = f"call_{i:02d}"
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "tap",
                    "args": {"target": {"index": 7 + (i % 5)},
                             "agent_thought": f"第 {i + 1} 步：先点搜索框聚焦。"},
                    "id": call_id,
                }],
            )
        )
        msgs.append(ToolMessage(content="Tapped element (610, 1258) successfully.", tool_call_id=call_id))
        line = f"[上一步的结果] 前台应用没变（还是 com.example.app）；屏幕上新出现了：{LABELS[i % len(LABELS)]}{i}"
        if merge:
            for f in by_step.get(i, []):
                line += (
                    f"；这一步失败了：[{f['index']}] \"{f['label']}\" "
                    f"{f['resource_id'].split('/')[-1]}（{f['reason']}）—— 不要再用这个目标"
                )
        msgs.append(HumanMessage(content=line))
    return msgs


def to_openai(messages) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if isinstance(m, HumanMessage) and not isinstance(m, ToolMessage):
            out.append({"role": "user", "content": str(m.content)})
        elif hasattr(m, "tool_calls") and getattr(m, "tool_calls", None):
            out.append({
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": tc["id"], "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["args"], ensure_ascii=False)},
                } for tc in m.tool_calls],
            })
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": str(m.content)})
        else:
            out.append({"role": "system", "content": str(m.content)})
    return out


def build_turn(ctx, *, turn: int, n_rows: int, steps: int, n_failures: int, merge: bool, tag: str):
    failures = make_failures(n_failures)
    elements = make_elements(turn, n_rows)
    msgs = build_executor_messages(
        ctx=ctx,
        goal=f"{GOAL}（探针 {tag}）",
        screen_elements=elements,
        # 现状 A：失败单独成块（在缓存尾巴里）；提案 B：并进历史（进可缓存前缀）
        failures=failures if not merge else [],
        local_messages=make_history(steps, failures, merge=merge),
        device_state_lines=["Device date: 2026-10-24 09:20:11", "Focused app info: com.example.app"],
    )
    return to_openai(msgs)


def call(messages: list[dict], tools: list[dict]) -> dict:
    body = {
        "model": MODEL, "messages": messages, "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "ExecutorOutput"}},
        "parallel_tool_calls": False, "temperature": 1, "max_tokens": 32,
        "output_config": {"effort": "low"},
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


def cached_of(u: dict) -> int:
    for key in ("prompt_cache_hit_tokens", "cached_tokens", "cache_read_input_tokens"):
        if isinstance(u.get(key), int):
            return u[key]
    details = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
    if isinstance(details, dict):
        for key in ("cached_tokens", "cache_read", "cache_hit_tokens"):
            if isinstance(details.get(key), int):
                return details[key]
    return 0


def _stream(msgs: list[dict]) -> str:
    return "".join(f"{m['role']}:{m.get('content') or ''}\n" for m in msgs)


def _lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def run(label: str, ctx, *, n_rows: int, turns: int, n_failures: int, merge: bool, tools) -> tuple[int, int, float]:
    print("=" * 96)
    print(f"{label}（{turns} 轮，元素表 {n_rows} 行，失败账本 {n_failures} 条）")
    print("=" * 96)
    tp = tc = 0
    prev = None
    lcp_ratio = 0.0
    for turn in range(1, turns + 1):
        msgs = build_turn(ctx, turn=turn, n_rows=n_rows, steps=turn - 1,
                          n_failures=n_failures, merge=merge, tag=label)
        if prev is not None:
            lcp_ratio = _lcp(_stream(prev), _stream(msgs)) / max(len(_stream(msgs)), 1)
        u = (call(msgs, tools) or {}).get("usage") or {}
        p, c = u.get("prompt_tokens", 0), cached_of(u)
        tp += p
        tc += c
        print(f"  第 {turn:>2} 轮  prompt={p:>6,}  cached={c:>6,}  ({c / max(p, 1):>6.1%})")
        prev = msgs
        time.sleep(0.4)
    rate = tc / tp if tp else 0.0
    print(f"  合计  prompt={tp:>7,}  cached={tc:>7,}  命中率={rate:>6.1%}  满价重算={tp - tc:>7,}"
          f"   末轮公共前缀占比={lcp_ratio:.1%}\n")
    return tp, tc, rate


def main() -> int:
    ctx = _ctx()
    tools = [convert_to_openai_tool(ExecutorOutput)]
    results = {}

    print("=" * 96)
    print("失败账本多大？（真机长任务里它是跨子目标累积的）")
    print("=" * 96)
    for n in (0, 4, 12, 24):
        fs = make_failures(n)
        rendered = build_executor_messages(
            ctx=ctx, goal=GOAL, screen_elements=make_elements(1, 60), failures=fs,
            local_messages=make_history(3, fs, merge=False),
            device_state_lines=["Device date: 2026-10-24 09:20:11", "Focused app info: com.example.app"],
        )
        ledger = next((str(m.content) for m in rendered if "already failed" in str(m.content)), "")
        print(f"  {n:>3} 条失败 → 账本块 {len(ledger):>5,} 字符（每轮全价重发）")
    print()

    for n_fail in (4, 12):
        tp_a, tc_a, r_a = run(f"A 现状·账本独立成块·{n_fail}条", ctx, n_rows=60, turns=8,
                              n_failures=n_fail, merge=False, tools=tools)
        tp_b, tc_b, r_b = run(f"B 提案·并进历史·{n_fail}条", ctx, n_rows=60, turns=8,
                              n_failures=n_fail, merge=True, tools=tools)
        results[n_fail] = (r_a, r_b, (tp_a - tc_a) - (tp_b - tc_b))

    print("=" * 96)
    print("小结")
    print("=" * 96)
    for n_fail, (r_a, r_b, saved) in results.items():
        print(f"  失败账本 {n_fail:>3} 条：现状 {r_a:>6.1%} → 提案 {r_b:>6.1%} "
              f"({(r_b - r_a) * 100:+.1f} 个百分点，8 轮少烧 {saved:,} token)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
