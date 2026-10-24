"""Executor 提示词组成 + 前缀缓存命中率实测（真 API）。

回答两个问题
------------
1. **组成**：Executor 每一轮的 prompt 由哪几块组成、各多少 token、哪些块每轮变。
   渲染走**生产代码**（``build_executor_messages``），不是另抄一份模板 ——
   抄的那份和真实发出去的可能已经不一样了。

2. **为什么命中率低**：把相邻两轮的请求逐块对比，找出「前缀在哪断」，
   再打真 API 读 provider 报的 ``cached_tokens`` 对账。

⚠️ **模拟值不算数**（``doc/cache-hit-rate-audit-20261020.md`` 的教训）：
本脚本的每一行命中率都来自真 API 的 usage 字段。离线那部分只用来
「定位断在哪个块」，不当作命中率结论。

跑法::

    uv run python scripts/probe_executor_cache_measure.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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

LABELS = [
    "首页", "搜索", "推荐", "关注", "我的", "消息", "视频", "直播", "设置",
    "返回", "更多", "收藏", "分享", "评论", "下载", "Wi-Fi", "蓝牙", "流量",
    "省电模式", "勿扰模式", "个性定制", "电池", "存储空间", "隐私保护",
]
CLASSES = ["TextView", "ImageView", "LinearLayout", "FrameLayout", "android.widget.Button"]


# --------------------------------------------------------------------------- env


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        # .env 里这个值是带引号写的，手写 loader 也得剥，否则 Authorization 头
        # 里会带上字面引号 -> 401。
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        env[k.strip()] = v
    return env


ENV = load_env()
API_KEY = ENV["MINIMAX_API_KEY"]


# ------------------------------------------------------------------- 素材构造


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
            planner=llm(),
            orchestrator=llm(),
            contextor=llm(),
            cortex=llm(),
            executor=llm(),
            utils=LLMConfigUtils(outputter=llm(), hopper=llm()),
        ),
    )


def make_elements(turn: int, n_rows: int) -> list[ScreenElement]:
    """造一屏元素表。**每轮都不一样**（滚动 + 文本变化）—— 真机上就是这样。"""
    out: list[ScreenElement] = []
    for i in range(n_rows):
        k = (i + turn * 3) % len(LABELS)
        label = LABELS[k]
        if i % 5 == 0:
            label = f"{label}{i}"
        rid = f"com.example.app:id/{label}_{i}"
        out.append(
            ScreenElement(
                index=i + 1,
                kind="ui",
                label=label,
                text="",
                resource_id=rid,
                display_id="",
                class_name=CLASSES[i % len(CLASSES)],
                bounds=ElementBounds(x=20 * (i % 6), y=120 * i, width=360, height=96),
                flags="CLFS" if i % 3 == 0 else "C",
            )
        )
    return out


def make_history(steps: int) -> list:
    """局部历史：动作 → 工具回执 → 「上一步的结果」。**只追加**（子目标内）。"""
    msgs: list = []
    for i in range(steps):
        call_id = f"call_{i:02d}"
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "tap",  # 生产里历史存的是**设备工具名**，发送时被改写
                        "args": {
                            "target": {"index": 7 + (i % 5)},
                            "agent_thought": f"第 {i + 1} 步：先点搜索框聚焦。",
                        },
                        "id": call_id,
                    }
                ],
            )
        )
        msgs.append(
            ToolMessage(
                content="Tapped element (610, 1258) successfully.",
                tool_call_id=call_id,
            )
        )
        msgs.append(
            HumanMessage(
                content=(
                    f"[上一步的结果] 前台应用没变（还是 com.example.app）；"
                    f"屏幕上新出现了：{LABELS[i % len(LABELS)]}{i}"
                )
            )
        )
    return msgs


def to_openai(messages) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": str(m.content)})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": str(m.content)})
        elif isinstance(m, AIMessage):
            msg: dict = {"role": "assistant", "content": str(m.content or "")}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"], ensure_ascii=False),
                        },
                    }
                    for tc in m.tool_calls
                ]
            out.append(msg)
        elif isinstance(m, ToolMessage):
            out.append(
                {"role": "tool", "tool_call_id": m.tool_call_id, "content": str(m.content)}
            )
    return out


def build_turn(
    ctx: MobileUseContext,
    *,
    turn: int,
    n_rows: int,
    steps: int,
    tag: str = "",
    table_first: bool = False,
    with_table: bool = True,
) -> list[dict]:
    """渲染第 ``turn`` 轮的请求体（走生产组装函数）。

    ``table_first=True`` 复刻**旧布局**（元素表排在局部历史之前），
    用来量「顺序错一次值多少钱」。

    ⚠️ ``tag`` 是**测缓存必须有的东西**：provider 会把每个请求的
    「用户输入末尾」固化成可复用单元（DeepSeek 官方文档：Persistence at
    request boundaries）。所以**同一个请求第二次发，整段都会命中** ——
    上一组序列跑过的请求会污染下一组的首轮。
    ⇒ 每组序列用**唯一的目标文本**当前缀冲刷缓存，各组之间互不污染。
    """
    elements = make_elements(turn, n_rows) if with_table else []
    goal = GOAL if not tag else f"{GOAL}（探针 {tag}）"
    msgs = build_executor_messages(
        ctx=ctx,
        goal=goal,
        screen_elements=elements,
        failures=[],
        local_messages=make_history(steps),
        device_state_lines=[
            "Device date: 2026-10-24 09:20:11",
            "Focused app info: com.example.app",
        ],
    )
    if table_first:
        # 把元素表块（最后一块 user）挪到局部历史前面 —— 只改顺序，内容不变。
        table_msg = msgs[-1]
        rest = msgs[:-1]
        history_end = 1 + 1  # system + goal
        msgs = rest[:history_end] + [table_msg] + rest[history_end:]
    return to_openai(msgs)


# ------------------------------------------------------------------- HTTP 调用


def call(messages: list[dict], tools: list[dict]) -> dict:
    body = {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "ExecutorOutput"}},
        "parallel_tool_calls": False,
        "temperature": 1,
        "max_tokens": 32,
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


# --------------------------------------------------------------------- 离线分析


def block_texts(messages: list[dict]) -> list[str]:
    return [f"{m['role']}:{m.get('content') or ''}" for m in messages]


def _stream(msgs: list[dict]) -> str:
    """把一轮请求拍平成一条内容流 —— 前缀缓存认的是**字节流**，不是块下标。"""
    return "".join(f"{m['role']}:{m.get('content') or ''}\n" for m in msgs)


def _lcp(a: str, b: str) -> int:
    """最长公共前缀长度（字符）。"""
    n = min(len(a), len(b))
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def prefix_relation(a: list[dict], b: list[dict]) -> tuple[int, int, int]:
    """两轮之间的公共前缀。返回 (公共前缀字符数, 结束于 b 的第几块, b 总字符数)。

    ⚠️ **不能用块下标比对** —— 局部历史每轮在中间插入 3 块（动作 / 回执 /
    上一步结果），下标全部平移，看起来像「每块都被改写」。真实情况是
    **纯追加**：老内容一个字节没动，只是后面多了东西。

    ⇒ 判据只能是**内容流的最长公共前缀**。
    """
    sa, sb = _stream(a), _stream(b)
    n = _lcp(sa, sb)
    consumed = 0
    end_block = 0
    for i, m in enumerate(b):
        piece = f"{m['role']}:{m.get('content') or ''}\n"
        if consumed + len(piece) > n:
            end_block = i
            break
        consumed += len(piece)
        end_block = i + 1
    return n, end_block, len(sb)


def first_diff_block(a: list[dict], b: list[dict]) -> tuple[str, str]:
    """相邻两轮的前缀关系（人类可读标签）。"""
    n, end_block, total = prefix_relation(a, b)
    if n >= total:
        return "相同", "逐字节相同"
    role = b[min(end_block, len(b) - 1)]["role"]
    kind = "新增" if n > 0 else "改写"
    return kind, f"第 {end_block} 块({role})起变了（公共前缀 {n:,} 字符）"


def print_composition(ctx: MobileUseContext, n_rows: int) -> list[dict]:
    """打印第 1 轮的组成表：每块的角色 / 字符数 / 每轮是否变。"""
    turn1 = build_turn(ctx, turn=1, n_rows=n_rows, steps=3)
    turn2 = build_turn(ctx, turn=2, n_rows=n_rows, steps=4)
    print("=" * 96)
    print(f"Executor 一轮 prompt 的组成（第 1 轮，{n_rows} 行元素表）")
    print("=" * 96)
    print(f"  {'块':>3}  {'角色':<10} {'字符数':>9}  每轮变?  内容预览")
    for i, m in enumerate(turn1):
        content = str(m.get("content") or "")
        role = m["role"]
        if role == "assistant":
            content = json.dumps(m.get("tool_calls"), ensure_ascii=False)
        changed = i < len(turn2) and block_texts(turn1)[i] != block_texts(turn2)[i]
        preview = content.replace("\n", " ")[:46]
        print(
            f"  {i:>3}  {role:<10} {len(content):>9,}  {'变' if changed else '不变':<6}  {preview}"
        )
    print(f"  合计字符 {sum(len(str(m.get('content') or '')) for m in turn1):,}")
    print()
    return turn1


# ------------------------------------------------------------------------- main


def run_sequence(
    label: str,
    ctx: MobileUseContext,
    *,
    n_rows: int,
    turns: int,
    table_first: bool = False,
    with_table: bool = True,
    tools: list[dict],
) -> tuple[int, int]:
    print("=" * 96)
    print(f"{label}（{turns} 轮，元素表 {n_rows} 行）")
    print("=" * 96)
    total_p = total_c = 0
    prev_msgs: list[dict] | None = None
    for turn in range(1, turns + 1):
        msgs = build_turn(
            ctx,
            turn=turn,
            n_rows=n_rows,
            steps=turn - 1,
            tag=label,
            table_first=table_first,
            with_table=with_table,
        )
        brk = "-"
        if prev_msgs is not None:
            _, brk = first_diff_block(prev_msgs, msgs)
        u = (call(msgs, tools) or {}).get("usage") or {}
        p, c = u.get("prompt_tokens", 0), cached_of(u)
        total_p += p
        total_c += c
        print(
            f"  第 {turn:>2} 轮  prompt={p:>6,}  cached={c:>6,}  "
            f"({c / max(p, 1):>6.1%})   前缀: {brk}"
        )
        prev_msgs = msgs
        time.sleep(0.4)
    rate = total_c / total_p if total_p else 0.0
    print(f"  合计  prompt={total_p:>7,}  cached={total_c:>7,}  命中率={rate:>6.1%}  "
          f"满价重算={total_p - total_c:>7,}\n")
    return total_p, total_c


def offline_prefix_shape(ctx: MobileUseContext, *, n_rows: int, turns: int) -> None:
    """不调 API，只打**前缀结构**：每轮的块、哪几块是稳定的、尾巴有多大。

    这是「为什么命中率是这个数」的静态解释 —— 真 API 的数字在
    ``run_sequence`` 里，两边对得上才算数。
    """
    print("=" * 96)
    print(f"前缀结构（离线，{turns} 轮，{n_rows} 行元素表）")
    print("=" * 96)
    prev = None
    for turn in range(1, turns + 1):
        msgs = build_turn(ctx, turn=turn, n_rows=n_rows, steps=turn - 1)
        total = sum(len(str(m.get("content") or "")) for m in msgs)
        rel = "-"
        stable = 0
        end_block = 0
        if prev is not None:
            n, end_block, _ = prefix_relation(prev, msgs)
            stable = n
            kind, rel = first_diff_block(prev, msgs)
        print(
            f"  第 {turn:>2} 轮  总字符={total:>6,}  与上轮公共前缀={stable:>6,}  "
            f"({stable / max(total, 1):>5.1%})  断在第 {end_block} 块  {rel}"
        )
        prev = msgs
    print()


def main() -> int:
    ctx = _ctx()
    tools = [convert_to_openai_tool(ExecutorOutput)]

    print_composition(ctx, n_rows=60)

    if "--offline" in sys.argv:
        offline_prefix_shape(ctx, n_rows=60, turns=8)
        offline_prefix_shape(ctx, n_rows=0, turns=8)
        return 0

    print("=" * 96)
    print("E0  逐字节确定性 + provider 缓存是否存在")
    print("=" * 96)
    m = build_turn(ctx, turn=1, n_rows=60, steps=3, tag="E0")
    u1 = (call(m, tools) or {}).get("usage") or {}
    time.sleep(1.0)
    u2 = (call(m, tools) or {}).get("usage") or {}
    print(f"  第 1 次 prompt_tokens={u1.get('prompt_tokens')}  cached={cached_of(u1)}")
    print(f"  第 2 次 prompt_tokens={u2.get('prompt_tokens')}  cached={cached_of(u2)}")
    print(f"  同一请求第二次命中 {cached_of(u2) / max(u2.get('prompt_tokens', 1), 1):.1%}"
          "（provider 缓存本身没问题的话应该接近 100%）")
    print()

    print("=" * 96)
    print("E1  生产布局：元素表在最后（当前实现）")
    print("=" * 96)
    p_ok, c_ok = run_sequence(
        "E1 生产布局", ctx, n_rows=60, turns=8, tools=tools
    )

    print("=" * 96)
    print("E2  旧布局：元素表在局部历史之前（只改顺序，内容不变）")
    print("=" * 96)
    p_bad, c_bad = run_sequence(
        "E2 旧布局", ctx, n_rows=60, turns=8, table_first=True, tools=tools
    )

    print("=" * 96)
    print("E3  元素表体积对命中率的影响（尾部长度 = 命中率的死穴）")
    print("=" * 96)
    for rows in (20, 60, 120, 200):
        run_sequence(f"E3 {rows} 行元素表", ctx, n_rows=rows, turns=6, tools=tools)

    print("=" * 96)
    print("E4  上界参考：完全没有元素表（只有 system + 目标 + 历史）")
    print("=" * 96)
    run_sequence("E4 无元素表", ctx, n_rows=0, turns=6, with_table=False, tools=tools)

    print("=" * 96)
    print("小结")
    print("=" * 96)
    print(f"  生产布局 8 轮：命中率 {c_ok / max(p_ok, 1):.1%}")
    print(f"  旧布局   8 轮：命中率 {c_bad / max(p_bad, 1):.1%}")
    print(f"  差值 = {(c_ok / max(p_ok, 1) - c_bad / max(p_bad, 1)) * 100:+.1f} 个百分点"
          f"（满价重算差 {p_ok - c_ok - (p_bad - c_bad):+,} token）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
