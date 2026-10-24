"""真 API 实测：**图放在 prompt 的哪个位置，决定缓存命中率**。

要回答的问题
------------
「有截图的智能体，缓存命中率也能到 97%」——它们是怎么做到的？

Anthropic 的做法（依据：本地装的 ``anthropic`` SDK 类型定义）：

* 图片是**内容块**，自带 ``cache_control`` 断点::

      {"type": "image",
       "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."},
       "cache_control": {"type": "ephemeral", "ttl": "5m"}}

* 缓存是**前缀**缓存：断点之前的内容必须**逐字节稳定**，改一个字符整块失效。
  ⇒ 易变的东西（图、当屏描述）必须放在**断点之后** —— 也就是 prompt 的**末尾**。

本脚本在**真端点**（MiniMax M3.1，OpenAI 兼容）上量我们自己的两种布局：

    A. 图在局部历史里（**当前实现**：图挂在工具结果上，历史在 prompt 中段）
    B. 图在 prompt 末尾（**Claude 式**：图跟着当屏描述一起放在最后）

每个布局跑两组：

    变图组：每轮换一张新截图（真机常态）
    不变组：同一张图复用多轮（"97% 命中"长什么样）

⚠️ **每组用唯一 tag 冲刷缓存**（provider 会把每个请求的「用户输入末尾」
固化成可复用单元，上一组的请求会污染下一组的首轮）。
⚠️ **命中率只认真 API 的 usage 字段**，离线推算不算数。

跑法：
    .venv\\Scripts\\python.exe scripts\\probe_image_cache_layout.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minitap"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from minitap.mobile_use.agents.executor.prompt import build_executor_messages  # noqa: E402
from minitap.mobile_use.agents.executor.types import ExecutorOutput  # noqa: E402
from probe_executor_cache_measure import (  # noqa: E402
    _ctx,
    call,
    cached_of,
    load_env,
    make_elements,
    make_history,
)

MODEL = "MiniMax-M3.1-Flash-Preview"
ENV = load_env()
API_KEY = ENV["MINIMAX_API_KEY"]
BASE = ENV.get("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")

#: **每次运行都必须换的冲刷标记。**
#:
#: provider 会把每个请求的「用户输入末尾」固化成可复用单元 ⇒ **逐字节相同的
#: 请求第二次发会整段命中**。上一轮跑过的同一组序列会污染这一组的首轮 ——
#: 实测踩到过：中断后重跑，第 1 轮就报 100% 命中（那是上一轮的缓存，不是本轮的）。
RUN_ID = f"{int(time.time())}-{os.urandom(2).hex()}"

GOAL = "进入 WLAN 页面并读出当前连接的 Wi-Fi 名称"
TURNs = 5
N_ROWS = 40
STEPS0 = 3


# --------------------------------------------------------------------- 素材


def make_jpeg(seed: int) -> str:
    """造一张**每轮都不一样**的假截图（640x1422，和真机缩放后同尺寸）。"""
    img = Image.new("RGB", (640, 1422), (30 + seed * 7 % 200, 90, 140))
    draw = ImageDraw.Draw(img)
    for i in range(12):
        y = 80 + i * 90 + (seed * 13) % 40
        draw.rectangle([40, y, 600, y + 60], fill=(20 + (i * 17 + seed) % 200, 60, 200))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=50, optimize=True)
    import base64

    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_block(jpeg_b64: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"}}


# --------------------------------------------------------------------- 组装


def to_openai(messages) -> list[dict]:
    """langchain 消息 → OpenAI 兼容请求体（图片块原样透传）。"""
    from langchain_core.messages import AIMessage as AI
    from langchain_core.messages import HumanMessage as HM
    from langchain_core.messages import SystemMessage as SM
    from langchain_core.messages import ToolMessage as TM

    out: list[dict] = []
    for m in messages:
        if isinstance(m, SM):
            out.append({"role": "system", "content": str(m.content)})
        elif isinstance(m, HM):
            content = m.content
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
            else:
                out.append({"role": "user", "content": content})
        elif isinstance(m, AI):
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
        elif isinstance(m, TM):
            content = m.content
            if isinstance(content, str):
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": content})
            else:
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": content})
    return out


def build_turn(
    ctx,
    *,
    turn: int,
    layout: str,
    change_image: bool,
    tag: str,
):
    """第 ``turn`` 轮的请求体。

    ``layout="history"``：图挂在**工具结果**上（当前实现）⇒ 图在 prompt 中段。
    ``layout="tail"``   ：图是**最后一块**（Claude 式）⇒ 稳定前缀全在图前面。
    """
    jpeg = make_jpeg(turn if change_image else 0)
    history = list(make_history(STEPS0 + turn - 1))

    if layout in ("history", "history_supersede"):
        # 图在局部历史里：工具结果那条消息带图（生产里就是这么挂的）
        #
        # ``history_supersede`` 复刻**生产的真实行为**：新图来了之后，
        # 旧图那条消息的**内容被替换**成一句话（见 tools/mobile/take_screenshot.py
        # 的 SUPERSEDED_NOTE）⇒ 改的是**历史中间**的字节 ⇒ 会不会断前缀，
        # 这一组就是量它。
        for i in range(1, turn + 1):
            call_id = f"shot_{i}"
            history.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "take_screenshot",
                            "args": {"agent_thought": "元素表不够，我要看一眼"},
                            "id": call_id,
                        }
                    ],
                )
            )
            if layout == "history_supersede" and i < turn:
                history.append(
                    ToolMessage(
                        tool_call_id=call_id,
                        content="(this screenshot was superseded by a newer one)",
                    )
                )
            else:
                history.append(
                    ToolMessage(
                        tool_call_id=call_id,
                        content=[
                            {
                                "type": "text",
                                "text": "Screenshot attached. It supplements the element table.",
                            },
                            image_block(make_jpeg(i) if change_image else jpeg),
                        ],
                    )
                )

    msgs = build_executor_messages(
        ctx=ctx,
        goal=f"{GOAL}（探针 {tag}）",
        screen_elements=make_elements(turn, N_ROWS),
        failures=[],
        local_messages=history,
        device_state_lines=[
            "Device date: 2026-10-24 09:20:11",
            "Focused app info: com.example.app",
        ],
    )

    if layout == "tail":
        # Claude 式：图作为**最后一块**内容（跟着当屏描述一起）
        msgs.append(
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Screenshot of the current screen (supplements the element table).",
                    },
                    image_block(jpeg),
                ]
            )
        )
    return to_openai(msgs)


def call_resilient(messages: list[dict], tools: list[dict]) -> dict:
    """带重试的调用。

    共享的 ``probe_executor_cache_measure.call`` 只重试 429/5xx，**不重试连接
    超时** —— 实测跑到一半 TLS 握手超时，整个探针就死了（30 次调用白跑）。
    这里把网络异常也纳入重试，并在重试之间留够退避。
    """
    last = ""
    for attempt in range(8):
        try:
            return call(messages, tools)
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:160]}"
            print(f"    [重试 {attempt + 1}] {last}")
            time.sleep(3.0 * (attempt + 1))
    raise RuntimeError(last)


# --------------------------------------------------------------------- 主流程


def run(layout: str, change_image: bool, tools: list[dict]) -> dict:
    ctx = _ctx()
    tag = f"{layout}-{'chg' if change_image else 'fix'}-{RUN_ID}"
    label = {
        ("history", True): "A. 图在局部历史里（当前实现）· 每轮换图",
        ("history", False): "A. 图在局部历史里（当前实现）· 同一张图",
        ("history_supersede", True): "C. 图在历史里 + 旧图被顶掉（生产真实行为）",
        ("history_supersede", False): "C. 图在历史里 + 旧图被顶掉 · 同一张图",
        ("tail", True): "B. 图在 prompt 末尾（Claude 式）· 每轮换图",
        ("tail", False): "B. 图在 prompt 末尾（Claude 式）· 同一张图",
    }[(layout, change_image)]
    print("=" * 92)
    print(label)
    print("=" * 92)

    total_p = total_c = 0
    rows = []
    for turn in range(1, TURNs + 1):
        msgs = build_turn(ctx, turn=turn, layout=layout, change_image=change_image, tag=tag)
        u = (call_resilient(msgs, tools) or {}).get("usage") or {}
        p, c = u.get("prompt_tokens", 0), cached_of(u)
        total_p += p
        total_c += c
        rows.append({"turn": turn, "prompt": p, "cached": c, "rate": round(c / max(p, 1), 4)})
        print(f"  第 {turn} 轮  prompt={p:>6,}  cached={c:>6,}  ({c / max(p, 1):>6.1%})")
        time.sleep(0.4)
    rate = total_c / total_p if total_p else 0.0
    print(f"  合计  prompt={total_p:,}  cached={total_c:,}  命中率={rate:.1%}\n")
    return {"label": label, "rows": rows, "prompt": total_p, "cached": total_c, "rate": rate}


def main() -> int:
    tools = [convert_to_openai_tool(ExecutorOutput)]
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")] or [
        "history",
        "history_supersede",
        "tail",
    ]
    results = []
    out_path = ROOT / "probe_out" / "image_cache_layout.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for layout in wanted:
        for change in (True, False):
            r = run(layout, change, tools)
            results.append(r)
            # 边跑边落盘：跑一半被杀也留得住已测的部分
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 92)
    print("小结（真 API 的 usage 字段）")
    print("=" * 92)
    for r in results:
        full_price = r["prompt"] - r["cached"]
        print(f"  {r['label']:<44} 命中率 {r['rate']:>6.1%}   满价重算 {full_price:>7,}")

    def pick(prefix, key):
        for r in results:
            if r["label"].startswith(prefix) and key in r["label"]:
                return r
        return None

    a_chg = pick("A.", "换图")
    b_chg = pick("B.", "换图")
    a_fix = pick("A.", "同一张")
    b_fix = pick("B.", "同一张")
    c_chg = pick("C.", "换图")
    print()
    if a_chg and b_chg:
        print(
            f"  每轮换图：现状(A) {a_chg['rate']:.1%} → 末尾布局(B) {b_chg['rate']:.1%}"
            f"  （{b_chg['rate'] - a_chg['rate']:+.1%}）"
        )
    if a_fix and b_fix:
        print(
            f"  同一张图：现状(A) {a_fix['rate']:.1%} → 末尾布局(B) {b_fix['rate']:.1%}"
            f"  （{b_fix['rate'] - a_fix['rate']:+.1%}）"
        )
    if c_chg and a_chg:
        print(
            f"  旧图被顶掉(C，生产真实行为)：{c_chg['rate']:.1%}"
            f"  （vs A {c_chg['rate'] - a_chg['rate']:+.1%}）"
        )

    out = ROOT / "probe_out" / "image_cache_layout.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  => {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
