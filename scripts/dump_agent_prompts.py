"""跑一个**真任务**，把每一轮真正发给 LLM 的 prompt 落盘成文件。

为什么不直接读 cortex.md
------------------------
cortex.md 只是 system prompt 的**模板源**。真正发出去的是
「模板渲染结果 + 设备信息 + 历轮 thoughts + 层级树 + OCR + 截图」拼成的
message 数组。想知道模型实际看到的原样，只能在**调用点**截获。

截获点选在哪
------------
所有 agent 都是同一个写法（见 agents/*/*.py）::

    llm = get_llm(ctx=..., name="cortex", temperature=1).with_structured_output(...)
    await llm.ainvoke(messages)

`get_llm` 是每个模块自己 `from ... import` 进来的，所以把各 agent 模块里的
`get_llm` 换成带记录的包装，`.ainvoke(messages)` 就能拿到原始 messages ——
包括 `with_structured_output` 之后的那一层。用 `__getattr__` 兜底转发，
对被包装对象透明，不改生产代码一行。

产物
----
``probe_out/prompts/<run>/`` 下：
    ``index.md``                每次调用一行（agent / 消息数 / 字数）
    ``<n>_<agent>.md``          单次调用的完整 prompt（文本原样）
    ``<n>_<agent>.img<k>.jpg``  消息里内嵌的截图（base64 解出来）

跑法::

    uv run python scripts/dump_agent_prompts.py --goal "打开设置并返回桌面"
    uv run python scripts/dump_agent_prompts.py --goal "..." --max-per-agent 5
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import importlib
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MOBILE_USE_TELEMETRY_ENABLED", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))

# .env 里才有 MINIMAX_API_KEY，配置加载顺序不保证在 import 之前
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:  # noqa: BLE001
    pass

AGENT_MODULES = {
    "planner": "minitap.mobile_use.agents.planner.planner",
    "orchestrator": "minitap.mobile_use.agents.orchestrator.orchestrator",
    "cortex": "minitap.mobile_use.agents.cortex.cortex",
    "executor": "minitap.mobile_use.agents.executor.executor",
    "contextor": "minitap.mobile_use.agents.contextor.contextor",
    "hopper": "minitap.mobile_use.agents.hopper.hopper",
    "outputter": "minitap.mobile_use.agents.outputter.outputter",
    "video_analyzer": "minitap.mobile_use.agents.video_analyzer.video_analyzer",
}

_DATA_URL = re.compile(r"data:(image/[a-zA-Z]+);base64,(.*)", re.S)


def _fence(text: str) -> str:
    """用比内容里最长反引号串更长的围栏，避免 prompt 自带的 ``` 提前闭合。"""
    runs = [len(m) for m in re.findall(r"`+", text)]
    fence = "`" * max(3, (max(runs) if runs else 0) + 1)
    return f"{fence}text\n{text}\n{fence}"


class Recorder:
    def __init__(self, out_dir: Path, max_per_agent: int):
        self.out_dir = out_dir
        self.max_per_agent = max_per_agent
        self.entries: list[dict] = []
        self.counts: dict[str, int] = {}

    # -- 落盘 ---------------------------------------------------------------

    def record(self, agent: str, model: str, messages: list) -> None:
        n = self.counts.get(agent, 0)
        self.counts[agent] = n + 1
        if n >= self.max_per_agent:
            return

        self.out_dir.mkdir(parents=True, exist_ok=True)
        idx = len(self.entries) + 1
        stem = f"{idx:03d}_{agent}"

        blocks: list[tuple[str, str, int, list[Path]]] = []
        text_chars = 0
        img_no = 0
        for msg in messages:
            body, chars, imgs = self._render(str(msg.content) if isinstance(msg.content, str) else msg.content, stem, img_no)
            img_no += len(imgs)
            text_chars += chars
            blocks.append((type(msg).__name__, body, chars, imgs))

        images = [p for _, _, _, imgs in blocks for p in imgs]
        lines = [
            f"# {agent} — call {n + 1}",
            "",
            f"- model: `{model}`",
            f"- messages: {len(messages)}",
            f"- text chars: {text_chars:,}",
            f"- images: {len(images)}",
            "",
        ]
        for i, (kind, body, chars, _imgs) in enumerate(blocks, start=1):
            lines += [f"## {i}. {kind}  ·  {chars:,} chars", "", _fence(body), ""]

        (self.out_dir / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
        self.entries.append(
            {
                "idx": idx,
                "agent": agent,
                "call": n + 1,
                "model": model,
                "messages": len(messages),
                "chars": text_chars,
                "images": images,
                "file": f"{stem}.md",
            }
        )

    # -- 内容渲染 -----------------------------------------------------------

    def _render(self, content, stem: str, img_no: int) -> tuple[str, int, list[Path]]:
        """把 message content 变成可读文本。图片解码落盘，正文留占位行。"""
        if isinstance(content, str):
            return content, len(content), []

        chunks: list[str] = []
        chars = 0
        images: list[Path] = []

        for block in content or []:
            if isinstance(block, str):
                chunks.append(block)
                chars += len(block)
                continue
            if not isinstance(block, dict):
                chunks.append(str(block))
                continue

            btype = block.get("type")
            if btype == "image_url":
                url = (block.get("image_url") or {}).get("url", "")
                m = _DATA_URL.match(url)
                if not m:
                    chunks.append(f"[IMAGE non-data url: {url[:80]}]")
                    continue
                mime, b64 = m.group(1), m.group(2)
                ext = mime.split("/")[-1].replace("jpeg", "jpg")
                img_no += 1
                path = self.out_dir / f"{stem}.img{img_no}.{ext}"
                try:
                    path.write_bytes(base64.b64decode(b64))
                except Exception as e:  # noqa: BLE001
                    chunks.append(f"[IMAGE decode failed: {type(e).__name__}: {e}]")
                    continue
                images.append(path)
                kb = path.stat().st_size // 1024
                chunks.append(f"[IMAGE {path.name} — {mime}, {kb} KB, 见同目录文件]")
            elif btype == "text":
                text = block.get("text", "")
                chunks.append(text)
                chars += len(text)
            else:
                chunks.append(f"[{btype} block] {str(block)[:300]}")

        return "\n".join(chunks), chars, images


def _detect_model(obj) -> str:
    """从 chat model 上摸出模型名。

    必须在 ``with_structured_output`` **之前**问：包完之后外面那层是
    ``Runnable``，model 字段已经摸不到了。
    """
    for attr in ("model", "model_name", "model_id", "deployment_name"):
        try:
            value = getattr(obj, attr, None)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(value, str) and value:
            return value
    return "?"


class _RecordingLLM:
    """透明代理：在 ainvoke 时把 messages 交给 Recorder。"""

    def __init__(self, inner, agent: str, recorder: Recorder, model: str):
        self.__dict__["_inner"] = inner
        self.__dict__["_agent"] = agent
        self.__dict__["_rec"] = recorder
        self.__dict__["_model"] = model

    @property
    def _model_name(self) -> str:
        inner = self.__dict__["_inner"]
        for attr in ("model", "model_name", "model_id", "deployment_name"):
            try:
                value = getattr(inner, attr, None)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(value, str):
                return value
        return "?"

    def with_structured_output(self, schema=None, **kwargs):
        inner = self.__dict__["_inner"].with_structured_output(schema, **kwargs)
        return _RecordingLLM(
            inner, self.__dict__["_agent"], self.__dict__["_rec"], self.__dict__["_model"]
        )

    async def ainvoke(self, messages, *args, **kwargs):
        self.__dict__["_rec"].record(
            self.__dict__["_agent"], self.__dict__["_model"], list(messages)
        )
        return await self.__dict__["_inner"].ainvoke(messages, *args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.__dict__["_inner"], item)


def install_patches(recorder: Recorder) -> list[str]:
    """把每个 agent 模块里的 get_llm 换成带记录的包装。"""
    patched: list[str] = []
    for agent, mod_path in AGENT_MODULES.items():
        try:
            mod = importlib.import_module(mod_path)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {agent}: import failed ({type(e).__name__}: {e})")
            continue
        orig = getattr(mod, "get_llm", None)
        if orig is None:
            print(f"  [skip] {agent}: no get_llm")
            continue

        def make(orig=orig, agent=agent):
            def patched(*args, **kwargs):
                llm = orig(*args, **kwargs)
                # model 名在这里取：此时还没被 with_structured_output 包成 Runnable
                return _RecordingLLM(llm, agent, recorder, _detect_model(llm))

            return patched

        mod.get_llm = make()
        patched.append(agent)
    return patched


async def run(goal: str, out_dir: Path, max_per_agent: int, timeout: int) -> int:
    from minitap.mobile_use.main import run_automation

    recorder = Recorder(out_dir, max_per_agent)
    print(f"[patch] get_llm wrapped in: {', '.join(install_patches(recorder))}")

    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(run_automation(goal=goal), timeout=timeout)
    except Exception as e:  # noqa: BLE001
        # 任务失败也要留下已经截到的 prompt
        print(f"[task] ended with {type(e).__name__}: {e}")
    dt = time.perf_counter() - t0

    lines = [
        "# prompt dump — index",
        "",
        f"- goal: `{goal}`",
        f"- wall time: {dt:.1f}s",
        f"- calls recorded: {len(recorder.entries)}",
        f"- per-agent max: {max_per_agent}",
        "",
        "| # | agent | call | model | msgs | chars | images |",
        "|---|-------|------|-------|------|-------|--------|",
    ]
    for e in recorder.entries:
        lines.append(
            f"| {e['idx']} | {e['agent']} | {e['call']} | `{e['model']}` | "
            f"{e['messages']} | {e['chars']:,} | {len(e['images'])} |"
        )
    if not recorder.entries:
        lines.append("| - | _没有截到任何调用_ | | | | | |")
    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[task] {dt:.1f}s -> 记录 {len(recorder.entries)} 次 prompt，输出 {out_dir}")
    for e in recorder.entries:
        print(
            f"   {e['idx']:>3} {e['agent']:<12} call#{e['call']} "
            f"msgs={e['messages']:<3} chars={e['chars']:<8,} {e['file']}"
        )
    return 0 if recorder.entries else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--goal", default="打开设置，进入 WLAN 页面，告诉我当前连接的 Wi-Fi 名称")
    p.add_argument("--max-per-agent", type=int, default=3)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--out", default=None, help="输出目录，默认 probe_out/prompts/<时间戳>")
    a = p.parse_args()

    out = Path(a.out) if a.out else ROOT / "probe_out" / "prompts" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[out] {out}")
    return asyncio.run(run(a.goal, out, a.max_per_agent, a.timeout))


if __name__ == "__main__":
    raise SystemExit(main())
