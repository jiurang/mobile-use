"""L5 探针：量「工具结果」到底有多长。

要回答的问题
------------
Executor 的**局部历史**里每一步都存着一条工具结果（ToolMessage.content）。
这条内容每轮都被重新发进 prompt —— 它是唯一随步数线性增长的那一段，
所以「它有多长」直接决定「历史成本能不能减半」。

数据源（都是真机跑的产物，不是我编的）
----------------------------------------
1. ``probe_out/dump_run.log`` —— 真机运行的控制台日志，**UTF-16LE**。
   Python 的 logging 在写控制台时因为 GBK 编码失败，会把每条记录打成
   ``--- Logging error --- / Message: <repr> / Arguments: ()`` 的形式。
   ``Message:`` 后面就是那条日志消息的 repr —— 工具结果就在里面。
   ⚠️ 控制台按 119 列硬换行，所以一条长记录会跨几行；本探针把它们接回去。

2. ``probe_out/prompts/**/*.md`` —— 同一批跑法的 **prompt 全文落盘**。
   旧架构里工具结果写进 ``agents_thoughts``（``[executor] ...`` 开头），
   新架构里它们就是 Executor 局部历史里的 ToolMessage —— **同一批字符串**。
   这里没有换行污染，是最干净的测量源。

跑法
----
    uv run python v/probe_tool_result_lengths.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))

# 控制台默认 GBK，中文/符号会炸 —— 强制 UTF-8（Windows 上要先把缓冲区换掉）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

LOG_PATH = ROOT / "probe_out" / "dump_run.log"
PROMPTS_DIR = ROOT / "probe_out" / "prompts"

# 控制台里 logger 打的 ANSI 颜色 + 递归器前缀
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_EMOJI_PREFIX = re.compile(r"^[^\w\[(]{1,4}\s*")

# executor 的「想法」散文（旧架构里 cortex 让它带 "Decisions reason:" 前缀）。
# 它**不是**工具结果 —— 工具结果是我们要量的东西。
_PROSE_PREFIXES = ("Decisions reason:", "Goals completion reason:")


def _strip_decorations(text: str) -> str:
    text = _ANSI.sub("", text)
    text = _EMOJI_PREFIX.sub("", text)
    return text.strip()


def _unescape_repr(raw: str) -> str:
    """把 ``Message:`` 后面的 Python repr 还原成字符串。

    控制台换行会把 repr 拆成多行 → ``ast.literal_eval`` 解析不了，
    这时退回手工反转义（长度统计只需要它足够接近）。
    """
    try:
        value = ast.literal_eval(raw)
        if isinstance(value, str):
            return value
    except (ValueError, SyntaxError):
        pass
    # 手工反转义：去掉首尾引号，处理最常见的几个转义
    body = raw.strip()
    if len(body) >= 2 and body[0] == body[-1] and body[0] in "\"'":
        body = body[1:-1]
    body = body.replace("\\n", "\n").replace("\\t", "\t")
    body = body.replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")
    return body


def extract_log_records(text: str) -> list[str]:
    """从 logging 的错误转储里抽出每条 ``Message:`` 记录（接回被换行拆开的）。"""
    lines = text.splitlines()
    records: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("Message: "):
            buffer = [line[len("Message: ") :]]
            index += 1
            # 换行后的续行一直读到 "Arguments:" 为止
            while index < len(lines) and not lines[index].startswith("Arguments:"):
                buffer.append(lines[index])
                index += 1
            records.append("\n".join(buffer))
        else:
            index += 1
    return records


def classify_tool_results(texts: list[str]) -> list[str]:
    """挑出「工具结果」，剔掉 executor 的想法散文和工具调用转储。"""
    results: list[str] = []
    for raw in texts:
        body = _strip_decorations(raw)
        if "[executor]" not in body:
            continue
        body = body.split("[executor]", 1)[1].strip()
        if body.startswith(_PROSE_PREFIXES):
            continue
        if body.startswith("-"):  # 工具调用的转储
            continue
        if body.startswith("Executing "):  # 「Executing N tool calls:」
            continue
        if not body:
            continue
        results.append(body)
    return results


def extract_from_prompt_dumps() -> list[str]:
    """从 prompt 落盘里抽 ``[executor] ...`` 行 —— 旧架构下那就是工具结果。"""
    results: list[str] = []
    if not PROMPTS_DIR.is_dir():
        return results
    for path in sorted(PROMPTS_DIR.rglob("*.md")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("[executor] "):
                continue
            body = line[len("[executor] ") :].strip()
            if body.startswith(_PROSE_PREFIXES):
                continue
            results.append(body)
    return results


def _percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * ratio))
    return ordered[index]


def _token_len(text: str) -> int:
    """粗略 token 估计：CJK 一字一 token，其余 4 字一 token。"""
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return cjk + (len(text) - cjk) // 4


def report(label: str, results: list[str]) -> None:
    print(f"\n=== {label} —— {len(results)} 条工具结果 ===")
    if not results:
        print("  (没有样本)")
        return
    chars = [len(r) for r in results]
    tokens = [_token_len(r) for r in results]
    print(
        f"  字符数   p50={_percentile(chars, 0.5):>6}  "
        f"p90={_percentile(chars, 0.9):>6}  max={max(chars):>6}"
    )
    print(
        f"  token估  p50={_percentile(tokens, 0.5):>6}  "
        f"p90={_percentile(tokens, 0.9):>6}  max={max(tokens):>6}"
    )
    print("  最长的 5 条：")
    for text in sorted(results, key=len, reverse=True)[:5]:
        preview = text.replace("\n", " / ")
        print(f"    [{len(text):>5} 字] {preview[:150]}")


def main() -> int:
    if LOG_PATH.is_file():
        text = LOG_PATH.read_text(encoding="utf-16")
        records = [_unescape_repr(record) for record in extract_log_records(text)]
        report("来源 1：probe_out/dump_run.log（utf-16）", classify_tool_results(records))
    else:
        print(f"跳过：{LOG_PATH} 不存在")

    report("来源 2：probe_out/prompts/**/*.md（prompt 全文）", extract_from_prompt_dumps())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
