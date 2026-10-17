"""校验 llm-config.override.jsonc 里每个节点的 provider / model / fallback。

为什么需要它
------------
`llm-config.override.jsonc` 是 **gitignore 的本地文件**，改错了没有任何测试会红，
也没有任何东西会提醒你。上一次就出现过「口头说撤掉了、文件里其实还在」的情况：
DeepSeek 兜底被宣布撤回，实际文件里 7 个节点全都还挂着，跑的时候它照样生效。

所以规则要变成**可执行的**，而不是靠记性：
    uv run python scripts/verify_llm_override.py
    uv run python scripts/verify_llm_override.py --expect-model MiniMax-M3.1-Flash-Preview

退出码 0 = 全部符合；非 0 = 有节点不符合，脚本会把差异逐条列出来。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OVERRIDE = ROOT / "llm-config.override.jsonc"

# agent 实际会用的节点：顶层 5 个 + utils 下 2 个。video_analyzer 可以是 null。
NODES = ["planner", "orchestrator", "cortex", "executor", "contextor", "hopper", "outputter"]


def load_stripped_comments(path: Path) -> tuple[dict, str]:
    """去掉 // 注释再解析。jsonc 的注释在 json.loads 里会直接炸。

    `utf-8-sig` 是刻意的：PowerShell 5.1 的 `Set-Content -Encoding UTF8` 会写 BOM，
    很多编辑器也会。带 BOM 的配置本身是合法 jsonc，不该因此报解析错。
    """
    raw = path.read_text(encoding="utf-8-sig")
    stripped = re.sub(r"^\s*//.*$", "", raw, flags=re.M)
    return json.loads(stripped), raw


def report_parse_failure(path: Path, exc: Exception) -> int:
    """配置坏掉时要给出可操作的诊断，而不是甩一个 traceback。

    坏配置最常见的原因就是编码（带 BOM / GBK），所以直接把第一行前几个字符
    的 repr 打出来，让人一眼看出是不是编码问题。
    """
    print(f"FAIL {path.name} 不是合法 jsonc：{exc}")
    try:
        head = path.read_bytes()[:16]
        print(f"  文件头 16 字节: {head!r}")
        if head.startswith(b"\xef\xbb\xbf"):
            print("  → 带 UTF-8 BOM。PowerShell 5.1 的 Set-Content -Encoding UTF8 会加这个；")
            print("    改文件请用编辑器或文件工具，不要用 Set-Content。")
    except OSError:
        pass
    return 1


def flatten(cfg: dict) -> dict[str, dict]:
    utils = cfg.get("utils") or {}
    out = {k: cfg[k] for k in NODES if k in cfg}
    for k in ("hopper", "outputter"):
        if k in utils:
            out[k] = utils[k]
    return out


def collect_providers(node: object, out: list[str]) -> None:
    """递归收集解析后配置里出现的所有 provider / model 字符串。

    刻意**不扫原始文本**：注释里写「DeepSeek 开 thinking 时绑定工具会被 400 拒」
    是有价值的背景说明，把它当失败项会逼人把注释删掉或改写，最后丢的是信息。
    真正要守的不变式是「没有任何活着的节点指向被禁用的 provider」。
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("provider", "model") and isinstance(value, str):
                out.append(value)
            else:
                collect_providers(value, out)
    elif isinstance(node, list):
        for item in node:
            collect_providers(item, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-provider", default="minimax")
    ap.add_argument("--expect-model", default=None, help="不传则只检查 provider，不检查模型名")
    ap.add_argument("--forbid", default="deepseek", help="解析后的配置里出现即失败（注释不计入）")
    args = ap.parse_args()

    if not OVERRIDE.exists():
        print(f"FAIL 没有 {OVERRIDE}")
        return 1

    cfg: dict
    try:
        cfg, _raw = load_stripped_comments(OVERRIDE)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return report_parse_failure(OVERRIDE, exc)
    nodes = flatten(cfg)

    problems: list[str] = []
    missing = [n for n in NODES if n not in nodes]
    if missing:
        problems.append(f"缺节点: {', '.join(missing)}")

    print(f"检查 {OVERRIDE.name}（{len(nodes)} 个节点）\n")
    for name in NODES:
        n = nodes.get(name)
        if not n:
            continue
        provider, model = n.get("provider"), n.get("model")
        fb = n.get("fallback") or {}
        fb_desc = f"{fb.get('provider')}/{fb.get('model')}" if fb else "（无）"
        print(f"  {name:<14} {provider}/{model:<28} fallback={fb_desc}")

        if provider != args.expect_provider:
            problems.append(f"{name}: provider={provider!r}，期望 {args.expect_provider!r}")
        if args.expect_model and model != args.expect_model:
            problems.append(f"{name}: model={model!r}，期望 {args.expect_model!r}")
        if fb and fb.get("provider") != args.expect_provider:
            problems.append(f"{name}: fallback provider={fb.get('provider')!r}，期望 {args.expect_provider!r}")

    used: list[str] = []
    collect_providers(cfg, used)
    hits = sorted({u for u in used if args.forbid.lower() in u.lower()})
    if hits:
        problems.append(f"解析后的配置里仍有节点指向 {args.forbid}: {', '.join(hits)}")

    print()
    if problems:
        print("FAIL 以下问题：")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"PASS 全部 {len(nodes)} 个节点都是 {args.expect_provider}"
          + (f"/{args.expect_model}" if args.expect_model else "")
          + f"，且没有任何节点指向 {args.forbid!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
