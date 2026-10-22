"""宏的命令行入口 —— 没有 bridge 也能把宏跑起来。

::

    # 查候选
    python -m minitap.mobile_use.macros search --q 快手

    # 取下来跑（需要连着设备）
    python -m minitap.mobile_use.macros run --id 123 --serial 89R8ZXNBGQIRJRLF

    # 把一份宏存进脚本库
    python -m minitap.mobile_use.macros save --file probe_out/macro_exec816.json \
        --name ks_feed_like --when-to-use "在快手极速版给当前视频点赞"

    # 从 manager 的执行记录现编一个
    python -m minitap.mobile_use.macros compile --exec-id 816 --save
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m minitap.mobile_use.macros")
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="查候选宏")
    p_search.add_argument("--q", default="", help="关键词")
    p_search.add_argument("--app", default=None, help="包名")
    p_search.add_argument("--limit", type=int, default=5)

    p_run = sub.add_parser("run", help="取一个宏下来跑")
    p_run.add_argument("--id", type=int, required=True)
    p_run.add_argument("--serial", default=None, help="设备 serial（默认第一个）")
    p_run.add_argument("--file", default=None, help="直接跑本地的一个宏文件（不连 manager）")

    p_save = sub.add_parser("save", help="把一份宏存进脚本库")
    p_save.add_argument("--file", required=True, help="宏文件（JSON）")
    p_save.add_argument("--name", default=None)
    p_save.add_argument("--when-to-use", default=None, dest="when_to_use")

    p_compile = sub.add_parser("compile", help="从 manager 的执行记录编一个宏")
    p_compile.add_argument("--exec-id", type=int, required=True)
    p_compile.add_argument("--save", action="store_true", help="存进脚本库")
    p_compile.add_argument("--name", default=None)
    p_compile.add_argument("--when-to-use", default=None, dest="when_to_use")
    return parser


def _load_macro_file(path: str) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    steps = raw.get("bridge_steps") or raw.get("steps") or []
    return {
        "name": raw.get("name") or "macro",
        "when_to_use": raw.get("description") or raw.get("when_to_use") or "",
        "app_package": raw.get("app_package"),
        "device_serial": raw.get("device_serial"),
        "steps": steps,
    }


async def _run(args: argparse.Namespace) -> int:
    from minitap.mobile_use.clients.ui_automator_client import UIAutomatorClient
    from minitap.mobile_use.config import initialize_llm_config
    from minitap.mobile_use.context import (
        DeviceContext,
        DevicePlatform,
        MobileUseContext,
        PerceptionConfig,
    )
    from minitap.mobile_use.macros.device_env import make_device_env
    from minitap.mobile_use.macros.model import Macro
    from minitap.mobile_use.macros.runner import run_macro
    from minitap.mobile_use.macros.store import MacroStore

    if args.file:
        raw = _load_macro_file(args.file)
    else:
        raw = MacroStore().get(args.id)

    macro = Macro.from_dict(raw)
    print("宏:", macro.name, "| app:", macro.app_package, "| 设备:", macro.device_serial)
    print("步数:", len(macro.steps))
    for i, step in enumerate(macro.steps, 1):
        print("  %2d. %-16s %s" % (i, step.tool, json.dumps(step.args, ensure_ascii=False)[:120]))

    import uiautomator2 as u2
    from adbutils import AdbClient

    serial = args.serial
    if not serial:
        devices = [d for d in u2.list_devices() if d]
        if not devices:
            print("没有连着的设备")
            return 1
        serial = devices[0]
    info = u2.connect(serial).info
    ctx = MobileUseContext(
        trace_id="macro-cli",
        device=DeviceContext(
            host_platform="WINDOWS",
            mobile_platform=DevicePlatform.ANDROID,
            device_id=serial,
            device_width=int(info["displayWidth"]),
            device_height=int(info["displayHeight"]),
        ),
        adb_client=AdbClient(),
        ui_adb_client=UIAutomatorClient(device_id=serial),
        llm_config=initialize_llm_config(),
        perception=PerceptionConfig(),
    )

    env = await make_device_env(ctx)
    t0 = time.perf_counter()
    result = await run_macro(env, macro)
    elapsed = time.perf_counter() - t0

    print()
    print("结果:", result.summary())
    print("耗时: %.1fs" % elapsed)
    for step in result.steps:
        print(
            "  %2d. %-16s %s %s"
            % (step["step"], step["tool"], "✅" if step["ok"] else "❌", step.get("message", "")[:110])
        )
    if result.degradations:
        print()
        print("降级记录（宏要的 resource_id 没命中，用了别的路径）：")
        for d in result.degradations:
            print("   第 %d 步 %s: 想要 %s → 实际 %s" % (d["step"], d["tool"], d["wanted"], d["got"]))
    return 0 if result.ok else 1


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "search":
        from minitap.mobile_use.macros.store import MacroStore

        candidates = MacroStore().find(args.q, args.app, args.limit)
        if not candidates:
            print("没有匹配的宏")
            return 0
        for c in candidates:
            print(c.summary())
        return 0

    if args.command == "save":
        from minitap.mobile_use.macros.store import MacroStore

        macro = _load_macro_file(args.file)
        if args.name:
            macro["name"] = args.name
        if args.when_to_use:
            macro["when_to_use"] = args.when_to_use
        steps = macro.pop("steps", [])
        macro["stepsJson"] = json.dumps(steps, ensure_ascii=False)
        macro["stepCount"] = len(steps)
        receipt = MacroStore().save(macro)
        print("已保存，id =", receipt.get("id"))
        return 0

    if args.command == "compile":
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "compile_macro", Path(__file__).resolve().parents[3] / "scripts" / "compile_macro.py"
        )
        cm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cm)
        from minitap.mobile_use.macros.store import manager_url

        raw = cm.compile_macro(args.exec_id, manager_url())
        steps = raw.get("bridge_steps") or []
        macro = {
            "name": args.name or raw["name"],
            "when_to_use": args.when_to_use or raw.get("description") or "",
            "app_package": raw.get("app_package"),
            "device_serial": raw.get("device_serial"),
            "steps": steps,
        }
        out = Path(f"probe_out/macro_exec{args.exec_id}.json")
        out.write_text(json.dumps(macro, ensure_ascii=False, indent=2), encoding="utf-8")
        print("已写出:", out, f"（{len(steps)} 步）")
        if args.save:
            from minitap.mobile_use.macros.store import MacroStore

            payload = {k: v for k, v in macro.items() if k != "steps"}
            payload["stepsJson"] = json.dumps(steps, ensure_ascii=False)
            payload["stepCount"] = len(steps)
            receipt = MacroStore().save(payload)
            print("已存进脚本库，id =", receipt.get("id"))
        return 0

    if args.command == "run":
        return asyncio.run(_run(args))

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
