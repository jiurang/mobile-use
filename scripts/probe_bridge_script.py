"""端到端：用 **bridge 自己的执行路径** 回放一个编译出来的宏。

这是「宏能不能落地」的真实验证 —— 走的是 bridge 那 19 个工具（和 DSH 侧
``POST /call`` 同一套），不是另写一套。

用法::

    python scripts/probe_bridge_script.py --exec-id 816
    python scripts/probe_bridge_script.py --exec-id 816 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from minitap.mobile_use.bridge.script import Script, run_script  # noqa: E402
from minitap.mobile_use.bridge.session import BridgeSession  # noqa: E402

MANAGER = "http://127.0.0.1:9107"

spec = importlib.util.spec_from_file_location("cm", REPO / "scripts" / "compile_macro.py")
cm = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(cm)  # type: ignore[union-attr]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec-id", type=int, required=True)
    ap.add_argument("--serial", default=None, help="设备 serial（默认第一个）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    macro = cm.compile_macro(args.exec_id, MANAGER)
    steps = macro.get("bridge_steps") or []
    print("=" * 74)
    print("宏:", macro.get("name"), "| 来源执行:", args.exec_id, "| app:", macro.get("app_package"))
    print("可回放的 bridge 步数:", len(steps), "/ 原始步数:", len(macro.get("steps") or []))
    print("=" * 74)
    for i, s in enumerate(steps, 1):
        print("  %2d. %-16s %s" % (i, s["tool"], json.dumps(s["args"], ensure_ascii=False)[:150]))

    if args.dry_run:
        return 0

    script = Script.from_dict(
        {
            "name": macro.get("name"),
            "when_to_use": macro.get("description"),
            "app_package": macro.get("app_package"),
            "device_serial": macro.get("device_serial"),
            "steps": steps,
        }
    )
    session = BridgeSession(device_id=args.serial)
    t0 = time.perf_counter()
    result = await run_script(session, script)
    elapsed = time.perf_counter() - t0

    print()
    print("=" * 74)
    print("回放结果:", result.summary())
    print("总耗时: %.1fs" % elapsed)
    print()
    for s in result.steps:
        print(
            "  %2d. %-16s %s %s"
            % (s["step"], s["tool"], "✅" if s["ok"] else "❌", s.get("message", "")[:110])
        )
    if result.degradations:
        print()
        print("降级记录（脚本要的 resource_id 没命中，用了别的路径）：")
        for d in result.degradations:
            print("   第 %d 步 %s: 想要 %s → 实际 %s" % (d["step"], d["tool"], d["wanted"], d["got"]))
    await session.close()
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
