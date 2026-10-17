"""验证 uiautomator2 的 screenshot 和 dump_hierarchy 能否真正并发。

为什么必须先量这个
------------------
Contextor 现在是纯串行（真机实测 194 + 1100 + 2 + 432 = 1728ms/轮）。
理论上让两个采集并发、OCR 藏进 dump 的窗口里，关键路径能压到 ~1100ms。

但这个理论只在「设备端 u2 agent 真的并发处理两个 HTTP 请求」时成立。
如果 agent 是单线程串行处理，那并发调用不但不省时间，还会因为两个线程
抢同一条 HTTP 连接而引入新的不确定性。所以先测，再改。

量三组：
    A 串行基线：screenshot -> dump
    B 并发：ThreadPoolExecutor 同时打两个
    C 并发稳定性：连跑 5 轮，看有没有偶发变慢/失败

跑法：
    uv run python scripts/probe_u2_concurrency.py [device_serial]
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "minitap"))

from minitap.mobile_use.clients.ui_automator_client import UIAutomatorClient  # noqa: E402


def main() -> int:
    serial = sys.argv[1] if len(sys.argv) > 1 else "89R8ZXNBGQIRJRLF"
    client = UIAutomatorClient(serial)
    device = client._ensure_connected()

    def shot():
        t = time.perf_counter()
        device.screenshot()
        return (time.perf_counter() - t) * 1000

    def dump():
        t = time.perf_counter()
        device.dump_hierarchy(compressed=True)
        return (time.perf_counter() - t) * 1000

    # ---- A. 串行基线 ----
    serial_shots, serial_dumps = [], []
    for _ in range(3):
        s = shot()
        d = dump()
        serial_shots.append(s)
        serial_dumps.append(d)
    a_total = (sum(serial_shots) + sum(serial_dumps)) / 3
    print("\n=== A. 串行基线（3 轮平均）===")
    print(f"  screenshot {sum(serial_shots) / 3:7.0f} ms")
    print(f"  dump       {sum(serial_dumps) / 3:7.0f} ms")
    print(f"  合计       {a_total:7.0f} ms")

    # ---- B/C. 并发 + 稳定性 ----
    print("\n=== B. 并发（ThreadPoolExecutor，5 轮）===")
    conc_shots, conc_dumps, conc_totals = [], [], []
    failures = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i in range(5):
            t0 = time.perf_counter()
            f_shot = pool.submit(shot)
            f_dump = pool.submit(dump)
            try:
                s, d = f_shot.result(), f_dump.result()
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"  round {i + 1}: FAILED {type(e).__name__}: {e}")
                continue
            wall = (time.perf_counter() - t0) * 1000
            conc_shots.append(s)
            conc_dumps.append(d)
            conc_totals.append(wall)
            print(
                f"  round {i + 1}: wall {wall:6.0f} ms  "
                f"(shot {s:5.0f} + dump {d:5.0f} = {s + d:5.0f} 串行等价, "
                f"重叠省下 {max(0, s + d - wall):5.0f} ms)"
            )

    if conc_totals:
        b_total = sum(conc_totals) / len(conc_totals)
        overlap = sum(conc_shots) / len(conc_shots) + sum(conc_dumps) / len(conc_dumps) - b_total
        print(f"\n  并发平均 wall {b_total:.0f} ms vs 串行 {a_total:.0f} ms")
        print(f"  重叠量 {overlap:.0f} ms/轮  ({overlap / a_total * 100:.0f}%)")
        print(f"  失败 {failures}/5")
        verdict = "可并发" if overlap > 150 and failures == 0 else "不可并发（保持串行）"
        print(f"\n  => 结论: {verdict}")

    # ---- D. 叠加 OCR：dump 窗口能不能藏下 OCR ----
    print("\n=== D. 关键路径推演 ===")
    avg_shot = sum(conc_shots) / len(conc_shots) if conc_shots else 0
    avg_dump = sum(conc_dumps) / len(conc_dumps) if conc_dumps else 0
    ocr_ms = 432  # probe_cortex_prefill.py 实测
    parse_ms = 2  # 实测
    print(f"  现在(串行+OCR): {avg_shot + avg_dump + parse_ms + ocr_ms:.0f} ms")
    print(f"  并发后      : max({avg_shot:.0f},{avg_dump:.0f}) + max({avg_shot:.0f}+{ocr_ms:.0f},{parse_ms:.0f})"
          f" = {max(avg_shot, avg_dump) + max(avg_shot + ocr_ms, parse_ms):.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
