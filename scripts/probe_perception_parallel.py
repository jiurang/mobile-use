"""实测感知并行化前后的关键路径对比。

改动（2026-10-17）
-----------------
改之前 Contextor 一轮本地感知是纯串行：
    screenshot -> dump_hierarchy -> parse_xml -> OCR
改之后：
    dump_hierarchy 在线程池里跑 || 本线程 screenshot -> OCR -> 等 dump

OCR 只有 432ms，dump 要 1188ms，所以 OCR 应该整个藏进 dump 的窗口。
本探针量的是**整段** `get_screen_data(on_screenshot=...)` 的墙上时间，
并和「不传回调 + 事后单独跑 OCR」的旧路径比。

跑法：
    uv run python scripts/probe_perception_parallel.py [device_serial]
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "minitap"))

from minitap.mobile_use.clients.ui_automator_client import (  # noqa: E402
    UIAutomatorClient,
    _parse_hierarchy_xml_to_elements,
    _pil_to_base64,
)
from minitap.mobile_use.services.ocr import OcrConfig, create_ocr_engine  # noqa: E402

ROUNDS = 4


def main() -> int:
    serial = sys.argv[1] if len(sys.argv) > 1 else "89R8ZXNBGQIRJRLF"
    client = UIAutomatorClient(serial)
    client._ensure_connected()
    engine = create_ocr_engine(OcrConfig())  # 预热，模型加载不计入

    # ---- 新路径：OCR 走 on_screenshot 回调 ----
    new_times, ocr_counts = [], []
    for _ in range(ROUNDS):
        box: list[int] = []
        started = time.perf_counter()
        data = client.get_screen_data(on_screenshot=lambda b64: box.append(len(engine.recognize(b64))))
        new_times.append((time.perf_counter() - started) * 1000)
        ocr_counts.append(box[0] if box else -1)
        assert data.elements, "elements empty"

    # ---- 旧路径：串行采集 + 事后单独 OCR ----
    old_times = []
    for _ in range(ROUNDS):
        started = time.perf_counter()
        screenshot = client._ensure_connected().screenshot()
        xml = client._ensure_connected().dump_hierarchy(compressed=True)
        _parse_hierarchy_xml_to_elements(xml)
        engine.recognize(_pil_to_base64(screenshot))
        old_times.append((time.perf_counter() - started) * 1000)

    new_avg = sum(new_times) / len(new_times)
    old_avg = sum(old_times) / len(old_times)

    print(f"\n=== 感知关键路径（{ROUNDS} 轮平均）===")
    print(f"  旧 串行 + 事后 OCR : {old_avg:7.0f} ms   {[round(t) for t in old_times]}")
    print(f"  新 并行 + OCR 内嵌 : {new_avg:7.0f} ms   {[round(t) for t in new_times]}")
    print(f"  节省              : {old_avg - new_avg:7.0f} ms  ({(old_avg - new_avg) / old_avg * 100:.0f}%)")
    print(f"  OCR 条数（新路径回调里拿到）: {ocr_counts}")
    assert all(c >= 0 for c in ocr_counts), "on_screenshot 回调没有被调用"

    print(f"\n=== 校验：OCR 确实在新路径里跑到了 ===")
    assert all(c > 0 for c in ocr_counts), f"OCR 返回 0 条：{ocr_counts}"
    print(f"  OK — 每轮回调都拿到了 OCR 结果，回调链路是通的")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
