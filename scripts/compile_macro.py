"""从 manager-server 的执行 trace 编译出可回放的宏。

为什么能编译
------------
trace 里每次动作都记了**两样**东西：

* 模型吐的动作：``{"action": "tap", "target": {"index": 7}, ...}``
* 执行结果：``Tap on element with resource_id='com.x:id/foo' -> coordinates (609, 1329) was successful.``

⇒ **稳定选择器（resource_id）是白送的**，不用改任何生产代码。

index 定位的那部分怎么办
------------------------
模型报的是 ``index``（当屏元素表的行号），跨运行必失效。但：

* 结果文字里有**当时点中的坐标**（``coordinates (609, 1329)``）；
* trace 的 prompt 里有**当时的元素表**（每行带 ``id=`` 尾段和 ``(x,y,WxH)``）。

⇒ 用坐标在元素表里反查那一行，拿到它的 ``id=`` 尾段 —— 即可回放的稳定选择器。

用法::

    python scripts/compile_macro.py --exec-id 816 --out probe_out/macro_816.json
    python scripts/compile_macro.py --exec-id 816 --print-steps
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minitap.mobile_use.utils.app_names import APP_NAMES_ZH  # noqa: E402
from minitap.mobile_use.utils.screen_fingerprint import parse_context  # noqa: E402

#: 新 trace 的「前台应用」已经是**中文应用名**（见 utils/app_names.py）；
#: 宏的前置条件要的是包名 ⇒ 认识的换回来，不认识的原样当包名用。
_DISPLAY_NAME_TO_PACKAGE = {name: package for package, name in APP_NAMES_ZH.items()}

RESOURCE_RE = re.compile(r"resource_id='([^']+)'")
INDEX_COORD_RE = re.compile(r"index=(\d+)\s+coordinates \((\d+),\s*(\d+)\)")
COORD_RE = re.compile(r"coordinates \((\d+),\s*(\d+)\)")
ROW_RE = re.compile(
    r'^\[(\d+)\]\s*"(?P<text>.*?)"\s*(?:\|\s*id=(?P<id>\S+)\s*)?\|?\s*'
    r"(?P<cls>\S*)\s*\|?\s*\((?P<x>\d+),(?P<y>\d+),(?P<w>\d+)x(?P<h>\d+)\)"
)


def get(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def text_of(msg: dict) -> str:
    return " ".join(b.get("text", "") for b in (msg.get("blocks") or []) if b.get("type") == "text")


def parse_element_table(prompt: dict) -> dict[int, dict]:
    """从 prompt 里把「当屏元素表」解析成 ``{index: {text, id, cls, x, y, w, h}}``。"""
    rows: dict[int, dict] = {}
    for m in prompt.get("messages") or []:
        if m.get("role") != "user":
            continue
        for b in m.get("blocks") or []:
            if b.get("type") != "text":
                continue
            for line in (b.get("text") or "").splitlines():
                mm = ROW_RE.match(line.strip())
                if not mm:
                    continue
                rows[int(mm.group(1))] = {
                    "text": mm.group("text"),
                    "id_tail": mm.group("id") or "",
                    "cls": mm.group("cls") or "",
                    "x": int(mm.group("x")),
                    "y": int(mm.group("y")),
                    "w": int(mm.group("w")),
                    "h": int(mm.group("h")),
                }
    return rows


def _center(row: dict) -> tuple[int, int]:
    return row["x"] + row["w"] // 2, row["y"] + row["h"] // 2


def resolve_stable_selector(result_text: str, rows: dict[int, dict]) -> tuple[str | None, str]:
    """从执行结果里恢复出稳定选择器。

    返回 ``(resource_id, 怎么得到的)``。``resource_id`` 可能是完整 id，也可能是
    **尾段**（``id=`` 只在文案缺失或重复时才出现在元素表里）—— 尾段够用了，
    回放时按尾段匹配即可。
    """
    m = RESOURCE_RE.search(result_text)
    if m:
        return m.group(1), "result_resource_id"

    m = INDEX_COORD_RE.search(result_text)
    if not m:
        m = COORD_RE.search(result_text)
        if not m:
            return None, "no_selector_in_result"
        idx, x, y = None, int(m.group(1)), int(m.group(2))
    else:
        idx, x, y = int(m.group(1)), int(m.group(2)), int(m.group(3))

    # 1) 先按 index 精确取那一行
    candidates = []
    if idx is not None and idx in rows:
        candidates.append(rows[idx])
    # 2) 再按坐标中心匹配（index 可能是当轮的，也可能是历史轮的）
    for r in rows.values():
        cx, cy = _center(r)
        if abs(cx - x) <= 8 and abs(cy - y) <= 8:
            candidates.append(r)
    for r in candidates:
        if r["id_tail"]:
            return r["id_tail"], "element_table_id_tail"
    return None, "no_stable_selector"


FOCUS_PACKAGE_RE = re.compile(r"Focused app info:\s*(\S+)")
#: 任务的目标 app 包名（assembledPrompt 里：「快手极速版（包名 com.kuaishou.nebula）」）
TASK_PACKAGE_RE = re.compile(r"包名\s*([A-Za-z0-9_.]+)")

# ---------------------------------------------------------------------------
# executor 工具名 → bridge 工具名
# ---------------------------------------------------------------------------
#
# trace 里的动作是 **executor** 的工具（``swipe_percentages`` /
# ``focus_and_input_text`` …），而脚本是给 **bridge** 跑的（``swipe`` /
# ``input_text`` …）。两边名字不一样，参数形状也不一样 ⇒ 必须显式映射。
#
# 映射表也顺手解决一件事：``launch_app`` 在 executor 那边收的是**应用名**
# （要调 hopper 小模型翻成包名），在 bridge 那边直接收**包名** —— 宏里存包名，
# 回放时零模型调用。
_BRIDGE_TOOL_MAP = {
    "tap": "tap",
    "long_press_on": "long_press",
    "swipe_percentages": "swipe",
    "swipe_coordinates": "swipe",
    "focus_and_input_text": "input_text",
    "focus_and_clear_text": "clear_text",
    "erase_one_char": "erase_one_char",
    "press_key": "press_key",
    "back": "back",
    "wait_for_delay": "wait",
    "launch_app": "launch_app",
    "stop_app": "stop_app",
}

#: bridge 里没有对应工具的动作（scratchpad 笔记那三个）→ 丢掉。
_UNSUPPORTED_ACTIONS = {"save_note", "read_note", "list_notes"}


def to_bridge_step(action: str, args: dict, selector: str | None = None) -> dict | None:
    """把一个 executor 动作翻译成 bridge 的工具调用。翻不了就返回 None。

    ⚠️ **resource_id 不在动作参数里** —— executor 那边模型报的是 ``index``，
    完整的 id 是**执行时解析出来的**，只出现在动作结果里（编译器已经存进
    ``stable_selector``）。所以这里用 ``selector``，不是 ``args['target']``。
    """
    if action in _UNSUPPORTED_ACTIONS:
        return None
    tool = _BRIDGE_TOOL_MAP.get(action)
    if tool is None:
        return None

    payload: dict = {}
    if action in ("tap", "long_press_on", "focus_and_input_text", "focus_and_clear_text", "erase_one_char"):
        # 只要**完整** id（含 "/"）：bridge 的重定位是精确匹配，尾段匹配不上。
        if not selector or "/" not in selector:
            return None
        payload["target"] = {"resource_id": selector}

    if action == "swipe_percentages":
        payload.update(
            {
                "use_percentages": True,
                "start_x_percent": args.get("start_x_percent", 50),
                "start_y_percent": args.get("start_y_percent", 70),
                "end_x_percent": args.get("end_x_percent", 50),
                "end_y_percent": args.get("end_y_percent", 35),
            }
        )
    elif action == "swipe_coordinates":
        payload.update(
            {
                "use_percentages": False,
                "start_x": args.get("start_x", 0),
                "start_y": args.get("start_y", 0),
                "end_x": args.get("end_x", 0),
                "end_y": args.get("end_y", 0),
            }
        )
    elif action == "focus_and_input_text":
        payload["text"] = args.get("text", "")
    elif action == "wait_for_delay":
        payload["seconds"] = round(float(args.get("time_in_ms", 1000)) / 1000.0, 3)
    elif action == "press_key":
        payload["key"] = args.get("key") or args.get("keys") or "back"
    elif action in ("launch_app", "stop_app"):
        # 包名在脚本级（app_package）给出，这里先占位。
        package = args.get("package") or args.get("app_name") or ""
        if package:
            payload["package"] = package
        # 没有包名时**不要**塞空串：bridge 的 stop_app 语义是「不给包名就停当前前台」，
        # 给空串会变成「停一个叫 '' 的 app」。

    return {"tool": tool, "args": payload}


def parse_task_package(execution: dict) -> str | None:
    """取**任务目标 app** 的包名。

    为什么不用 trace 里的「Focused app info」：那个是**某一瞬间**的前台 app，
    录制时经常是 runner 自己的 app（通知栏/悬浮窗）—— 真机实测 816 的第 1 步
    录到的就是 ``com.mobilemanager.runnerapk``。用它当前置条件会把宏
    一上来就判死。
    """
    m = TASK_PACKAGE_RE.search(execution.get("assembledPrompt") or "")
    return m.group(1) if m else None


def parse_focused_package(prompt: dict) -> str | None:
    """从 prompt 里取「当时的前台 app 包名」—— 宏的前置条件之一。"""
    for m in prompt.get("messages") or []:
        if m.get("role") != "user":
            continue
        for b in m.get("blocks") or []:
            if b.get("type") != "text":
                continue
            mm = FOCUS_PACKAGE_RE.search(b.get("text") or "")
            if mm:
                raw = mm.group(1)
                return _DISPLAY_NAME_TO_PACKAGE.get(raw, raw)
    return None


def compile_macro(exec_id: int, manager: str) -> dict:
    base = f"{manager}/api/executions/{exec_id}"
    items = get(base + "/prompts?size=500")["items"]
    calls = sorted([i for i in items if i.get("agent") == "executor"], key=lambda x: x["callIndex"])

    rounds = []
    for it in calls:
        d = get("%s/prompts/%d" % (base, it["callIndex"]))
        p = json.loads(d["promptJson"])
        r = json.loads(d["responseJson"]) if d.get("responseJson") else {}
        tcs = r.get("tool_calls") or []
        actions = ((tcs[0].get("args") or {}).get("actions") or []) if tcs else []
        # 本轮 prompt 里的最后一条 tool 消息 = **上一轮**动作的结果
        last_tool = None
        for m in p["messages"]:
            if m.get("role") == "tool":
                last_tool = text_of(m)
        rounds.append(
            {
                "call_index": it["callIndex"],
                "actions": actions,
                "prev_result": last_tool,
                "rows": parse_element_table(p),
                "package": parse_focused_package(p),
            }
        )

    steps = []
    for idx, rnd in enumerate(rounds):
        if not rnd["actions"]:
            continue
        action = dict(rnd["actions"][0])
        # 结果在**下一轮**的 prompt 里（下一轮的 prompt 带着上一轮的结果）
        result = rounds[idx + 1]["prev_result"] if idx + 1 < len(rounds) else None
        if result is None and idx > 0:
            result = rnd["prev_result"]
        result = result or ""

        # ⚠️ 元素表必须取**决策那一轮**的（``rnd["rows"]``）—— 下一轮的 prompt 里
        # 已经是**另一个屏**的元素表了，index 完全对不上。
        selector, how = resolve_stable_selector(result, rnd["rows"])
        # ---- 降级链的第 2 级：**位置** ----
        # 真机实测（2026-10-22，816 → 4 小时后回放）：快手把同一个「搜索」入口的
        # resource_id 从 ``id/right_tv`` 改成了 ``id/thanos_home_top_search``，
        # 而它的**位置几乎没变**（中心点差 15px）。
        # ⇒ 对这种高频改版的 app，**位置比 id 稳**（和仓库里已有的结论一致：
        #   「位置比内容稳」—— 这里是「位置比 id 稳」）。
        target = (action.get("target") or {}) if isinstance(action.get("target"), dict) else {}
        row = None
        idx = target.get("index")
        if idx is not None:
            try:
                row = rnd["rows"].get(int(idx))
            except (TypeError, ValueError):
                row = None
        if row is None:
            # 坐标兜底：用结果文字里的坐标在元素表里找同一行
            m = COORD_RE.search(result)
            if m:
                x, y = int(m.group(1)), int(m.group(2))
                for r in rnd["rows"].values():
                    cx, cy = r["x"] + r["w"] // 2, r["y"] + r["h"] // 2
                    if abs(cx - x) <= 8 and abs(cy - y) <= 8:
                        row = r
                        break
        step = {
            "action": action.get("action"),
            "args": {k: v for k, v in action.items() if k != "action"},
            "stable_selector": selector,
            "selector_source": how,
            "recorded_result": result[:200],
            # ---- 降级链第 3 级：上下文指纹 ----
            # 工具结果文字里带着 ``ctx:``（由 tools/mobile/tap.py 写入），
            # 它记的是「目标 + 前后邻居」—— app 改版后 id 和坐标都会变，
            # 邻居不会。
            "context": parse_context(result),
            # ---- 降级链的第 2 级：录制时它在哪 ----
            "recorded_bounds": (
                {"x": row["x"], "y": row["y"], "w": row["w"], "h": row["h"]} if row else None
            ),
            "recorded_text": row["text"] if row else None,
            "recorded_class": row["cls"] if row else None,
            # ---- 前置条件：这一步是在**哪一屏**上做的 ----
            # 录制的屏和回放的屏不一定一样（app 状态、tab、页面都可能是另一处）。
            # 没有前置条件，宏就会在错的屏上照样点 —— 而 adb tap 不报错，
            # 「点到屏幕外」也会被当成成功（真机实测：tap(1486,2118) 在 1220 宽的屏上）。
            "expect_package": rnd["package"],
            "expect_ids": sorted({r["id_tail"] for r in rnd["rows"].values() if r["id_tail"]}),
            "expect_row_count": len(rnd["rows"]),
        }
        steps.append(step)

    execution = get(base)
    return {
        "name": f"exec{exec_id}_replay",
        "description": f"从执行 {exec_id} 的 trace 编译出来的动作序列",
        "source_exec_id": exec_id,
        "app_package": parse_task_package(execution),
        "device_serial": execution.get("serial"),
        "steps": steps,
        "bridge_steps": _to_bridge_steps(steps, parse_task_package(execution)),
    }


def _to_bridge_steps(steps: list[dict], app_package: str | None) -> list[dict]:
    """把编译出来的动作翻成 **bridge 工具调用**（脚本真正要跑的东西）。"""
    out = []
    for step in steps:
        action = step.get("action")
        args = step.get("args") or {}
        bridge_step = to_bridge_step(str(action), args, step.get("stable_selector"))
        if bridge_step is None:
            continue
        if action == "launch_app" and app_package:
            bridge_step["args"]["package"] = app_package
        # 附上录制时的信息（给人看 / 给引擎做降级兜底）。
        # ⚠️ 放在 ``fallback`` 里，**不能**塞进 ``args`` —— args 会原样发给工具，
        # 多余的键会被参数校验当成「未知参数」拒掉。
        recorded_bounds = step.get("recorded_bounds")
        fallback: dict[str, Any] = {
            "selector": step.get("stable_selector"),
            "bounds": recorded_bounds,
            "text": step.get("recorded_text"),
        }
        if step.get("context"):
            fallback["context"] = step["context"]
        bridge_step["fallback"] = fallback
        out.append(bridge_step)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec-id", type=int, required=True)
    ap.add_argument("--manager", default="http://127.0.0.1:9107")
    ap.add_argument("--out", default=None)
    ap.add_argument("--print-steps", action="store_true")
    args = ap.parse_args()

    macro = compile_macro(args.exec_id, args.manager)
    steps = macro["steps"]
    with_sel = sum(1 for s in steps if s["stable_selector"])
    print("执行 %d —— %d 步，其中 %d 步有稳定选择器" % (args.exec_id, len(steps), with_sel))
    for i, s in enumerate(steps, 1):
        print(
            "  %2d. %-22s sel=%-46s (%s)  args=%s"
            % (i, s["action"], (s["stable_selector"] or "-")[:46], s["selector_source"], s["args"])
        )

    out = args.out or f"probe_out/macro_exec{args.exec_id}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(macro, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n宏文件:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
