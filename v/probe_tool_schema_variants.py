"""实测：工具 schema 的 token 到底花在哪，以及「精简描述 / 中文化 / 结构共享」各自能省多少。

不猜。直接对真实的 bind_tools schema 做变体，逐个 tiktoken 计数。

用法：python v/probe_tool_schema_variants.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

import tiktoken
from langchain_core.utils.function_calling import convert_to_openai_tool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))

from minitap.mobile_use.context import DeviceContext, DevicePlatform, MobileUseContext  # noqa: E402
from minitap.mobile_use.config import LLMConfig  # noqa: E402
from minitap.mobile_use.tools.index import EXECUTOR_WRAPPERS_TOOLS, get_tools_from_wrappers  # noqa: E402

ENC = tiktoken.get_encoding("cl100k_base")


def ntok(t: str) -> int:
    return len(ENC.encode(t))


def real_schemas() -> list[dict]:
    ctx = MobileUseContext.model_construct(
        trace_id="probe",
        device=DeviceContext(
            host_platform="WINDOWS",
            mobile_platform=DevicePlatform.ANDROID,
            device_id="probe-device",
            device_width=1080,
            device_height=2400,
        ),
        llm_config=LLMConfig.model_construct(),
        video_recording_enabled=False,
    )
    tools = get_tools_from_wrappers(ctx, list(EXECUTOR_WRAPPERS_TOOLS))
    return [convert_to_openai_tool(t) for t in tools]


# ==========================================================================
# 0. 先看清 baseline 的 token 去向
# ==========================================================================
def show_anatomy(schemas: list[dict]):
    print("=" * 78)
    print("0. BASELINE 拆解：long_press_on 的 646 token 花在哪")
    print("=" * 78)
    lp = next(s for s in schemas if s["function"]["name"] == "long_press_on")
    fn = lp["function"]
    desc = fn.get("description", "")
    print(f"  description 总计           {ntok(desc):>5} token")
    props = fn["parameters"]["properties"]
    for pname, p in props.items():
        d = p.get("description", "")
        print(f"    - {pname:<22} {ntok(d):>5} token")
    # Target 的 5 个字段在每个工具里都重复一遍
    target_fields = ["resource_id", "resource_id_index", "text", "text_index", "bounds"]
    total_target = sum(
        ntok(props[f].get("description", ""))
        for f in target_fields
        if f in props
    )
    print(f"  → 其中 Target 五字段描述合计 {total_target} token")
    print()


# ==========================================================================
# 变体 1：精简英文描述
#     规则：删掉 Args: 里与 tool summary 重复的部分、去掉冗余修饰语
# ==========================================================================
def slim_en(schemas: list[dict]) -> list[dict]:
    SLIM = {
        "The resource-id of the element.": "resource-id.",
        "The zero-based index if multiple elements share the same resource-id.":
            "Index among elements sharing this resource-id.",
        "The text content of the element (e.g., a label or placeholder).": "Element text.",
        "The zero-based index if multiple elements share the same text.":
            "Index among elements sharing this text.",
        "The x, y, width, and height of the element.": "x, y, w, h.",
        "The thought of the agent.": "Why this call.",
        "The UI element to tap on (bounds, resource_id, or text).": "Target element.",
        "The target of the text input (if available).": "Target field.",
        "The text to type.": "Text to type.",
        "Duration of the long press in milliseconds. Choose based on interaction:\n"
        "                        - 500-800ms: Quick long press (e.g., selecting text, haptic feedback)\n"
        "                        - 1000ms (default): Standard long press (most common use case)\n"
        "                        - 1500-2000ms: Extended long press (e.g., context menus, special actions)\n"
        "                        - 2500ms+: Very long press (e.g., accessibility, advanced gestures)":
            "Press duration in ms. Default 1000.",
    }
    # tool summary 去掉与 Args 重复的段落
    SUMMARY = {
        "long_press_on": "Long press a UI element. Locator fallback order: bounds -> resource_id -> text.",
        "tap": "Tap a UI element. Locator fallback order: bounds -> resource_id -> text.",
        "focus_and_clear_text": "Focus a text field and clear it. Locator fallback: bounds -> resource_id -> text.",
    }
    out = copy.deepcopy(schemas)
    for s in out:
        fn = s["function"]
        name = fn["name"]
        if name in SUMMARY:
            fn["description"] = SUMMARY[name]
        for p in fn.get("parameters", {}).get("properties", {}).values():
            d = p.get("description")
            if d in SLIM:
                p["description"] = SLIM[d]
    return out


# ==========================================================================
# 变体 2：中文描述（忠实翻译，不删信息）
# ==========================================================================
ZH_FIELD = {
    "The resource-id of the element.": "元素的 resource-id。",
    "The zero-based index if multiple elements share the same resource-id.":
        "同名 resource-id 元素中的序号（从 0 开始）。",
    "The text content of the element (e.g., a label or placeholder).": "元素文本。",
    "The zero-based index if multiple elements share the same text.": "同名文本元素中的序号（从 0 开始）。",
    "The x, y, width, and height of the element.": "元素的 x、y、宽、高。",
    "The thought of the agent.": "调用原因。",
    "The UI element to tap on (bounds, resource_id, or text).": "目标元素。",
    "The target of the text input (if available).": "目标输入框。",
    "The text to type.": "要输入的文本。",
    "Duration of the long press in milliseconds. Choose based on interaction:\n"
    "                        - 500-800ms: Quick long press (e.g., selecting text, haptic feedback)\n"
    "                        - 1000ms (default): Standard long press (most common use case)\n"
    "                        - 1500-2000ms: Extended long press (e.g., context menus, special actions)\n"
    "                        - 2500ms+: Very long press (e.g., accessibility, advanced gestures)":
        "长按时长（毫秒）。默认 1000。",
}
ZH_SUMMARY = {
    "long_press_on": "长按 UI 元素。定位回退顺序：bounds -> resource_id -> text。",
    "tap": "点击 UI 元素。定位回退顺序：bounds -> resource_id -> text。",
    "focus_and_clear_text": "聚焦并清空输入框。定位回退顺序：bounds -> resource_id -> text。",
}


def to_zh(schemas: list[dict], *, also_slim: bool) -> list[dict]:
    out = copy.deepcopy(schemas)
    for s in out:
        fn = s["function"]
        name = fn["name"]
        if name in ZH_SUMMARY:
            fn["description"] = ZH_SUMMARY[name]
        for p in fn.get("parameters", {}).get("properties", {}).values():
            d = p.get("description")
            if d in ZH_FIELD:
                p["description"] = ZH_FIELD[d]
            elif also_slim:
                p["description"] = ZH_FIELD.get(d, d)
    return out


# ==========================================================================
# 变体 3：结构共享 —— Target 只定义一次，其余用 $ref
# ==========================================================================
TARGET_FIELDS = ["resource_id", "resource_id_index", "text", "text_index", "bounds"]


def shared_defs(schemas: list[dict]) -> list[dict]:
    """把每个工具里重复展开的 Target 换成 $ref + 顶层 $defs。"""
    out = copy.deepcopy(schemas)
    target_props = None
    for s in out:
        props = s["function"].get("parameters", {}).get("properties", {})
        if all(f in props for f in TARGET_FIELDS):
            if target_props is None:
                target_props = {f: props[f] for f in TARGET_FIELDS}
            props["target"] = {"$ref": "#/$defs/Target"}
            for f in TARGET_FIELDS:
                props.pop(f)
            s["function"]["parameters"].setdefault("$defs", {})["Target"] = {
                "title": "Target",
                "type": "object",
                "description": "Element locator.",
                "properties": copy.deepcopy(target_props),
            }
    return out


# ==========================================================================
def report(label: str, schemas: list[dict], base: int):
    total = ntok(json.dumps(schemas))
    delta = total - base
    pct = delta / base * 100
    print(f"  {label:<34} {total:>6,} token   {delta:>+7,}  ({pct:>+6.1f}%)")
    return total


if __name__ == "__main__":
    base_schemas = real_schemas()
    show_anatomy(base_schemas)

    base = ntok(json.dumps(base_schemas))
    print("=" * 78)
    print("1. 变体对比（bind_tools 每次重发的 16 个工具 schema）")
    print("=" * 78)
    print(f"  {'BASELINE（现状）':<32} {base:>6,} token")
    print()

    v_en = report("精简英文描述", slim_en(base_schemas), base)
    v_zh = report("中文化（忠实翻译，不删信息）", to_zh(base_schemas, also_slim=False), base)
    v_zh_slim = report("中文 + 精简", to_zh(base_schemas, also_slim=True), base)
    v_defs = report("结构共享（Target 用 $ref）", shared_defs(base_schemas), base)
    v_all = report(
        "结构共享 + 精简中文",
        shared_defs(to_zh(base_schemas, also_slim=True)),
        base,
    )

    print()
    print("=" * 78)
    print("2. 单轮 executor 的实际收益")
    print("=" * 78)
    print("  executor 每轮 = system 336 + decisions 59 + thoughts 10 + messages + schema")
    for label, sch in [
        ("baseline", base_schemas),
        ("精简中文 + 结构共享", shared_defs(to_zh(base_schemas, also_slim=True))),
    ]:
        s = ntok(json.dumps(sch))
        for k in (1, 4, 8):
            hist = 199 * k / 1
            total = 336 + 59 + 10 + hist + s
            print(f"  {label:<20} {k} 轮: executor 总计 {total:>6,} token")
    print()
