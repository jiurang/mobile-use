"""实测 v2：工具 schema 的真实结构与各优化变体的收益。

v1 的教训（已修正）：
  - LangChain **不会**把 Google 风格 docstring 里的 `Args:` 解析成 per-parameter description，
    整块都留在 tool 级 description 里。所以按属性名替换 description 基本不命中。
  - 真正的大头不是措辞，而是**结构噪音**：
      * 每个 nullable 字段都展开成 anyOf:[{type:X},{type:"null"}]
      * 每个字段都带 title（"X" / "Width" / "ElementBounds" …）
      * Target 在 4 个工具里完整内联，连 bounds 的 x/y/width/height 都各自带描述

用法：python v/probe_tool_schema_variants_v2.py
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
TARGET_FIELDS = ["resource_id", "resource_id_index", "text", "text_index", "bounds"]


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
    return [
        convert_to_openai_tool(t)
        for t in get_tools_from_wrappers(ctx, list(EXECUTOR_WRAPPERS_TOOLS))
    ]


# --------------------------------------------------------------------------
# 结构手术
# --------------------------------------------------------------------------
def strip_titles(node):
    """删掉所有 title 键（纯展示用，模型不需要）。"""
    if isinstance(node, dict):
        node.pop("title", None)
        for v in node.values():
            strip_titles(v)
    elif isinstance(node, list):
        for v in node:
            strip_titles(v)


def collapse_nullable(node):
    """anyOf:[{type:X},{type:"null"}] -> X（并保留 default:null 表示可省略）。"""
    if isinstance(node, dict):
        ao = node.get("anyOf")
        if isinstance(ao, list) and len(ao) == 2:
            non_null = [b for b in ao if b != {"type": "null"}]
            nulls = [b for b in ao if b == {"type": "null"}]
            if len(non_null) == 1 and nulls:
                merged = dict(node)
                merged.pop("anyOf")
                merged.update(non_null[0])
                node.clear()
                node.update(merged)
        for v in node.values():
            collapse_nullable(v)
    elif isinstance(node, list):
        for v in node:
            collapse_nullable(v)


def extract_target_props(schemas):
    """抓一份 Target 的完整 properties 定义。"""
    for s in schemas:
        props = s["function"].get("parameters", {}).get("properties", {})
        t = props.get("target")
        if t and "properties" in t:
            return copy.deepcopy(t)
    return None


def share_target(schemas, target_props):
    """把内联的 Target 换成 $ref，顶层定义一次。"""
    out = copy.deepcopy(schemas)
    for s in out:
        params = s["function"].get("parameters", {})
        props = params.get("properties", {})
        t = props.get("target")
        if t and "properties" in t:
            props["target"] = {"$ref": "#/$defs/Target"}
            params.setdefault("$defs", {})["Target"] = {
                "type": "object",
                "description": "Element locator. Fallback: bounds -> resource_id -> text.",
                "properties": copy.deepcopy(target_props),
            }
    return out


# --------------------------------------------------------------------------
# 措辞手术
# --------------------------------------------------------------------------
EN_DESC = {
    "The resource-id of the element.": "resource-id.",
    "The zero-based index if multiple elements share the same resource-id.": "Index among same resource-id.",
    "The text content of the element (e.g., a label or placeholder).": "Element text.",
    "The zero-based index if multiple elements share the same text.": "Index among same text.",
    "The x, y, width, and height of the element.": "x, y, w, h.",
    "The x coordinate of the top-left corner of the element.": "x.",
    "The y coordinate of the top-left corner of the element.": "y.",
    "The width of the element.": "w.",
    "The height of the element.": "h.",
    "A comprehensive locator for a UI element, supporting a fallback mechanism.":
        "Element locator. Fallback: bounds -> resource_id -> text.",
}

ZH_DESC = {
    "The resource-id of the element.": "元素 resource-id。",
    "The zero-based index if multiple elements share the same resource-id.": "同名 id 中的序号。",
    "The text content of the element (e.g., a label or placeholder).": "元素文本。",
    "The zero-based index if multiple elements share the same text.": "同名文本中的序号。",
    "The x, y, width, and height of the element.": "x、y、宽、高。",
    "The x coordinate of the top-left corner of the element.": "x。",
    "The y coordinate of the top-left corner of the element.": "y。",
    "The width of the element.": "宽。",
    "The height of the element.": "高。",
    "A comprehensive locator for a UI element, supporting a fallback mechanism.":
        "元素定位，回退顺序：bounds -> resource_id -> text。",
}

# tool 级 description 里的 Args: 块（v1 的教训：它没有下沉到字段，必须整块处理）
ARGS_BLOCK_RE_HINT = "Args:"


def strip_args_block(node):
    """把 tool description 里的 Args: 段落整段删掉（字段语义已在字段自身描述里）。"""
    if isinstance(node, dict):
        d = node.get("description")
        if isinstance(d, str) and ARGS_BLOCK_RE_HINT in d:
            node["description"] = d.split(ARGS_BLOCK_RE_HINT)[0].strip()
        for v in node.values():
            strip_args_block(v)
    elif isinstance(node, list):
        for v in node:
            strip_args_block(v)


def map_desc(node, mapping):
    if isinstance(node, dict):
        d = node.get("description")
        if isinstance(d, str) and d in mapping:
            node["description"] = mapping[d]
        for v in node.values():
            map_desc(v, mapping)
    elif isinstance(node, list):
        for v in node:
            map_desc(v, mapping)


def variant(schemas, *, strip_title, collapse, share, drop_args, zh):
    out = copy.deepcopy(schemas)
    target_props = extract_target_props(out) if share else None
    if strip_title:
        strip_titles(out)
    if collapse:
        collapse_nullable(out)
    if share and target_props:
        if strip_title:
            strip_titles(target_props)
        if collapse:
            collapse_nullable(target_props)
        out = share_target(out, target_props)
    if drop_args:
        strip_args_block(out)
    map_desc(out, ZH_DESC if zh else EN_DESC)
    return out


def report(label, schemas, base):
    t = ntok(json.dumps(schemas))
    d = t - base
    print(f"  {label:<40} {t:>6,}   {d:>+7,}  ({d / base * 100:>+6.1f}%)")
    return t


if __name__ == "__main__":
    base_schemas = real_schemas()
    base = ntok(json.dumps(base_schemas))

    print("=" * 78)
    print("1. 结构噪音占比（不改任何语义，只删 JSON 冗余）")
    print("=" * 78)
    print(f"  {'BASELINE':<40} {base:>6,}")
    print()
    t_title = report("仅删 title", variant(base_schemas, strip_title=True, collapse=False,
                                           share=False, drop_args=False, zh=False), base)
    t_coll = report("仅折叠 anyOf-null", variant(base_schemas, strip_title=False, collapse=True,
                                                 share=False, drop_args=False, zh=False), base)
    t_struct = report("删 title + 折叠 anyOf", variant(base_schemas, strip_title=True, collapse=True,
                                                      share=False, drop_args=False, zh=False), base)
    report("以上 + Target 共享 $ref", variant(base_schemas, strip_title=True, collapse=True,
                                              share=True, drop_args=False, zh=False), base)
    print()

    print("=" * 78)
    print("2. 措辞层面：英文精简 vs 中文化（都在已做结构优化的基础上）")
    print("=" * 78)
    kw = dict(strip_title=True, collapse=True, share=True)
    report("结构优化（保留英文原描述）", variant(base_schemas, drop_args=False, zh=False, **kw), base)
    s_en = report("结构优化 + 删 Args 块", variant(base_schemas, drop_args=True, zh=False, **kw), base)
    s_zh = report("结构优化 + 删 Args 块 + 中文化", variant(base_schemas, drop_args=True, zh=True, **kw), base)
    print()
    print(f"  → 仅「英文→中文」这一步的净收益: {s_en - s_zh:+,} token")
    print()

    print("=" * 78)
    print("3. 组合拳 vs baseline")
    print("=" * 78)
    final = variant(base_schemas, drop_args=True, zh=True, **kw)
    t = report("结构共享 + 精简中文", final, base)
    print()

    print("=" * 78)
    print("4. 落到 executor 单轮")
    print("=" * 78)
    for label, sch in [("baseline", base_schemas), ("组合拳", final)]:
        s = ntok(json.dumps(sch))
        for k in (1, 4, 8):
            total = 336 + 59 + 10 + 199 * k + s
            print(f"  {label:<10} {k} 轮: executor 总计 {total:>6,} token")
    print()

    print("=" * 78)
    print("5. 中文的 token 密度实测（同义句对照）")
    print("=" * 78)
    pairs = [
        ("Element locator. Fallback: bounds -> resource_id -> text.",
         "元素定位，回退顺序：bounds -> resource_id -> text。"),
        ("The zero-based index if multiple elements share the same resource-id.",
         "同名 resource-id 元素中的序号（从 0 开始）。"),
        ("Long press a UI element. Locator fallback order: bounds -> resource_id -> text.",
         "长按 UI 元素。定位回退顺序：bounds -> resource_id -> text。"),
        ("Duration of the long press in milliseconds. Default 1000.",
         "长按时长（毫秒）。默认 1000。"),
    ]
    print(f"  {'英文':<62} tok")
    for en, zh in pairs:
        print(f"  {en:<62} {ntok(en):>3}")
        print(f"  {'  ↳ 中文':<62} {ntok(zh):>3}")
        print()
