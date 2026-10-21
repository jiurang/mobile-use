"""工具 schema 结构手术的行为锁定。

手术只做两件事（都不动语义描述）：
1. 删 ``title`` —— pydantic 自动生成的展示名
2. ``anyOf: [X, null]`` → ``X``，保留 ``default: null``

不做的：描述简化、中文化（实测中文化是负收益）。
"""

from __future__ import annotations

import json

from langchain_core.utils.function_calling import convert_to_openai_tool

from minitap.mobile_use.tools.index import EXECUTOR_WRAPPERS_TOOLS, get_tools_from_wrappers
from minitap.mobile_use.utils.tool_schema_surgery import (
    apply_tool_schema_surgery,
    surgery_tool_schema,
    surgery_tool_schemas,
)


def test_drops_title_keys():
    schema = {
        "type": "object",
        "title": "Target",
        "properties": {"index": {"type": "integer", "title": "Index"}},
    }
    out = surgery_tool_schema(schema)
    assert "title" not in out
    assert "title" not in out["properties"]["index"]


def test_collapses_nullable_anyof_but_keeps_default():
    schema = {
        "type": "object",
        "properties": {
            "index": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "description": "the index",
            }
        },
    }
    prop = surgery_tool_schema(schema)["properties"]["index"]
    assert prop["type"] == "integer"
    assert "anyOf" not in prop
    # default 是「可省略」的信号，必须留着
    assert prop["default"] is None
    assert prop["description"] == "the index"


def test_keeps_non_nullable_anyof():
    schema = {
        "properties": {
            "x": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        }
    }
    prop = surgery_tool_schema(schema)["properties"]["x"]
    assert prop["anyOf"] == [{"type": "string"}, {"type": "integer"}]


def test_recurses_into_nested_structures():
    schema = {
        "properties": {
            "bounds": {
                "title": "ElementBounds",
                "properties": {
                    "x": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None}
                },
            },
            "items": {"type": "array", "items": {"title": "T", "type": "string"}},
        },
    }
    out = surgery_tool_schema(schema)
    assert "title" not in out["properties"]["bounds"]
    assert out["properties"]["bounds"]["properties"]["x"]["type"] == "integer"
    assert "title" not in out["properties"]["items"]["items"]


def test_unknown_tool_shapes_pass_through():
    tools = [
        {"type": "function", "function": {"name": "x"}},  # 没有 parameters
        {"type": "custom", "custom": {"a": 1}},
    ]
    assert surgery_tool_schemas(tools) == tools


def test_apply_to_payload_in_place():
    payload = {
        "model": "m",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "t",
                    "parameters": {
                        "type": "object",
                        "title": "T",
                        "properties": {"a": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
                    },
                },
            }
        ],
    }
    out = apply_tool_schema_surgery(payload)
    params = out["tools"][0]["function"]["parameters"]
    assert "title" not in params
    assert params["properties"]["a"]["type"] == "string"


def test_apply_without_tools_is_noop():
    payload = {"model": "m"}
    assert apply_tool_schema_surgery(payload) == {"model": "m"}


def test_real_executor_tools_get_slimmer():
    """真实工具集：手术必须真的变小，且不能把语义描述删掉。"""
    tools = get_tools_from_wrappers(None, EXECUTOR_WRAPPERS_TOOLS)  # type: ignore[arg-type]
    openai_tools = [convert_to_openai_tool(t) for t in tools]
    before = json.dumps(openai_tools, ensure_ascii=False, sort_keys=True)
    after = json.dumps(surgery_tool_schemas(openai_tools), ensure_ascii=False, sort_keys=True)
    assert len(after) < len(before), "手术没有让 schema 变小"
    # 语义描述一个都不能少
    for b, a in zip(openai_tools, surgery_tool_schemas(openai_tools)):
        assert b["function"]["description"] == a["function"]["description"]
        before_desc = json.dumps(b["function"]["parameters"], ensure_ascii=False)
        after_desc = json.dumps(a["function"]["parameters"], ensure_ascii=False)
        for desc_key in ("description",):
            assert before_desc.count(f'"{desc_key}"') == after_desc.count(f'"{desc_key}"'), (
                f"{b['function']['name']} 的 {desc_key} 数量变了"
            )
