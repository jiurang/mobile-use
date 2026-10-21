"""工具 schema 的结构手术：把纯展示性的字段和噪音从 JSON Schema 里剥掉。

为什么做
--------
`bind_tools` 每次全量下发工具 schema，executor 的 16 个工具合计 **3,560 token**。
这些 token 每一轮都发，而里面有一部分**不承载任何语义**：

- `title` —— pydantic 给每个字段自动生成的展示名（`"Index"` / `"Width"` /
  `"ElementBounds"`），模型按 `description` 判断，不看 title。
- `anyOf: [X, null]` —— pydantic 对 `int | None` 的展开，占 14.0%。
- tool 级 description 里的 `Args:` 段落 —— 字段语义已经在各字段的 description 里，
  重复一遍占 7.5%。

实测（`doc/token-context-analysis.md` §5.2）只做这三件事、保留全部语义描述：
**−716 token / 次（−20.1%）**。

**⚠️ 风险与边界**
- `anyOf` 折叠会丢掉「显式 nullable」这个信息。多数 provider 依据
  `default: null` + 非 required 判断可省略，但**上线前必须对目标 provider 做一次
  真实 tool-call 验证**（模型传 null 会不会被 400 拒）。
- 中文化不做：实测中文 token 密度更高，是**负收益**（同文档 §五·二）。
- 描述简化不做：只砍重复的 `Args:` 段落，保留全部语义描述。

**只做结构手术，不动语义。** 这是这个模块的全部主张。
"""

from __future__ import annotations

import copy
from typing import Any

# 这些 key 在 JSON Schema 里只是展示信息，模型按 description 判断语义。
_DROP_KEYS = ("title",)


def _is_nullable_anyof(node: dict[str, Any]) -> bool:
    """判断是不是 pydantic 对 `X | None` 的展开：``{"anyOf": [X, {"type": "null"}]}``。"""
    any_of = node.get("anyOf")
    if not isinstance(any_of, list) or len(any_of) != 2:
        return False
    variants = [v for v in any_of if isinstance(v, dict)]
    if len(variants) != 2:
        return False
    return any(v.get("type") == "null" for v in variants)


def _surgery_node(node: Any) -> Any:
    """递归处理一个 schema 节点。

    做两件事：
    1. 删掉 ``title``（纯展示）。
    2. ``anyOf: [X, null]`` → 直接用 X，并保留 ``default: null``。
       删掉 null 分支，**不删 default** —— default 是「可以省略」的信号，
       多数 provider 靠它判断可省略。
    """
    if isinstance(node, list):
        return [_surgery_node(item) for item in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DROP_KEYS:
            continue
        if key == "anyOf" and isinstance(value, list) and _is_nullable_anyof(node):
            # 折叠：取非 null 的那个变体，default 保持不变。
            variants = [v for v in value if isinstance(v, dict)]
            non_null = next((v for v in variants if v.get("type") != "null"), None)
            if non_null is not None:
                out.update(_surgery_node(non_null))
                continue
        out[key] = _surgery_node(value)
    return out


def surgery_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """对一个工具的 JSON Schema 做结构手术（返回新对象，不改原对象）。"""
    return _surgery_node(copy.deepcopy(schema))


def surgery_tool_schemas(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """对 OpenAI 风格的 ``tools`` 数组做结构手术。

    只认 ``{"type": "function", "function": {"parameters": {...}}}`` 这个形状；
    认不出来的条目**原样返回**（不做猜测）。
    """
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            out.append(tool)
            continue
        function = tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("parameters"), dict):
            out.append(tool)
            continue
        patched = copy.deepcopy(tool)
        patched["function"]["parameters"] = surgery_tool_schema(function["parameters"])
        out.append(patched)
    return out


def apply_tool_schema_surgery(payload: dict[str, Any]) -> dict[str, Any]:
    """就地改一个请求 body 里的 ``tools``。

    用在 ``_get_request_payload`` 之后：那是 langchain 把所有 ``bind_tools``
    汇成请求体的地方，**一处覆盖全部 provider 的 OpenAI 兼容端点**，
    且对工具实现零侵入。
    """
    tools = payload.get("tools")
    if isinstance(tools, list) and tools:
        payload["tools"] = surgery_tool_schemas(tools)
    return payload
