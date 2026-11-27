"""MCP 与 function calling 的互转。

关系说清楚（面试常问）：
    · function calling 是**模型侧能力**：模型把「调什么、参数是什么」用 JSON 说出来；
    · MCP 是**连接侧协议**：工具在哪、怎么连、怎么描述、怎么鉴权。
    · 两者不冲突：MCP 的工具列表最终也要转成 function spec 交给模型，
      模型产出 tool_call 后，由 MCP client 去执行、再把结果回填。

本模块只做「描述层」的互转，不实现传输（stdio / streamable-http 由 SDK 负责）。
这样换传输方式不用改这里，换模型也不用改 MCP server。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

try:
    from tools.registry import ToolSpec
except ImportError:  # pragma: no cover
    from .registry import ToolSpec  # type: ignore


def from_mcp_tools(mcp_tools: Iterable[dict[str, Any]]) -> list[ToolSpec]:
    """MCP 的 tools/list 结果 → 本地 ToolSpec 列表。

    MCP 的字段是 name / description / inputSchema，与 function calling 的
    name / description / parameters 一一对应，所以只是改名 + 兜底。
    """
    specs: list[ToolSpec] = []
    for t in mcp_tools:
        name = (t.get("name") or "").strip()
        if not name:
            continue
        schema = t.get("inputSchema") or t.get("parameters") or {"type": "object", "properties": {}}
        if "type" not in schema:
            schema = {"type": "object", "properties": schema}
        specs.append(
            ToolSpec(
                name=name,
                description=(t.get("description") or name)[:200],
                parameters=schema,
                handler=None,  # 远程工具的执行交给 MCP client，本地不持有实现
                read_only=bool(t.get("annotations", {}).get("readOnlyHint", True)),
            )
        )
    return specs


def to_mcp_tools(specs: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    """本地 ToolSpec → MCP 描述（用于把自家工具挂到 MCP server 上给别人用）。"""
    return [
        {"name": s.name, "description": s.description, "inputSchema": s.parameters} for s in specs
    ]


def filter_tools(
    specs: Sequence[ToolSpec],
    allow_prefix: Sequence[str] = (),
    deny_prefix: Sequence[str] = (),
    max_tools: int = 12,
) -> list[ToolSpec]:
    """按前缀筛选并截断数量。

    为什么必须截断：工具描述会占用上下文，十几个工具就吃掉上千 token，
    而且选择越多模型越容易选错。宁可少给、给准的。
    """
    out = []
    for s in specs:
        if deny_prefix and any(s.name.startswith(p) for p in deny_prefix):
            continue
        if allow_prefix and not any(s.name.startswith(p) for p in allow_prefix):
            continue
        out.append(s)
    return out[:max_tools]


def same_toolset(a: Sequence[ToolSpec], b: Sequence[ToolSpec]) -> bool:
    """判断两个工具集是否等价（用于「MCP 与本地工具是否重复暴露」的自检）。"""
    return {s.name for s in a} == {s.name for s in b}
