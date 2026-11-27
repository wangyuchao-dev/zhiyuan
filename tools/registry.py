"""工具注册表：把「函数」变成「模型能调的工具」。

三件事：
  1. 生成 function calling 描述（name / description / parameters）；
  2. 调用前校验参数（缺参、类型不对、未知参数），错误按 core.errors 分类；
  3. 统一执行入口，任何异常都转成 ToolError，不让堆栈冒到模型面前。

为什么不用装饰器自动扫描：
    工具的执行位置（进程内 / MCP）和鉴权方式不同，显式注册比魔法更清楚；
    而且注册表就是「模型能看到哪些能力」的唯一真相，越白盒越好。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

try:
    from core.errors import ErrorKind, ToolError, classify, to_model_message
except ImportError:  # pragma: no cover
    from ..core.errors import ErrorKind, ToolError, classify, to_model_message  # type: ignore

_JSON_TYPE = {"string": str, "integer": int, "number": (int, float),
              "boolean": bool, "array": list, "object": dict}


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]                  # JSON Schema
    handler: Optional[Callable[..., Any]] = None
    requires_perm: Sequence[str] = field(default_factory=tuple)
    timeout_s: float = 10.0
    read_only: bool = True

    def to_function_spec(self) -> Dict[str, Any]:
        """OpenAI / 通义 / 智谱 通用的 function calling 结构。"""
        return {"type": "function",
                "function": {"name": self.name,
                             "description": self.description,
                             "parameters": self.parameters}}


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: Optional[ToolError] = None
    elapsed_ms: float = 0.0

    def model_view(self) -> str:
        """给模型看的执行结果（错误也翻译成人话）。"""
        if self.ok:
            return str(self.data)
        return to_model_message(self.error, attempts=1)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    # ---- 注册 ----
    def register(self, spec: ToolSpec) -> None:
        if not spec.name or not spec.description:
            raise ValueError("工具的 name / description 不能为空（模型靠它判断该不该调）")
        if "type" not in spec.parameters:
            raise ValueError(f"{spec.name}: parameters 必须是 JSON Schema，至少含 type")
        self._tools[spec.name] = spec

    def register_simple(self, name: str, description: str,
                        params: Dict[str, str], handler: Callable[..., Any],
                        required: Sequence[str] = (), **kwargs: Any) -> None:
        """用 {参数名: 类型} 的简写注册，省掉手写 schema。"""
        props = {k: {"type": v} for k, v in params.items()}
        schema = {"type": "object", "properties": props,
                  "required": list(required) or list(params)}
        self.register(ToolSpec(name, description, schema, handler, **kwargs))

    # ---- 暴露给模型 ----
    def specs(self, allow: Sequence[str] | None = None) -> List[Dict[str, Any]]:
        """按白名单裁剪 —— 不同角色能看到的工具不同，这是最小权限的第一层。"""
        names = allow if allow is not None else self._tools.keys()
        return [self._tools[n].to_function_spec() for n in names if n in self._tools]

    def names(self) -> List[str]:
        return list(self._tools.keys())

    # ---- 校验 ----
    def validate(self, name: str, args: Dict[str, Any]) -> Optional[ToolError]:
        spec = self._tools.get(name)
        if spec is None:
            return ToolError(ErrorKind.PARAM, "UNKNOWN_TOOL",
                             f"没有名为 {name} 的工具",
                             hint="可用工具：" + "、".join(self._tools))
        schema = spec.parameters
        props = schema.get("properties", {})
        required = schema.get("required", [])

        missing = [k for k in required if k not in args or args[k] in (None, "")]
        if missing:
            return ToolError(ErrorKind.PARAM, "MISSING_ARG",
                             f"缺少参数 {missing}",
                             hint="必填：" + "、".join(required))
        unknown = [k for k in args if k not in props]
        if unknown:
            return ToolError(ErrorKind.PARAM, "UNKNOWN_ARG",
                             f"不认识的参数 {unknown}",
                             hint="可用参数：" + "、".join(props))
        for key, value in args.items():
            want = props[key].get("type")
            py = _JSON_TYPE.get(want or "")
            if py and not isinstance(value, py):
                return ToolError(ErrorKind.PARAM, "TYPE_MISMATCH",
                                 f"参数 {key} 应为 {want}，实际是 {type(value).__name__}",
                                 hint=f"{key} 传 {want}")
        return None

    # ---- 执行 ----
    def call(self, name: str, args: Dict[str, Any],
             granted_perms: Sequence[str] = ()) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(False, error=ToolError(ErrorKind.PARAM, "UNKNOWN_TOOL", name))
        if spec.requires_perm and not set(spec.requires_perm).issubset(set(granted_perms)):
            # 权限不足不重试、不提示「换个参数」，否则等于教模型越权
            return ToolResult(False, error=ToolError(
                ErrorKind.PERMISSION, "NO_PERM",
                f"缺少权限 {list(spec.requires_perm)}"))
        bad = self.validate(name, args)
        if bad:
            return ToolResult(False, error=bad)

        t0 = time.perf_counter()
        try:
            data = spec.handler(**args) if spec.handler else None
        except Exception as exc:  # 统一兜住，别让堆栈进 prompt
            return ToolResult(False, error=classify(exc),
                              elapsed_ms=(time.perf_counter() - t0) * 1000)
        return ToolResult(True, data, elapsed_ms=(time.perf_counter() - t0) * 1000)
