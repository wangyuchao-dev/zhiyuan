"""工具/接口错误的分类与重试策略。

为什么单独一个模块：
    「调用失败」不是一个东西。超时和越权都叫失败，但处理方式相反 ——
    超时要退避重试，越权要立刻拒绝（重试只会让系统看起来像在暴力尝试）。
    把分类写在 if/else 里，三处代码就会长出三种口径；所以集中在这里。

给模型的错误信息也要经过这里翻译：
    原始错误是 `HTTPConnectionPool(host=..., port=8000): Read timed out`，
    模型看不懂，只会重复同样的调用。翻译成「检索服务超时，已重试 2 次仍失败，
    请直接回答『暂时无法查询』」它才知道该干什么。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

# 退避序列（秒）：宁可少量多次，也不要一次等 30 秒把用户晾着
BACKOFF = (0.5, 1.0, 2.0)


class ErrorKind(enum.StrEnum):
    """五类错误，策略两两不同。"""

    TRANSIENT = "transient"  # 超时 / 限流 / 5xx —— 可重试
    PARAM = "param"  # 参数非法 —— 可重试一次，但要带修正提示
    PERMISSION = "permission"  # 无权限 —— 不重试，直接拒
    EMPTY = "empty"  # 查到了但为空 —— 不重试，走降级话术
    LOGIC = "logic"  # 业务/解析错误 —— 不重试，需人工看

    @property
    def retryable(self) -> bool:
        return self in (ErrorKind.TRANSIENT, ErrorKind.PARAM)


@dataclass
class ToolError(Exception):
    """统一错误对象：分类 + 原始信息 + 给模型看的翻译。"""

    kind: ErrorKind
    code: str
    detail: str = ""
    retryable: bool = field(init=False)
    hint: str = ""  # 给模型的修正提示（参数错误时填合法值）

    def __post_init__(self) -> None:
        self.retryable = self.kind.retryable

    def __str__(self) -> str:  # 日志里好看一点
        return f"[{self.kind.value}:{self.code}] {self.detail}"


def classify(
    status: int | None = None, exc: BaseException | None = None, empty: bool = False
) -> ToolError:
    """把 HTTP 状态码或异常翻译成 ToolError。

    顺序很重要：先看「空结果」（它不是错误，是一种正常返回值），
    再看异常类型，最后才看状态码 —— 因为有些 SDK 会把 4xx 抛成异常。
    """
    if isinstance(status, BaseException):  # 兼容 classify(exc) 的写法
        exc, status = status, None
    if empty:
        return ToolError(ErrorKind.EMPTY, "EMPTY_RESULT", "检索结果为空")
    if status is not None:
        if status in (401, 403):
            return ToolError(ErrorKind.PERMISSION, f"HTTP_{status}", "无权限访问该资源")
        if status == 429:
            return ToolError(ErrorKind.TRANSIENT, "RATE_LIMITED", "触发限流")
        if status >= 500:
            return ToolError(ErrorKind.TRANSIENT, f"HTTP_{status}", "服务端错误")
        if status >= 400:
            return ToolError(ErrorKind.PARAM, f"HTTP_{status}", "请求参数被拒绝")
    if exc is not None:
        name = type(exc).__name__.lower()
        if "timeout" in name or "timedout" in name:
            return ToolError(ErrorKind.TRANSIENT, "TIMEOUT", str(exc)[:120])
        if "connection" in name:
            return ToolError(ErrorKind.TRANSIENT, "CONN_ERROR", str(exc)[:120])
        if isinstance(exc, (KeyError, ValueError, TypeError)):
            return ToolError(ErrorKind.PARAM, "BAD_ARG", str(exc)[:120])
        return ToolError(ErrorKind.LOGIC, "UNKNOWN", f"{type(exc).__name__}: {exc}"[:160])
    return ToolError(ErrorKind.LOGIC, "UNSPECIFIED", "未识别的失败")


def next_delay(err: ToolError, attempt: int) -> float | None:
    """第 attempt 次失败后还要不要等。返回 None = 放弃。

    attempt 从 1 开始（第一次失败是 attempt=1）。
    参数错误只给一次机会：改参数重试一次，再错就是模型没听懂，继续重试无意义。
    """
    if not err.retryable:
        return None
    if err.kind is ErrorKind.PARAM and attempt >= 2:
        return None
    if attempt > len(BACKOFF):
        return None
    return BACKOFF[attempt - 1]


def to_model_message(err: ToolError, attempts: int = 1) -> str:
    """翻译成模型能照着行动的话。

    三条要求：① 说清发生了什么（不贴堆栈）；② 说清还能不能继续；
             ③ 给出下一步动作（改参数 / 换个说法 / 直接告诉用户）。
    """
    if err.kind is ErrorKind.TRANSIENT:
        return (
            f"检索服务暂时不可用（{err.detail}），已尝试 {attempts} 次。"
            "请不要重复调用同一工具，直接告诉用户「暂时无法查询，请稍后再试」。"
        )
    if err.kind is ErrorKind.PARAM:
        fixed = f"，请改用：{err.hint}" if err.hint else ""
        return (
            f"工具参数有误（{err.detail}）{fixed}。"
            "请修正参数后重新调用一次；若仍失败，说明该信息不在知识库中。"
        )
    if err.kind is ErrorKind.PERMISSION:
        return (
            "当前账号无权访问该数据。不要尝试绕过或换参数，直接告知用户「该内容超出你的权限范围」。"
        )
    if err.kind is ErrorKind.EMPTY:
        return (
            "知识库中没有检索到相关内容。请直接回答「材料中没有提及」，"
            "不要用常识补充，也不要编造出处。"
        )
    return "工具调用出现未预期错误，已记录日志。请告知用户稍后重试，不要重复调用。"


def as_dict(err: ToolError) -> dict[str, Any]:
    """给评测/日志用的结构化表示（eval 的 trace_ok 认这个）。"""
    return {
        "status": "error",
        "kind": err.kind.value,
        "code": err.code,
        "detail": err.detail,
        "retryable": err.retryable,
    }
