"""显式的执行计划：workflow 做主骨架，只在局部留一点「重来时换个参数」的余地。

设计立场（也是选型答案）：
    · 主干步骤是**代码写死**的：判意图 → 取记忆 → 检索 → 精排 → 生成。
      因为这条链路每天都一样，交给模型规划只会更贵、更慢、更难测。
    · 只有在**单步内部**才允许「失败 → 换个参数再试一次」，次数有上限，
      这就是 ReAct 的合理用法：局部、有界、可观测。
    · 每次尝试都进 trace：没有 trace 的重试等于赌博。

    Plan 对象本身可打印、可断言 —— 测试里可以直接验证「闲聊不该走检索」。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

try:
    from core.errors import ErrorKind, ToolError, classify, next_delay
except ImportError:  # pragma: no cover
    from ..core.errors import ErrorKind, ToolError, classify, next_delay  # type: ignore


@dataclass
class Step:
    name: str
    kind: str = "workflow"          # workflow（确定性） / agent（模型决定）
    required: bool = True           # 失败是否中断整条链路
    timeout_ms: int = 8000
    max_attempts: int = 1           # 1 = 不重试

    def allow_retry(self) -> bool:
        return self.max_attempts > 1


@dataclass
class Plan:
    route: str                      # rag / chitchat / complaint / error
    steps: List[Step] = field(default_factory=list)
    note: str = ""

    def step_names(self) -> List[str]:
        return [s.name for s in self.steps]

    def __str__(self) -> str:
        return f"Plan({self.route}: {' → '.join(self.step_names())})"


@dataclass
class StepResult:
    name: str
    ok: bool
    value: Any = None
    error: Optional[ToolError] = None
    attempts: int = 1
    elapsed_ms: float = 0.0
    degraded: bool = False          # 可选步骤失败但被跳过


@dataclass
class ExecResult:
    ok: bool
    outputs: Dict[str, Any] = field(default_factory=dict)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    failed_step: str = ""
    error: Optional[ToolError] = None


def build_plan(intent: str, need_rewrite: bool, has_documents: bool) -> Plan:
    """按意图决定链路。**不调模型**，纯规则。

    · chitchat / complaint：不检索，直接生成（省一次检索 + 省 token）；
    · rag：检索是主干，改写是可选项 —— 改写失败不该让整个问题答不出来；
    · 没绑定文档：直接给「先上传文档」，连检索都不发起。
    """
    if intent in ("chitchat", "complaint"):
        return Plan(intent, [Step("compose", kind="agent", max_attempts=2)],
                    note="闲聊/投诉不检索，避免用制度条款回应情绪")

    if not has_documents:
        return Plan("error", [Step("compose_no_doc", kind="agent", required=False)],
                    note="没有可用文档，直接提示上传")

    steps: List[Step] = []
    if need_rewrite:
        # 改写是可选的：失败就退回原问题继续检索（degraded=True）
        steps.append(Step("rewrite", required=False, max_attempts=1))
    steps += [
        Step("retrieve", required=True, max_attempts=3, timeout_ms=3000),
        Step("rerank", required=False, max_attempts=1, timeout_ms=2000),
        Step("compose", kind="agent", required=True, max_attempts=2, timeout_ms=20000),
    ]
    return Plan("rag", steps, note="检索是主干；改写与精排失败可降级")


def execute(plan: Plan, handlers: Dict[str, Callable[[Dict[str, Any]], Any]],
            ctx: Optional[Dict[str, Any]] = None,
            sleep: Callable[[float], None] = time.sleep) -> ExecResult:
    """按计划执行，单步内有界重试。

    约定：handler 签名 `(ctx) -> value`，抛异常表示失败；
          想主动返回「空结果」这种非错误状态，返回空列表即可（由调用方判空）。
    """
    ctx = dict(ctx or {})
    result = ExecResult(ok=True)
    t_all = time.perf_counter()

    for step in plan.steps:
        handler = handlers.get(step.name)
        if handler is None:
            if step.required:
                result.ok = False
                result.failed_step = step.name
                result.error = ToolError(ErrorKind.LOGIC, "NO_HANDLER", step.name)
                break
            result.trace.append({"step": step.name, "status": "skipped", "reason": "no_handler"})
            continue

        attempts, err = 0, None
        t0 = time.perf_counter()
        while attempts < step.max_attempts:
            attempts += 1
            try:
                value = handler({**ctx, "attempt": attempts, "step": step.name})
                ctx[step.name] = value
                result.outputs[step.name] = value
                result.trace.append({"step": step.name, "attempt": attempts,
                                     "status": "ok",
                                     "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)})
                err = None
                break
            except Exception as exc:
                err = exc if isinstance(exc, ToolError) else classify(exc)
                delay = next_delay(err, attempts)
                result.trace.append({"step": step.name, "attempt": attempts,
                                     "status": "error", "kind": err.kind.value,
                                     "code": err.code,
                                     "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)})
                if delay is None:
                    break
                sleep(delay)

        if err is not None:
            if step.required:
                result.ok = False
                result.failed_step = step.name
                result.error = err
                break
            # 可选步骤：跳过并标记降级（改写失败 → 用原问题检索）
            result.trace.append({"step": step.name, "status": "degraded",
                                 "code": err.code})
            result.outputs.setdefault(step.name + "_degraded", True)

    result.trace.append({"step": "_total", "status": "ok" if result.ok else "error",
                         "elapsed_ms": round((time.perf_counter() - t_all) * 1000, 1)})
    return result


def summarize_trace(trace: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """给评测和日志用：重试了几次、哪步降级了、总耗时。"""
    retries = sum(1 for t in trace if t.get("attempt", 1) > 1)
    degraded = [t["step"] for t in trace if t.get("status") == "degraded"]
    errors = [t for t in trace if t.get("status") == "error"]
    total = next((t["elapsed_ms"] for t in reversed(trace) if t["step"] == "_total"), 0)
    return {"retries": retries, "degraded": degraded,
            "errors": len(errors), "total_ms": total}
