"""双路并行召回：向量与 BM25 同时跑，谁慢谁被超时切断，另一路照常返回。

为什么要并行：
    两路各 0.3~1.2 秒，串行就是相加；并行只取较慢的那一路，并且
    任一路挂了不影响另一路 —— 这是「降级」最便宜的实现方式。

为什么用线程而不是 asyncio：
    两路底层都是同步库（Chroma 客户端、rank_bm25），包一层 async 只是换个写法；
    真要压到极限再换异步 + 批量接口，现在没必要。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field

try:
    from core.errors import ErrorKind, ToolError, classify
except ImportError:  # pragma: no cover - 单独跑本文件时的兜底

    class ErrorKind:  # type: ignore
        TRANSIENT = "transient"
        LOGIC = "logic"

    class ToolError(Exception):
        def __init__(self, kind, code, detail=""):
            self.kind, self.code, self.detail = kind, code, detail

    def classify(exc=None, empty=False, status=None):
        return ToolError("logic", "UNKNOWN", str(exc))


@dataclass
class BranchResult:
    name: str
    ok: bool
    items: list[str] = field(default_factory=list)
    error: ToolError | None = None
    elapsed_ms: float = 0.0


def run_parallel(
    tasks: dict[str, Callable[[], Sequence[str]]], timeout: float = 2.0, workers: int = 4
) -> dict[str, BranchResult]:
    """并行执行多路召回。

    约定：
      · 每一路是 `() -> [item_id, ...]` 的函数（已按相关性排序）；
      · 超时/异常只影响该路，返回 ok=False，调用方据此决定降级；
      · 线程超时后线程池里的任务仍会跑完（Python 限制），所以**下游必须自己带超时**，
        这里只保证主流程不被拖住。
    """
    out: dict[str, BranchResult] = {}
    if not tasks:
        return out
    with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        futures = {name: pool.submit(_timed, fn) for name, fn in tasks.items()}
        for name, fut in futures.items():
            try:
                items, elapsed = fut.result(timeout=timeout)
                out[name] = BranchResult(name, True, list(items), None, elapsed)
            except FutureTimeout:
                out[name] = BranchResult(
                    name,
                    False,
                    [],
                    ToolError(ErrorKind.TRANSIENT, "TIMEOUT", f"{name} 超过 {timeout}s"),
                    elapsed_ms=timeout * 1000,
                )
            except Exception as exc:  # 兜底：任何一路炸都不许影响整体
                out[name] = BranchResult(name, False, [], classify(exc), 0.0)
    return out


def _timed(fn: Callable[[], Sequence[str]]) -> tuple[list[str], float]:
    t0 = time.perf_counter()
    items = list(fn() or [])
    return items, (time.perf_counter() - t0) * 1000


def rrf_merge(
    ranked: dict[str, Sequence[str]], weights: dict[str, float] | None = None, k: int = 60
) -> list[tuple[str, float]]:
    """RRF：按名次融合，不按分数相加。

    公式：score(d) = Σ w_b / (k + rank_b(d))
    为什么不用分数相加：向量相似度是 0~1，BM25 是 0~30+，量纲不同，相加等于让 BM25 说话。
    名次是对齐后的可比量，这也是跨文档/跨库融合唯一靠谱的方式。
    """
    weights = weights or dict.fromkeys(ranked, 1.0)
    scores: dict[str, float] = {}
    for name, items in ranked.items():
        w = weights.get(name, 1.0)
        for rank, item in enumerate(items, 1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def merge_with_fallback(
    results: dict[str, BranchResult], weights: dict[str, float] | None = None, k: int = 60
) -> tuple[list[tuple[str, float]], list[str]]:
    """融合 + 记录降级：返回 (合并结果, 失败分支名列表)。

    只有一路可用时不做加权（权重没意义），直接把那一路的名次当结果。
    """
    ok_ranked = {name: r.items for name, r in results.items() if r.ok and r.items}
    failed = [name for name, r in results.items() if not r.ok]
    if not ok_ranked:
        return [], failed
    if len(ok_ranked) == 1:
        name = next(iter(ok_ranked))
        return [(item, 1.0 / (k + rank)) for rank, item in enumerate(ok_ranked[name], 1)], failed
    return rrf_merge(ok_ranked, weights, k), failed
