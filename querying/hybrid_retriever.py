"""混合检索：向量 + BM25 双路召回，用 RRF 按排名融合。

两路分数量纲不同（向量 0~1、BM25 可能 0~30），不能直接相加，所以只比排名。

两路**并行**跑，且互不牵连：
    · 并行：各 0.3~1.2 秒，串行就是相加，并行只取较慢的那一路；
    · 隔离：任一路抛异常，另一路照常返回 —— 这是最便宜的降级，
      总比「BM25 索引坏了，整站问答不可用」强。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document

from config.setting import QaConfig
from querying.bm25_retriever import BM25Retriever
from querying.rrf import rrf_scores
from querying.vector_retriever import recall_with_scores

log = logging.getLogger("querying.hybrid")


class HybridRetriever:
    """
    混合检索：稠密向量 + BM25 稀疏检索，用 RRF 融合。

    为什么两路都要：
      - 向量擅长「意思像」：「退了吧」能找到「申请退款」
      - BM25 擅长「词对得上」：型号 E1024、条款 3.2、订单号这类
        精确编号在向量空间里没有语义，稠密检索经常召不回

    为什么用 RRF 融合：
      两路分数量纲不同（向量 0~1，BM25 可能 0~30），不能直接加权相加。
      RRF 只看排名，天然规避这个问题。
    """

    def __init__(self, vectorstore, config: QaConfig) -> None:
        self._vs = vectorstore
        self._config = config
        self._bm25 = BM25Retriever(vectorstore)
        # 最近一次检索的降级情况，供上层日志与评测观察
        self.last_degraded: list[str] = []

    # ---------- 单路封装 ----------
    def _dense(self, query: str, k: int) -> list[Document]:
        docs, _ = recall_with_scores(query, self._vs, k=k, min_score=0.0)
        return docs

    def _sparse(self, query: str, k: int) -> list[Document]:
        return self._bm25.retrieve(query, k=k)

    def _recall_both(self, query: str, k: int) -> tuple[list[Document], list[Document], list[str]]:
        """两路并行召回。返回 (dense, sparse, 失败分支名)。"""
        failed: list[str] = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_dense = pool.submit(self._dense, query, k)
            f_sparse = pool.submit(self._sparse, query, k)

            try:
                dense = f_dense.result()
            except Exception as exc:
                failed.append("dense")
                log.warning("向量召回失败，本路降级：%s: %s", type(exc).__name__, exc)
                dense = []

            try:
                sparse = f_sparse.result()
            except Exception as exc:
                failed.append("sparse")
                log.warning("BM25 召回失败，本路降级：%s: %s", type(exc).__name__, exc)
                sparse = []

        return dense or [], sparse or [], failed

    def retrieve_with_scores(
        self,
        query: str,
        k: int = None,
        rrf_top_n: int = None,
    ) -> tuple[list[Document], list[float]]:
        """返回 (融合后的文档, 各自的名次分)"""
        k = k or self._config.recall_k

        dense_docs, sparse_docs, failed = self._recall_both(query, k)
        self.last_degraded = failed

        # 两路都空：直接返回，别做无意义的融合
        if not dense_docs and not sparse_docs:
            return [], []

        # 只剩一路：权重没有意义（没有可比的另一路），按名次直接给递减分。
        # 这里刻意**不**把权重乘进去 —— 否则单路时 dense_weight=0.7 会让所有分数
        # 无端缩水，和「两路都成功」的结果不可比。
        if not dense_docs or not sparse_docs:
            docs = dense_docs or sparse_docs
            scores = [1.0 / (self._config.rrf_k + rank) for rank in range(1, len(docs) + 1)]
            if rrf_top_n:
                docs, scores = docs[:rrf_top_n], scores[:rrf_top_n]
            return docs, scores

        pairs = rrf_scores(
            [dense_docs, sparse_docs],
            k=self._config.rrf_k,
            weights=[self._config.dense_weight, self._config.bm25_weight],
        )
        if rrf_top_n:
            pairs = pairs[:rrf_top_n]

        return [d for d, _ in pairs], [s for _, s in pairs]

    def retrieve(self, query: str, k: int = None, rrf_top_n: int = None) -> list[Document]:
        """混合召回的简化入口，只要文档不要分数"""
        docs, _ = self.retrieve_with_scores(query, k=k, rrf_top_n=rrf_top_n)
        return docs
