"""混合检索：向量 + BM25 双路召回，用 RRF 按排名融合。

两路分数量纲不同（向量 0~1、BM25 可能 0~30），不能直接相加，所以只比排名。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple

from langchain_core.documents import Document

from config.setting import QaConfig
from querying.vector_retriever import recall_with_scores
from querying.bm25_retriever import BM25Retriever
from querying.rrf import rrf_scores


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

    def retrieve_with_scores(
            self,
            query: str,
            k: int = None,
            rrf_top_n: int = None,
    ) -> Tuple[List[Document], List[float]]:
        """返回 (融合后的文档, 各自的 RRF 分数)"""
        k = k or self._config.recall_k

        # 两路并行召回（都是本地计算，串行也够快）
        dense_docs, _ = recall_with_scores(query, self._vs, k=k, min_score=0.0)
        sparse_docs = self._bm25.retrieve(query, k=k)

        pairs = rrf_scores([dense_docs, sparse_docs], k=self._config.rrf_k,
                           weights=[self._config.dense_weight, self._config.bm25_weight])
        if rrf_top_n:
            pairs = pairs[:rrf_top_n]

        docs = [d for d, _ in pairs]
        scores = [s for _, s in pairs]
        return docs, scores

    def retrieve(self, query: str, k: int = None, rrf_top_n: int = None) -> List[Document]:
        """混合召回的简化入口，只要文档不要分数"""
        docs, _ = self.retrieve_with_scores(query, k=k, rrf_top_n=rrf_top_n)
        return docs
