"""BM25 稀疏检索：擅长精确编号（E1017、条款 3.2），弥补向量的盲区。

中文没有空格，用 bi-gram 切词（不依赖 jieba）；英文与数字按整词切。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """
    中文 bi-gram + 英文数字整词。

    为什么不用 jieba：没装，装不上（沙箱网络受限）。
    bi-gram 是无分词器时的标准替代 —— 「倒车入库」会切成
    倒车/车入/入库，虽然多了些冗余，但检索时反而更鲁棒
    （不会因为分词边界不同而漏匹配）。
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)  # 英文/数字整词，如 e1024、xr2000
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):  # 连续中文段
        if len(seg) == 1:
            tokens.append(seg)
        else:
            tokens += [seg[i : i + 2] for i in range(len(seg) - 1)]
    return tokens


class BM25Retriever:
    """
    稀疏检索：基于词频的经典算法。

    擅长什么：精确编号（型号 E1024、条款 3.2、订单号）、
              固定术语 —— 这些词在向量空间里没有「语义」，向量检索会失效。
    不擅长什么：同义改写（「退了吧」找不到「申请退款」）。

    所以它和向量检索是互补的，两路合并才有意义。
    """

    def __init__(self, vectorstore) -> None:
        data = vectorstore.get(include=["documents", "metadatas"])
        raw_docs = data.get("documents") or []
        raw_metas = data.get("metadatas") or [{}] * len(raw_docs)

        self._docs: list[Document] = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(raw_docs, raw_metas, strict=False)
        ]
        self._tokenized = [tokenize(d.page_content) for d in self._docs]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def retrieve(self, query: str, k: int = 8) -> list[Document]:
        """按 BM25 分数取前 k 条，分数为 0（一个词都没命中）的直接丢掉"""
        if not query or not self._docs or self._bm25 is None:
            return []

        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        # 分数为 0 说明一个词都没命中，直接丢掉
        return [self._docs[i] for i in order if scores[i] > 0]
