"""RRF（倒数排名融合）：score = Σ 1/(k + rank)，只看排名不看分数。

为什么不用加权求和：两路分数量纲不同（向量 0~1、BM25 0~30），直接相加没有意义。
k=60 是平滑常数，避免头部名次差距过分悬殊。"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections.abc import Sequence

from langchain_core.documents import Document


def _doc_key(doc: Document) -> str:
    """
    给文档算一个稳定且唯一的融合键。

    原来直接拿 page_content 当 key，会有两个问题：
      1. 两段文字完全相同的片段（比如每页页脚、重复条款）会被当成同一篇，融合时被吞掉一条；
      2. 内容一旦有空白差异又会被当成两篇，导致同一片段在结果里出现两次。
    用「来源 + 页码 + 内容」一起做哈希，既唯一又稳定。
    """
    meta = doc.metadata or {}
    raw = f"{meta.get('source', '')}|{meta.get('page', '')}|{doc.page_content}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def rrf_fuse(
    result_lists: Sequence[list[Document]],
    k: int = 60,
    top_n: int = None,
) -> list[Document]:
    """
    Reciprocal Rank Fusion（倒数排名融合）。

    公式：score(doc) = Σ 1 / (k + rank)

    为什么用 RRF 而不是加权求和两路分数：
      向量分是 0~1 的相关性，BM25 分可能是 0~30 的原始分，
      **量纲不同，直接相加没有意义**。
      RRF 只看「排第几」，天然避开这个问题。

    k 的作用：k 越小，排名靠前的权重差异越大；k=60 是论文默认值，
              也是比较稳健的选择。

    :param result_lists: 多路检索结果，每路是按相关性降序排好的 Document 列表
    :param k: RRF 平滑常数，默认 60
    :param top_n: 只返回前 N 条
    """
    if not result_lists:
        return []

    # 用稳定键做唯一标识（同一片段在多路里出现要合并，内容相同但出处不同不能合并）
    doc_map: dict[str, Document] = {}
    scores: dict[str, float] = {}

    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            key = _doc_key(doc)
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if top_n:
        ranked = ranked[:top_n]

    return [doc_map[key] for key, _ in ranked]


def rrf_scores(
    result_lists: Sequence[list[Document]],
    k: int = 60,
    weights: Sequence[float] = None,
) -> list[tuple]:
    """
    返回 [(doc, rrf_score)]。

    weights: 各路权重，默认等权。给 BM25 降权可以抑制「高频词把正确结果挤掉」的问题。
    """
    if not result_lists:
        return []

    if weights is None:
        weights = [1.0] * len(result_lists)

    doc_map: dict[str, Document] = {}
    scores: dict[str, float] = {}

    for w, results in zip(weights, result_lists, strict=False):
        for rank, doc in enumerate(results, start=1):
            key = _doc_key(doc)
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + w / (k + rank)

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return [(doc_map[key], score) for key, score in ranked]
