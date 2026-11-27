"""向量召回：调 Chroma 的相似度检索，并把分数归一到 0~1。

建库时指定了 cosine，所以 relevance score 有绝对含义，可以直接设阈值。"""
from typing import List, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document


def recall_with_scores(
        query: str,
        vectorstore: Chroma,
        k: int = 10,
        min_score: float = 0.0
) -> Tuple[List[Document], List[float]]:
    """
    向量检索召回
    :param query: 查询字符串
    :param vectorstore: Chroma向量库实例
    :param k: 召回数量
    :param min_score: 相关性下限（0~1，越大越严格），低于它的结果直接丢弃
    :return: (文档列表, 相关性分数列表) —— 分数越大越相关

    说明：
    建库时已指定 hnsw:space="cosine"，所以 similarity_search_with_relevance_scores
    返回的是 1 - cosine_distance/2，范围 0~1（1 表示完全同向）。
    这才是能设阈值的分数——不像 L2 距离那样上万、还没有上界。
    """
    if not query or not vectorstore:
        return [], []

    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)

    docs = []
    scores = []
    for doc, score in results:
        # 低于阈值直接丢掉：宁可返回空，也别把无关内容硬塞给模型
        if score < min_score:
            continue
        doc.page_content = doc.page_content.strip()
        docs.append(doc)
        scores.append(score)

    return docs, scores
