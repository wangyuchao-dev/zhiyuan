"""精排 + 阈值过滤：用专用 rerank 模型对融合结果逐条重打分。

QaConfig.min_score 在这里生效：低于阈值的片段直接丢弃 ——
宁可变空让模型说「未找到」，也不把无关内容硬塞进上下文。
精排接口异常时降级为按召回顺序返回，不让整个问答失败。"""
import os
import sys

# 必须放在所有项目内 import 之前：
# 顶层 import 在模块加载时就执行，放到 __main__ 里再改 sys.path 已经晚了
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import List, Tuple, Optional

import dashscope
from langchain_core.documents import Document

from config.setting import QaConfig


@dataclass
class RetrievedEvidence:
    """检索到的证据片段（经过重排序后输出的结构化信息）"""
    content: str # 证据内容
    source: str # 证据来源
    page: Optional[int] # 证据所在页码（0-based）
    score: float # 证据相关性分数


class RerankPipeline:
    """
    重排序 Pipeline：召回 → 专用 Rerank 模型精排
    """

    def __init__(self, config: QaConfig, model: str = None) -> None:
        self.__config = config
        self.__model = model or config.rerank_model

    def rerank(self,
               query: str,
               recalled_docs: List[Document],
               final_top_n: int = 4) -> Tuple[List[RetrievedEvidence], str]:
        """
        重排序流程入口

        :param query: 检索查询
        :param recalled_docs: 已召回的文档列表
        :param final_top_n: 最终输出的证据数量
        :return: (evidence_list, formatted_context)

        相关性下限（QaConfig.min_score）就是在这一步生效的：
        精排模型给出的 relevance_score 是 0~1，低于阈值的片段直接丢弃 ——
        宁可返回空、让模型说「未找到相关资料」，也不要把无关内容硬塞进去。
        """

        # 1. 预过滤（长度 + 内容去重）
        pre_docs = self.__filter(recalled_docs)

        # 召回为空时直接返回：不去调精排接口（传空 documents 会报错）
        if not pre_docs:
            return [], ""

        # 2. 专用 Rerank 模型精排（返回 [{index, relevance_score}]，按相关性降序）
        docs = [doc.page_content for doc in pre_docs]
        try:
            resp = dashscope.TextReRank.call(
                model=self.__model,
                query=query,
                documents=docs,
                top_n=min(final_top_n, len(docs)),
                return_documents=False,
            )
            results = resp.output["results"] if getattr(resp, "output", None) else []
        except Exception as e:
            # 精排是「锦上添花」不是「必经之路」：
            # 接口抖动时退化成按召回顺序取前 N 条，别让整个问答 500。
            print(f"[rerank] 精排调用失败，退化为召回顺序：{type(e).__name__}: {e}")
            results = [{"index": i, "relevance_score": None} for i in range(len(pre_docs))]

        # 注意：res["index"] 是相对 documents（也就是 pre_docs）的下标，
        # 不能用 recalled_docs 取 —— 一旦预过滤剔掉了文档，下标就全错位了
        ranked = [(pre_docs[res["index"]], res.get("relevance_score"))
                  for res in results
                  if res.get("index") is not None and res["index"] < len(pre_docs)]

        # 3. 阈值过滤 + 截断
        min_score = getattr(self.__config, "min_score", 0.0)
        kept = []
        for doc, score in ranked:
            # score 为 None 表示精排降级了，此时不过滤（否则会把结果全丢光）
            if score is not None and score < min_score:
                continue
            kept.append((doc, score))
        ranked = kept[:final_top_n]

        # 4. 组装证据 + 格式化上下文
        evidence_list, context_lines = [], []
        for idx, (doc, score) in enumerate(ranked, 1):
            meta = doc.metadata or {}
            ev = RetrievedEvidence(
                content=doc.page_content,
                source=meta.get("source", "Unknown"),
                page=meta.get("page"),
                score=float(score) if score is not None else 0.0
            )
            evidence_list.append(ev)
            if ev.content:
                page_str = f"第{ev.page + 1}页" if ev.page is not None else "未知页"
                context_lines.append(f"[证据{idx} | 来源：{ev.source} | {page_str}]\n{ev.content}")
        context = "\n\n".join(context_lines)

        return evidence_list, context

    def __filter(self, docs: List[Document], min_len: int = 40, max_len: int = 2000) -> List[Document]:
        """过滤掉太短/太长的片段，再按内容去重"""
        filtered = []
        for doc in docs:
            text = doc.page_content
            if min_len <= len(text) <= max_len:
                filtered.append(doc)

        # 去重：内容完全一样的只留一条
        seen = set()
        kept = []
        for doc in filtered:
            key = doc.page_content
            if key not in seen:
                seen.add(key)
                kept.append(doc)
        return kept
