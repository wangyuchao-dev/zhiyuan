"""在线问答流水线（检索 + 生成）。

拆成三步，方便普通问答与流式问答复用：
  rewrite_query()                     查询改写（消解指代）
  retrieve_multi()                    跨文档召回 → RRF 融合 → 精排 → 阈值过滤
  generate_answer() / stream_answer() 依据证据生成

上下文为空时直接返回「未找到相关资料」，不调模型。"""
import os
import sys

# 必须放在所有项目内 import 之前：
# 顶层 import 在模块加载时就执行，放到 __main__ 里再改 sys.path 已经晚了
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage

from config.setting import QaConfig
from indexing.vector_storage import VectorStoreManager

# 用绝对导入：querying 目录没有 __init__.py，
# 写 from .xxx import 直接运行时会报 "attempted relative import with no known parent package"
from querying.query_rewrite import QueryRewriter
from querying.answer import AnswerGenerator
from querying.retrank import RerankPipeline
from querying.hybrid_retriever import HybridRetriever
from querying.rrf import rrf_fuse


class RagPipeline:
    """RAG 在线问答流水线"""

    def __init__(self, config: QaConfig):
        self._config = config
        self._llm = ChatTongyi(model=self._config.chat_model)
        self._embeddings = DashScopeEmbeddings(model=self._config.embedding_model)
        self._vectorstores = {}  # 缓存向量库
        self._hybrid_cache = {}  # 缓存混合检索器（BM25 索引建一次就够）

        # 构造时就建好子组件（原来是懒加载，并行调用时可能拿到 None）
        self._rewriter = QueryRewriter(self._llm)
        self._rerank_pipeline = RerankPipeline(self._config)
        self._answer_gen = AnswerGenerator(self._llm)

    def _load_vectorstore(self, file_hash: str) -> Chroma:
        """加载向量库（带缓存）"""
        cache_key = file_hash
        if cache_key not in self._vectorstores:
            manager = VectorStoreManager(self._config, self._embeddings)
            self._vectorstores[cache_key] = manager.load_or_build(file_hash)
        return self._vectorstores[cache_key]

    def _get_hybrid(self, vectorstore, cache_key: str) -> HybridRetriever:
        """混合检索器按文档缓存：BM25 索引只需要建一次"""
        if cache_key not in self._hybrid_cache:
            self._hybrid_cache[cache_key] = HybridRetriever(vectorstore, self._config)
        return self._hybrid_cache[cache_key]

    def rewrite_query(self, question: str, chat_history: List[BaseMessage]) -> str:
        """
        对外暴露的查询改写入口。

        Agent 需要「先判要不要改写、再决定跑不跑」，直接访问 self._rag._rewriter
        属于跨层拿私有成员，这里包一层公开方法收口。
        """
        return self._rewriter.rewrite(question, chat_history)

    def query(self, file_hash: str, question: str, chat_history: List[BaseMessage], *,
              recall_k: int = 10, top_k: int = 4, precomputed_query: str = None) -> Dict[str, Any]:
        """
        单文档查询 —— 保留旧签名，内部走多文档逻辑（只传一份）。

        兼容原因：eval_retrieval.py、tests、老的调用方都还是按单文档写的，
        多文档改造不应该让它们全部返工。
        """
        return self.query_multi(
            [file_hash], question, chat_history,
            recall_k=recall_k, top_k=top_k, precomputed_query=precomputed_query,
        )

    def query_multi(self, file_hashes: List[str], question: str,
                    chat_history: List[BaseMessage], *,
                    recall_k: int = 10, top_k: int = 4,
                    precomputed_query: str = None) -> Dict[str, Any]:
        """
        跨多份文档的 RAG 查询（检索 + 生成，一次性返回）。

        内部拆成两步：retrieve_multi() 负责找证据，这里再调模型生成。
        拆开是为了让流式接口复用同一套检索逻辑（先检索，再逐字流式生成）。
        """
        found = self.retrieve_multi(
            file_hashes, question, chat_history,
            recall_k=recall_k, top_k=top_k, precomputed_query=precomputed_query,
        )

        # 没有任何一条证据过阈值：直接给固定答复，不调模型。
        # 既省一次调用，也从根上杜绝「拿无关内容硬编」。
        if not found["context"]:
            found["answer"] = "未找到相关资料。"
        else:
            # 注意：这里传【原始问题】而不是改写后的 search_query。
            # 改写的目的是「让检索更准」，不是「替换用户的话」——
            # 用户原话里的编号、时间、限定词一旦被改写丢掉，答案就会跑偏。
            found["answer"] = self._answer_gen.generate(
                question, chat_history, found["context"]
            )
        return found

    def retrieve_multi(self, file_hashes: List[str], question: str,
                       chat_history: List[BaseMessage], *,
                       recall_k: int = 10, top_k: int = 4,
                       precomputed_query: str = None) -> Dict[str, Any]:
        """
        只做检索，不生成答案。

        Args:
            file_hashes: 本次会话已绑定的所有文档指纹（一份文档一个 Chroma 集合）
            question: 用户原始问题
            chat_history: 对话上下文（含摘要与事实槽）
            recall_k: **每份文档**各自召回的条数
            top_k: 精排后最终保留的证据条数
            precomputed_query: 外部已经算好的改写结果（与意图识别并行时复用）

        Returns:
            dict: evidence / context / search_query / rewritten / doc_count / doc_total

        为什么跨文档要再过一次 RRF：
            每份文档都是各自召回的，各自的相关性分数不可比（不同集合的向量距离、
            不同 BM25 语料统计），所以只能按「在这份文档里排第几」来合并 ——
            这正是 RRF 的适用场景，和单文档内「向量 vs BM25」融合是同一个道理。
        """
        # 1. 查询改写（消除指代，把追问变成能独立检索的查询）
        search_query = precomputed_query or self._rewriter.rewrite(question, chat_history)

        # 2. 逐份文档做混合召回
        per_doc_hits: List[List] = []
        for file_hash in file_hashes or []:
            try:
                vectorstore = self._load_vectorstore(file_hash)
                docs = self._get_hybrid(vectorstore, file_hash).retrieve(
                    search_query, k=recall_k
                )
                if docs:
                    per_doc_hits.append(docs)
            except Exception as e:
                # 单份文档坏了（比如集合被清空、文件损坏）不该拖垮整个会话，
                # 跳过它继续查其余文档，把问题记到日志里。
                print(f"[pipeline] 文档 {file_hash[:8]} 召回失败，已跳过：{type(e).__name__}: {e}")

        # 3. 跨文档融合：只有一份就直接用，多份按排名融合
        if not per_doc_hits:
            recalled_docs = []
        elif len(per_doc_hits) == 1:
            recalled_docs = per_doc_hits[0]
        else:
            recalled_docs = rrf_fuse(per_doc_hits, k=self._config.rrf_k)

        # 4. 精排 + 阈值过滤（低于 QaConfig.min_score 的片段在这里被丢掉）
        evidence_list, context = self._rerank_pipeline.rerank(
            query=search_query,
            recalled_docs=recalled_docs,
            final_top_n=top_k,
        )

        # 5. 只返回检索结果，生成交给调用方（普通问答 / 流式问答共用这一份）
        return {
            "answer": "",
            "context": context,
            "evidence": evidence_list,
            "search_query": search_query,
            "rewritten": search_query != question,
            "doc_count": len(evidence_list),
            "doc_total": len(file_hashes or []),
        }

    def stream_answer(self, question: str, chat_history: List[BaseMessage],
                      context: str):
        """
        流式生成答案（逐 token 产出），供 SSE 接口使用。

        为什么单独开一个方法而不是给 query_multi 加参数：
            检索（改写 / 召回 / 精排）必须一次性完成 —— 你得先知道证据是什么，
            才能开始生成。所以流式只在「生成」这一段有意义，
            检索部分照旧同步执行，代码路径反而更简单。
        """
        return self._answer_gen.stream(question, chat_history, context)

    def generate_answer(self, question: str, chat_history: List[BaseMessage],
                        context: str) -> str:
        """
        一次性生成答案（与 stream_answer 配对）。

        上下文为空时直接返回「未找到相关资料。」，不走模型 ——
        这条短路保证了「库里没有就不编」这个产品承诺是代码层面的硬约束，
        而不只是提示词里的一句请求。
        """
        if not context:
            return "未找到相关资料。"
        return self._answer_gen.generate(question, chat_history, context)
