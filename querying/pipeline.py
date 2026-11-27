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

import logging
from typing import Any

from langchain_chroma import Chroma
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.messages import BaseMessage

from config.setting import QaConfig
from core.cache import build_cache, corpus_fingerprint
from core.context_budget import estimate_tokens
from core.postcheck import postcheck
from indexing.vector_storage import VectorStoreManager
from querying.answer import AnswerGenerator
from querying.hybrid_retriever import HybridRetriever

log = logging.getLogger("querying.pipeline")

# 用绝对导入：querying 目录没有 __init__.py，
# 写 from .xxx import 直接运行时会报 "attempted relative import with no known parent package"
from querying.query_rewrite import QueryRewriter
from querying.retrank import RerankPipeline, RetrievedEvidence
from querying.rrf import rrf_fuse


def _render_evidence(ev: RetrievedEvidence, idx: int) -> str:
    """单条证据的上下文格式。

    格式必须与提示词约定一致（`编号对应上下文的【证据N】`），
    也和 RerankPipeline 内部保持一致 —— 两处格式不同会让引用编号对不上。
    """
    page = f"第{ev.page + 1}页" if ev.page is not None else "未知页"
    return f"[证据{idx} | 来源：{ev.source} | {page}]\n{ev.content}"


def _dump_evidence(ev: RetrievedEvidence) -> dict:
    """证据对象 → 可 JSON 序列化的 dict（缓存用）"""
    return {"content": ev.content, "source": ev.source, "page": ev.page, "score": ev.score}


def _build_skills():
    """建技能注册表；目录不存在或解析失败就返回 None（绝不阻断问答）。

    技能是「锦上添花」：它规定同类问题的组织方式，不该成为启动的硬依赖。
    """
    from pathlib import Path

    from skills.registry import SkillRegistry

    root = Path(__file__).resolve().parent.parent / "skills"
    if not root.is_dir():
        return None
    try:
        reg = SkillRegistry(str(root))
        reg.index()
        return reg
    except Exception as exc:
        log.warning("技能注册表加载失败，按无技能运行：%s: %s", type(exc).__name__, exc)
        return None


class RagPipeline:
    """RAG 在线问答流水线"""

    def __init__(self, config: QaConfig):
        self._config = config
        self._llm = ChatTongyi(model=self._config.chat_model)
        self._embeddings = DashScopeEmbeddings(model=self._config.embedding_model)
        self._vectorstores = {}  # 缓存向量库
        self._hybrid_cache = {}  # 缓存混合检索器（BM25 索引建一次就够）

        # 检索缓存：key 里带「语料指纹」，加/删文档后自动失效，不靠人记得清。
        # 没配 REDIS_URL 时自动退化成进程内缓存，服务照常起。
        self._cache = build_cache(os.getenv("REDIS_URL"), namespace="zy")

        # 技能注册表：只读 front-matter 建目录，命中才展开正文。
        # 几十个技能也只占几行上下文 —— 全塞进提示词等于每次都付一遍钱。
        self._skills = _build_skills()

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

    def skills_text(self, question: str) -> str:
        """按问题匹配技能，返回要注入提示词的正文（没命中就是空串）。

        匹配靠 front-matter 里的 keywords + summary，不做语义检索 ——
        技能数量级是几十，关键词足够；真到几百个再换向量召回。
        """
        if not self._skills:
            return ""
        try:
            return self._skills.render(question, max_skills=1)
        except Exception as exc:  # 技能坏了不能拖垮问答
            log.warning("技能渲染失败，按无技能处理：%s: %s", type(exc).__name__, exc)
            return ""

    def rewrite_query(self, question: str, chat_history: list[BaseMessage]) -> str:
        """
        对外暴露的查询改写入口。

        Agent 需要「先判要不要改写、再决定跑不跑」，直接访问 self._rag._rewriter
        属于跨层拿私有成员，这里包一层公开方法收口。
        """
        return self._rewriter.rewrite(question, chat_history)

    def query(
        self,
        file_hash: str,
        question: str,
        chat_history: list[BaseMessage],
        *,
        recall_k: int = 10,
        top_k: int = 4,
        precomputed_query: str = None,
    ) -> dict[str, Any]:
        """
        单文档查询 —— 保留旧签名，内部走多文档逻辑（只传一份）。

        兼容原因：eval_retrieval.py、tests、老的调用方都还是按单文档写的，
        多文档改造不应该让它们全部返工。
        """
        return self.query_multi(
            [file_hash],
            question,
            chat_history,
            recall_k=recall_k,
            top_k=top_k,
            precomputed_query=precomputed_query,
        )

    def query_multi(
        self,
        file_hashes: list[str],
        question: str,
        chat_history: list[BaseMessage],
        *,
        recall_k: int = 10,
        top_k: int = 4,
        precomputed_query: str = None,
    ) -> dict[str, Any]:
        """
        跨多份文档的 RAG 查询（检索 + 生成，一次性返回）。

        内部拆成两步：retrieve_multi() 负责找证据，这里再调模型生成。
        拆开是为了让流式接口复用同一套检索逻辑（先检索，再逐字流式生成）。
        """
        found = self.retrieve_multi(
            file_hashes,
            question,
            chat_history,
            recall_k=recall_k,
            top_k=top_k,
            precomputed_query=precomputed_query,
        )

        # 没有任何一条证据过阈值：直接给固定答复，不调模型。
        # 既省一次调用，也从根上杜绝「拿无关内容硬编」。
        if not found["context"]:
            found["answer"] = "未找到相关资料。"
        else:
            # 注意：这里传【原始问题】而不是改写后的 search_query。
            # 改写的目的是「让检索更准」，不是「替换用户的话」——
            # 用户原话里的编号、时间、限定词一旦被改写丢掉，答案就会跑偏。
            raw = self._answer_gen.generate(
                question, chat_history, found["context"], self.skills_text(question)
            )
            found["answer"], found["postcheck"] = self.verify(raw, found["evidence"])
        return found

    def verify(self, answer: str, evidence_list: list[RetrievedEvidence]) -> tuple[str, dict]:
        """生成后的校验：抓「证据里没有的错误码/数字/日期」。

        刻意不做「自动改写数字」—— 把 15 分钟改回 5 分钟，等于用程序替模型
        编第二个答案，比原来的错更危险。这里只做两件事：
          · 关键事实在证据里找不到 → 整条替换为保守话术；
          · 引用了不存在的片段编号 → 只移除编号，重排会把引用指错。
        """
        contents = [e.content for e in evidence_list]
        report = postcheck(answer, contents, len(contents))
        if not report.ok:
            log.warning(
                "后置校验未通过 action=%s kinds=%s",
                report.action,
                [v.kind for v in report.violations],
            )
        return report.answer, report.as_dict()

    def retrieve_multi(
        self,
        file_hashes: list[str],
        question: str,
        chat_history: list[BaseMessage],
        *,
        recall_k: int = 10,
        top_k: int = 4,
        precomputed_query: str = None,
    ) -> dict[str, Any]:
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

        # 1.5 检索缓存：改写之后、召回之前。
        # key 用「改写后的查询」而不是原问题 —— 同一个问题在不同上下文里
        # 改写结果不同（「那它呢」指代不同），用原问题做 key 会串答案。
        # 参数也进 key：调了 recall_k / min_score 就应该是另一个结果。
        fp = corpus_fingerprint(file_hashes or [])
        cache_key = self._cache.build_key(
            "retrieve",
            search_query,
            corpus_fp=fp,
            params={
                "recall_k": recall_k,
                "top_k": top_k,
                "min_score": self._config.min_score,
                # 换了 embedding / rerank 模型，旧向量与旧分数都不再可比
                "emb": self._config.embedding_model,
                "rerank": self._config.rerank_model,
            },
        )
        cached = self._cache.get_json(cache_key)
        if cached and "context" in cached:
            cached["evidence"] = [RetrievedEvidence(**e) for e in cached.get("evidence", [])]
            cached["cached"] = True
            return cached

        # 2. 逐份文档做混合召回
        per_doc_hits: list[list] = []
        for file_hash in file_hashes or []:
            try:
                vectorstore = self._load_vectorstore(file_hash)
                docs = self._get_hybrid(vectorstore, file_hash).retrieve(search_query, k=recall_k)
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
        evidence_list, _ = self._rerank_pipeline.rerank(
            query=search_query,
            recalled_docs=recalled_docs,
            final_top_n=top_k,
        )

        # 4.5 上下文预算：超出总预算的证据整条丢弃（不截断，半句话比没有更误导），
        # 编号按保留后的顺序重排 —— 否则会出现 [1][3][5]，模型引用会乱。
        evidence_list, context, ctx_stats = self._fit_budget(evidence_list)

        # 5. 只返回检索结果，生成交给调用方（普通问答 / 流式问答共用这一份）
        result = {
            "answer": "",
            "context": context,
            "evidence": evidence_list,
            "search_query": search_query,
            "rewritten": search_query != question,
            "doc_count": len(evidence_list),
            "doc_total": len(file_hashes or []),
            "context_stats": ctx_stats,
            "cached": False,
        }
        # 命中率低的场景（同一份制度被反复问）收益最大；空结果不缓存（Cache 内部已挡）
        self._cache.set_json(
            cache_key,
            {**result, "evidence": [_dump_evidence(e) for e in evidence_list]},
        )
        return result

    def _fit_budget(self, evidence_list: list[RetrievedEvidence]):
        """按总预算挑选证据，返回 (保留的证据, 上下文文本, 统计)。

        按精排分数降序（rerank 已排好），装不下就停 —— 后面的分数更低，
        硬塞进来只会稀释重点。至少保留一条：全丢等于告诉用户「没有」，
        而实际上第一条可能就是答案。
        """
        budget = getattr(self._config, "context_budget", 2600)
        kept: list[RetrievedEvidence] = []
        used = 0
        for ev in evidence_list:
            cost = estimate_tokens(_render_evidence(ev, len(kept) + 1))
            if kept and used + cost > budget:
                break
            kept.append(ev)
            used += cost

        context = "\n\n".join(_render_evidence(ev, idx) for idx, ev in enumerate(kept, 1))
        stats = {
            "budget": budget,
            "used_tokens": used,
            "kept": len(kept),
            "dropped": max(0, len(evidence_list) - len(kept)),
        }
        return kept, context, stats

    def stream_answer(self, question: str, chat_history: list[BaseMessage], context: str):
        """
        流式生成答案（逐 token 产出），供 SSE 接口使用。

        为什么单独开一个方法而不是给 query_multi 加参数：
            检索（改写 / 召回 / 精排）必须一次性完成 —— 你得先知道证据是什么，
            才能开始生成。所以流式只在「生成」这一段有意义，
            检索部分照旧同步执行，代码路径反而更简单。
        """
        return self._answer_gen.stream(question, chat_history, context, self.skills_text(question))

    def generate_answer(self, question: str, chat_history: list[BaseMessage], context: str) -> str:
        """
        一次性生成答案（与 stream_answer 配对）。

        上下文为空时直接返回「未找到相关资料。」，不走模型 ——
        这条短路保证了「库里没有就不编」这个产品承诺是代码层面的硬约束，
        而不只是提示词里的一句请求。
        """
        if not context:
            return "未找到相关资料。"
        return self._answer_gen.generate(
            question, chat_history, context, self.skills_text(question)
        )
