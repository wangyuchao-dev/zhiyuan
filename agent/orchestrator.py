"""意图 → 路由 → 检索 → 生成的总编排（Agent）。

三段式：
  prepare()      判意图、抽事实、选路由、检索证据（一次性问答与流式共用）
  chat()         一次性取回整段答案
  chat_stream()  逐 token 产出，供 SSE 接口

意图判定是三级：强特征（寒暄/投诉）→ 文档特征词 → 都没命中才调模型。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.intent import IntentRecognizer
from agent.memory import Memory
from config.prompts import (
    CHITCHAT_PROMPT,
    COMPLAINT_PROMPT,
)
from config.setting import QaConfig
from querying.pipeline import RagPipeline


class Agent:
    """
    RAG Agent：意图识别 → 路由 → 记忆 → 回答

    路由规则：
      chitchat  → 闲聊对话（不查库）
      doc_query → 走 RAG（改写 → 召回 → 精排 → 生成）
      other     → 有文档就当文档问答，没文档就走闲聊兜底
    """

    def __init__(self, config: QaConfig) -> None:
        self._config = config
        self._llm = ChatTongyi(model=self._config.chat_model)

        # 业务分支专用：限制输出长度。
        # 实测 47 字的回答也要 9 秒，说明慢在模型响应而非 token 数；
        # 但限长能避免它越写越长（之前退款分支跑到 30 秒）。
        self._business_llm = ChatTongyi(model=self._config.chat_model, max_tokens=320)

        self._intent = IntentRecognizer(self._llm)
        self._rag = RagPipeline(config)

        self._memories: dict[str, Memory] = {}
        # session_id -> 已绑定的文档指纹列表（有顺序、去重）。
        # 从单值改成列表，是为了支持「一次会话传多份文档、跨文档提问」：
        # 每份 PDF 仍然各自一个 Chroma 集合，检索时逐份召回再按排名融合。
        self._session_docs: dict[str, list[str]] = {}

        self._chitchat_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", CHITCHAT_PROMPT),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            | self._llm
            | StrOutputParser()
        )

        # 业务分支各自的链：同一套消息结构，只是 system 提示词不同
        # （目前 BUSINESS_PROMPTS 里只有 complaint 一条，加新分支不用改路由）
        self._business_chains = {}
        for name, system_prompt in self.BUSINESS_PROMPTS.items():
            self._business_chains[name] = (
                ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt + "\n\n## 参考资料（可能为空）：\n{context}"),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )
                | self._business_llm
                | StrOutputParser()
            )

    # ---------- 会话与文档绑定 ----------
    def _get_memory(self, session_id: str) -> Memory:
        if session_id not in self._memories:
            self._memories[session_id] = Memory(session_id)
        return self._memories[session_id]

    def bind_document(self, session_id: str, file_hash: str) -> None:
        """
        上传并入库完成后，把文档绑定到当前会话。

        支持累加：同一会话可以连续上传多份 PDF，检索时会跨这些文档一起查。
        重复上传同一份文件不会重复绑定（同一指纹只留一条）。
        """
        docs = self._session_docs.setdefault(session_id, [])
        if file_hash not in docs:
            docs.append(file_hash)

    def get_bound_document(self, session_id: str):
        """返回本次会话已绑定的文档指纹（单文档时为长度 1 的列表，未绑定为空列表）"""
        return list(self._session_docs.get(session_id, []))

    def get_bound_documents(self, session_id: str) -> list[str]:
        """语义更明确的别名，供接口层使用"""
        return self.get_bound_document(session_id)

    def unbind_documents(self, session_id: str) -> None:
        """解绑本会话的全部文档（不动记忆）"""
        self._session_docs.pop(session_id, None)

    # ---------- 三个分支 ----------
    def _answer_chitchat(self, question: str, memory: Memory) -> str:
        return self._chitchat_chain.invoke(
            {
                # 统一走 build_context：最近消息 + 摘要 + 关键事实
                "chat_history": memory.build_context(self._config.history_limit),
                "input": question,
            }
        )

    def _answer_with_prompt(self, intent: str, question: str, memory: Memory) -> dict[str, Any]:
        """
        业务分支（目前只有投诉）：纯对话，不查文档。

        投诉要的是「先接住情绪 + 记录转交」，不是给一堆制度条款，
        所以这里不做检索，直接用该分支的提示词生成。
        """
        answer = self._business_chains[intent].invoke(
            {
                "context": "",
                "chat_history": memory.build_context(self._config.history_limit),
                "input": question,
            }
        )
        return {"answer": answer, "evidence": [], "search_query": question}

    # ---------- 主入口 ----------
    def prepare(self, session_id: str, question: str) -> dict[str, Any]:
        """
        「想清楚怎么答」这一步：判意图 → 取事实 → 选路由 → 检索证据。

        为什么单独抽出来：
            普通问答（/api/chat）和流式问答（/api/chat/stream）的前半段完全一样，
            只差「最后怎么把答案吐出去」。抽成 prepare() 之后两条路径共用一份逻辑，
            不会出现「流式接口少判了一次意图」这种漂移。

        返回的 plan 结构：
            kind    = "answer"（直接给固定答复）/ "stream"（要走模型生成）
            route   = chitchat / complaint / rag / error
            context = rag 分支的证据上下文（其余为空串）
            evidence= 证据列表
        """
        question = (question or "").strip()
        if not question:
            return {
                "kind": "answer",
                "answer": "请输入问题。",
                "intent": "",
                "confidence": 0.0,
                "route": "empty",
                "evidence": [],
                "search_query": "",
                "context": "",
                "session_id": session_id,
            }

        memory = self._get_memory(session_id)
        hist_msgs = memory.build_context(self._config.history_limit)
        need_rewrite = self._need_rewrite(question)

        # ---- 第 1 道优化：规则快判意图 ----
        # 三级判定：① 强特征（寒暄/投诉）直接定 ② 文档特征词定 rag ③ 都没命中才调模型
        quick = self._quick_intent(question)

        # 闲聊走快通道：不查文档、不判槽位，直接交给对话链
        if quick == "chitchat":
            return {
                "kind": "stream",
                "route": "chitchat",
                "intent": "chitchat",
                "confidence": 1.0,
                "evidence": [],
                "search_query": question,
                "context": "",
                "session_id": session_id,
            }

        slots: dict[str, Any] = {}
        if quick:
            # 规则命中：省掉一次模型调用（约 3 秒）
            intent, confidence = quick, 0.9
            # 需要改写才调模型；问题本身完整就直接拿去检索（再省约 2 秒）
            search_query = (
                self._rag.rewrite_query(question, hist_msgs) if need_rewrite else question
            )
        else:
            # 规则判不出来才走模型兜底。
            # 意图识别和改写互不依赖，能并行就并行 —— 但问题完整时不去白跑改写。
            hist_text = memory.history.to_plain_text(limit=self._config.history_limit)
            if need_rewrite:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    f_intent = pool.submit(self._intent.recognize, question, hist_text)
                    f_query = pool.submit(self._rag.rewrite_query, question, hist_msgs)
                    r = f_intent.result()
                    search_query = f_query.result()
            else:
                r = self._intent.recognize(question, hist_text)
                search_query = question
            intent = r.intents[0] if r.intents else "general"
            confidence = r.confidence
            slots = r.slots or {}

        # ---- 事实槽：模型给的 slots + 从问题里正则抽出来的实体 ----
        # 这条以前是断的：update_facts 从来没被调用，事实槽永远是空的
        facts = dict(self._extract_entities(question))
        facts.update({k: v for k, v in slots.items() if v})
        memory.update_facts(facts)

        # ---- 路由 ----
        base = {
            "intent": intent,
            "confidence": confidence,
            "search_query": search_query,
            "session_id": session_id,
        }

        if intent == "chitchat":
            return {**base, "kind": "stream", "route": "chitchat", "evidence": [], "context": ""}

        if intent in self.BUSINESS_PROMPTS:
            # 投诉分支：只共情 + 记录，不查文档
            return {**base, "kind": "stream", "route": intent, "evidence": [], "context": ""}

        if not self._session_docs.get(session_id):
            # 该查文档但这个会话还没传文档
            return {
                **base,
                "kind": "answer",
                "route": "no_doc",
                "evidence": [],
                "context": "",
                "answer": "这个问题需要查文档，但当前会话还没有绑定文档。请先上传一份 PDF。",
            }

        # ---- RAG：先检索（证据、上下文都在这一步拿到）----
        found = self._rag.retrieve_multi(
            file_hashes=self._session_docs.get(session_id) or [],
            question=question,
            chat_history=memory.build_context(self._config.history_limit),
            recall_k=self._config.recall_k,
            top_k=self._config.rerank_top_n,
            precomputed_query=search_query,
        )
        if not found["context"]:
            # 没有一条证据过阈值：不调模型，直接说没找到
            return {
                **base,
                "kind": "answer",
                "route": "rag",
                "evidence": [],
                "context": "",
                "answer": "未找到相关资料。",
            }

        return {
            **base,
            "kind": "stream",
            "route": "rag",
            "evidence": found["evidence"],
            "context": found["context"],
            "search_query": found["search_query"],
        }

    def _stream_chain(self, plan: dict[str, Any], question: str, memory: Memory):
        """
        按 route 选一条链做流式生成。

        三种情况统一成「产出字符串片段」的迭代器，接口层不用关心是哪条链。
        """
        if plan["route"] == "rag":
            return self._rag.stream_answer(
                question, memory.build_context(self._config.history_limit), plan["context"]
            )
        if plan["route"] == "chitchat":
            return self._chitchat_chain.stream(
                {
                    "chat_history": memory.build_context(self._config.history_limit),
                    "input": question,
                }
            )
        # 业务分支（投诉）
        return self._business_chains[plan["route"]].stream(
            {
                "context": "",
                "chat_history": memory.build_context(self._config.history_limit),
                "input": question,
            }
        )

    def _remember(self, session_id: str, question: str, answer: str) -> None:
        """一轮问答结束后写回记忆（历史 + 触发摘要）"""
        memory = self._get_memory(session_id)
        memory.add_user(question, self._llm)
        memory.add_ai(answer or "", self._llm)

    def chat(self, session_id: str, question: str) -> dict[str, Any]:
        """一次性问答：走完 prepare() 后把整段答案取回来"""
        question = (question or "").strip()
        plan = self.prepare(session_id, question)

        if plan["kind"] == "answer":
            answer = plan["answer"]
        else:
            memory = self._get_memory(session_id)
            if plan["route"] == "rag":
                raw = self._rag.generate_answer(
                    question,
                    memory.build_context(self._config.history_limit),
                    plan["context"],
                )
                # 生成后校验：证据里找不到的错误码/数字/日期一律拦掉换保守话术。
                # 放在这里而不是流式里，是因为流式已经把字吐给前端了，收不回来。
                answer, plan["postcheck"] = self._rag.verify(raw, plan["evidence"])
            elif plan["route"] == "chitchat":
                answer = self._answer_chitchat(question, memory)
            else:
                answer = self._answer_with_prompt(plan["route"], question, memory)["answer"]

        self._remember(session_id, question, answer)

        return {
            "answer": answer,
            "intent": plan["intent"],
            "confidence": plan["confidence"],
            "route": plan["route"],
            "evidence": plan["evidence"],
            "search_query": plan["search_query"],
            "session_id": session_id,
            "postcheck": plan.get("postcheck"),
        }

    def chat_stream(self, session_id: str, question: str):
        """
        流式问答：逐 token 产出，供 SSE 接口使用。

        事件顺序：
            {"event": "meta"}      → 意图 / 路由 / 检索词（前端可先渲染标签）
            {"event": "evidence"}  → 证据列表（答案之前给，方便边看边核）
            {"event": "delta"}     → 文本片段（可能很多条）
            {"event": "done"}      → 结束（此时记忆已写回）
        出错时给 {"event": "error"}，不让连接悬着。
        """
        question = (question or "").strip()
        try:
            plan = self.prepare(session_id, question)
            yield {
                "event": "meta",
                "intent": plan["intent"],
                "route": plan["route"],
                "confidence": plan["confidence"],
                "search_query": plan["search_query"],
            }
            yield {
                "event": "evidence",
                "evidence": [
                    {
                        "content": ev.content,
                        "source": ev.source,
                        "page": ev.page,
                        "score": round(float(ev.score), 4),
                    }
                    for ev in plan["evidence"]
                ],
            }

            if plan["kind"] == "answer":
                answer = plan["answer"]
                if answer:
                    yield {"event": "delta", "text": answer}
            else:
                memory = self._get_memory(session_id)
                parts = []
                for chunk in self._stream_chain(plan, question, memory):
                    if not chunk:
                        continue
                    parts.append(chunk)
                    yield {"event": "delta", "text": chunk}
                answer = "".join(parts)

            # 流式已经把字吐给前端了，正文改不了；这里只做检测，
            # 把结果挂在 done 事件上 —— 前端可据此提示「本条未通过校验」，
            # 离线统计也能拿到拦截率。检测不影响已经显示的答案。
            # 固定答复（kind=answer，如「未找到相关资料」）是代码写死的，不用校验。
            postcheck_info = None
            if plan["kind"] != "answer" and plan["route"] == "rag" and answer:
                _, postcheck_info = self._rag.verify(answer, plan["evidence"])

            self._remember(session_id, question, answer)
            yield {"event": "done", "answer": answer, "postcheck": postcheck_info}
        except Exception as e:
            import traceback

            traceback.print_exc()
            yield {"event": "error", "message": f"{type(e).__name__}: {e}"}

    # ---------- 业务分支 → 提示词 ----------
    # 「一个意图一块提示词」的落地位置：意图识别只负责分诊，
    # 具体怎么答由各自的提示词决定（rag → ANSWER_PROMPT，闲聊 → CHITCHAT_PROMPT，
    # 投诉 → COMPLAINT_PROMPT）。
    # 目前只有「投诉」需要独立提示词；加新业务分支就在这加一条，路由代码不用动。
    BUSINESS_PROMPTS = {
        "complaint": COMPLAINT_PROMPT,
    }

    # ---------- 规则快判 ----------
    CHITCHAT_WORDS = (
        "你好",
        "您好",
        "谢谢",
        "再见",
        "拜拜",
        "在吗",
        "你是谁",
        "你能做什么",
        "你能干什么",
        "哈喽",
        "早上好",
        "晚安",
    )

    # 投诉类关键词（情绪优先，命中就分到投诉分支）
    BUSINESS_KEYWORDS = {
        "complaint": (
            "投诉",
            "不满",
            "举报",
            "意见",
            "太麻烦",
            "太差",
            "什么破",
            "抱怨",
            "不合理",
            "很生气",
            "受不了",
        ),
    }

    # 文档问答的强特征词：命中就按 rag 处理，不必再调模型分诊
    # 覆盖两类：① 疑问词（什么/怎么/多少/几/谁/是否…）② 制度语境词（规定/流程/审批/条…）
    RAG_HINTS = (
        "什么",
        "怎么",
        "如何",
        "多少",
        "多久",
        "几",
        "哪",
        "谁",
        "是否",
        "能不能",
        "可不可以",
        "有没有",
        "规定",
        "流程",
        "标准",
        "要求",
        "审批",
        "申请",
        "第",
        "条",
    )

    def _quick_intent(self, q: str):
        """
        三级规则快判，判不出来返回 None（交给模型兜底）。

        ① 强特征：寒暄 / 投诉 —— 有明确词表，命中即定
        ② 文档特征：带「怎么/多少/第几条/规定」这类问法 —— 按 rag 处理，不调模型
        ③ 都没命中（比如「E1017」「那这个呢」这种极短或纯编号输入）—— 返回 None

        第 ③ 档是必须存在的：以前这里无条件 return "rag"，导致下面的模型分支
        永远不可达，IntentRecognizer（结构化输出 + 槽位抽取）成了死代码，
        事实槽也就永远拿不到 slots。
        """
        # 投诉优先：有情绪先安抚，别急着查资料
        for intent, words in self.BUSINESS_KEYWORDS.items():
            if any(w in q for w in words):
                return intent

        # 纯寒暄
        if len(q) <= 12 and any(w in q for w in self.CHITCHAT_WORDS):
            return "chitchat"

        # 带文档问答特征词
        if any(w in q for w in self.RAG_HINTS):
            return "rag"

        # 判不出来 —— 交给模型
        return None

    PRONOUNS = (
        "这个",
        "那个",
        "它",
        "上面",
        "刚才",
        "前面",
        "这条",
        "那条",
        "该",
        "此",
        "上述",
        "其",
    )

    def _need_rewrite(self, q: str) -> bool:
        """含指代词才需要改写；本身完整的问题直接拿去检索"""
        return any(w in q for w in self.PRONOUNS)

    # ---------- 实体抽取（喂事实槽，不额外调模型）----------
    # 抽取文档里会反复追问的精确标识：编号、条款号、错误码、文件名
    _ENTITY_PATTERNS = (
        # 制度编号：HR-2026-001 / FIN-2026-003 / SEC-2026-005
        ("doc_no", re.compile(r"\b[A-Z]{2,4}-\d{4}-\d{3,}\b")),
        # 错误码：E1017 / E1024
        ("error_code", re.compile(r"\bE\d{3,5}\b", re.IGNORECASE)),
        # 条款号：第3条 / 第 3.2 条
        ("clause", re.compile(r"第\s*\d+(?:\.\d+)?\s*条")),
        # 文件名：xxx.pdf / xxx手册
        ("doc_name", re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+\.pdf")),
    )

    def _extract_entities(self, q: str) -> dict[str, Any]:
        """从问题里抽精确标识写进事实槽 —— 多轮追问时不用再问用户一遍"""
        found: dict[str, Any] = {}
        for name, pattern in self._ENTITY_PATTERNS:
            m = pattern.search(q or "")
            if m:
                found[name] = m.group(0)
        return found

    def reset_session(self, session_id: str) -> None:
        """清空某个会话的记忆与文档绑定（前端「清空会话」按钮的落地逻辑）"""
        if session_id in self._memories:
            self._memories[session_id].clear_all()
            del self._memories[session_id]
        self._session_docs.pop(session_id, None)
