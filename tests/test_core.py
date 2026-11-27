# -*- coding: utf-8 -*-
"""
知渊核心逻辑回归测试（标准库 unittest，不需要 pytest，不联网、不调大模型）

覆盖 2026-11-10 这轮修复的每个点：
  1. RRF 加权融合 + 稳定键去重
  2. 记忆三层（历史/摘要/事实）顺序 + clear_all + 摘要落库
  3. 精排阈值过滤 / 空召回 / 接口异常降级 / 长度过滤
  4. 意图三级快判（含「判不出来返回 None」）+ 实体抽取
  5. 向量库空集合自愈

跑法（项目环境）：
  D:\\Anaconda3\\envs\\langchain\\python.exe -m unittest discover -s tests -v
或：
  D:\\Anaconda3\\envs\\langchain\\python.exe tests\\test_core.py
"""
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document


def _doc(text, page=None, source="a.pdf"):
    return Document(page_content=text, metadata={"source": source, "page": page})


class DbIsolated(unittest.TestCase):
    """
    每个用例一个独立的 SQLite 文件。

    记忆改成落库后，测试之间会互相污染（同一个 session_id 会读到上一轮的记录），
    所以统一在 setUp 里把 persistence 指到临时目录，tearDown 里清掉。
    """

    def setUp(self):
        import agent.persistence as p
        self._tmpdir = tempfile.mkdtemp()
        self._db = p.set_db_path(os.path.join(self._tmpdir, "memory.db"))

    def tearDown(self):
        import agent.persistence as p
        p.set_db_path(None)          # 复位到默认路径，避免影响别的用例
        shutil.rmtree(self._tmpdir, ignore_errors=True)


# ============================================================
# 1. RRF 融合
# ============================================================
class TestRRF(unittest.TestCase):
    "RRF 融合：加权、去重键、空输入"

    def test_same_chunk_merged_across_two_lists(self):
        from querying.rrf import rrf_scores
        d1 = _doc("年假五天", page=0)
        d2 = _doc("请假需提前三个工作日", page=1)
        pairs = rrf_scores([[d1, d2], [d2, d1]], k=60, weights=[0.7, 0.3])
        self.assertEqual(len(pairs), 2, "同一片段不应重复出现")

    def test_identical_text_different_page_not_merged(self):
        """修复点：以前用 page_content 当 key，会把不同页的同文片段吞掉"""
        from querying.rrf import rrf_scores
        a = _doc("本页无正文", page=0)
        b = _doc("本页无正文", page=5)
        pairs = rrf_scores([[a], [b]], k=60, weights=[0.7, 0.3])
        self.assertEqual(len(pairs), 2, "出处不同的片段不能被当成同一篇")

    def test_weights_change_ranking(self):
        from querying.rrf import rrf_scores
        x = _doc("X", page=0)
        y = _doc("Y", page=0)
        pairs = rrf_scores([[x, y], [y, x]], k=60, weights=[0.7, 0.3])
        self.assertEqual(pairs[0][0].page_content, "X", "dense 权重大时 dense 第一名应第一")
        pairs2 = rrf_scores([[x, y], [y, x]], k=60, weights=[0.3, 0.7])
        self.assertEqual(pairs2[0][0].page_content, "Y", "bm25 权重大时结果应反转")

    def test_empty_input(self):
        from querying.rrf import rrf_fuse, rrf_scores
        self.assertEqual(rrf_fuse([]), [])
        self.assertEqual(rrf_scores([]), [])


# ============================================================
# 2. 记忆三层
# ============================================================
class TestMemory(DbIsolated):
    "记忆三层：顺序、事实槽、clear_all、摘要阈值"

    def _mem(self, sid):
        from agent.memory import Memory
        return Memory(sid)

    def test_three_layer_order(self):
        """事实必须放最后（lost in the middle）—— 记忆设计的关键"""
        m = self._mem("t-order")
        m.add_user("我的订单号是 HR-2026-001", None)
        m.add_ai("已记录", None)
        m.update_facts({"doc_no": "HR-2026-001"})
        m._summary = "用户询问员工手册编号"

        ctx = m.build_context(limit=10)
        texts = [getattr(x, "content", "") for x in ctx]
        self.assertTrue(texts, "上下文不能为空")
        self.assertIn("HR-2026-001", texts[0], "最近消息应在最前")
        self.assertIn("本次会话摘要", texts[-1], "摘要应在最后一条")
        self.assertIn("当前已知事实", texts[-1], "事实必须在最后一条")

    def test_context_uses_system_message_not_ai(self):
        """修复点：摘要/事实以前用 AIMessage，语义是「AI 说过的话」，
        会让模型把系统注入当成自己的发言，并出现连续三条 ai。"""
        from langchain_core.messages import SystemMessage, AIMessage
        m = self._mem("t-role")
        m.add_user("年假有几天", None)
        m.add_ai("年假五天", None)
        m.update_facts({"doc_no": "HR-2026-001"})
        m._summary = "问过年假"

        ctx = m.build_context(limit=10)

        # 最后一条必须是 SystemMessage，且只有一条
        self.assertIsInstance(ctx[-1], SystemMessage)
        self.assertEqual(sum(isinstance(x, SystemMessage) for x in ctx), 1,
                         "只能有一条 SystemMessage")
        # 正常消息里不能出现连续两条 AI
        normal = [x for x in ctx if not isinstance(x, SystemMessage)]
        for a, b in zip(normal, normal[1:]):
            self.assertFalse(
                isinstance(a, AIMessage) and isinstance(b, AIMessage),
                "出现连续两条 AIMessage，对话结构被破坏",
            )

    def test_update_facts_skips_empty(self):
        m = self._mem("t-facts")
        m.update_facts({"order_id": "A1", "addr": "", "x": None})
        self.assertEqual(m.facts.get_all(), {"order_id": "A1"})

    def test_update_facts_tolerates_empty_dict(self):
        m = self._mem("t-facts2")
        m.update_facts({})
        m.update_facts(None)
        self.assertEqual(m.facts.get_all(), {})

    def test_clear_all(self):
        """修复点：这个方法以前不存在，导致 /api/reset 必然 500"""
        m = self._mem("t-clear")
        m.add_user("问个事", None)
        m.update_facts({"k": "v"})
        m._summary = "旧摘要"
        m.summary.set("旧摘要")

        m.clear_all()

        self.assertEqual(m.history.get_all_messages(), [])
        self.assertEqual(m.facts.get_all(), {})
        self.assertEqual(m._summary, "")
        self.assertEqual(m.summary.get(), "")
        self.assertEqual(m._unsummaryed, 0)

    def test_summary_triggers_at_threshold_and_persists(self):
        m = self._mem("t-sum")
        calls = {"n": 0}

        def fake_gen(old, new_msgs, llm):
            calls["n"] += 1
            return "摘要%d" % calls["n"]

        with patch.object(m.summary, "generate_incremental_summary", fake_gen):
            for i in range(9):
                m.add_user("q%d" % i, None)
            self.assertEqual(calls["n"], 0, "没到阈值不该压摘要")
            m.add_ai("第10条", None)          # 第 10 条触发
            self.assertEqual(calls["n"], 1, "到阈值必须压一次")

        self.assertEqual(m.summary.get(), "摘要1", "摘要要落到模块级 store")

    def test_summary_readable_from_new_instance(self):
        m1 = self._mem("t-persist")
        m1.summary.set("持久化摘要")
        m2 = self._mem("t-persist")
        self.assertEqual(m2._summary, "持久化摘要")


# ============================================================
# 3. 精排
# ============================================================
class TestRerank(unittest.TestCase):
    "精排：阈值过滤、空召回、接口异常降级、长度过滤"

    def _pipe(self):
        from config.setting import QaConfig
        from querying.retrank import RerankPipeline
        return RerankPipeline(QaConfig())

    def _docs(self, n=3):
        return [Document(page_content=("内容%d" % i) * 30,
                         metadata={"source": "a.pdf", "page": i}) for i in range(n)]

    @staticmethod
    def _resp(scores):
        class Resp:
            output = {"results": scores}
        return Resp()

    def test_below_min_score_dropped(self):
        """修复点：0.3 阈值以前定义了但从未生效"""
        import dashscope
        pipe = self._pipe()
        docs = self._docs(3)
        scores = [{"index": 0, "relevance_score": 0.9},
                  {"index": 1, "relevance_score": 0.338},
                  {"index": 2, "relevance_score": 0.1}]
        with patch.object(dashscope.TextReRank, "call",
                          staticmethod(lambda **kw: self._resp(scores))):
            ev, ctx = pipe.rerank("q", docs, final_top_n=3)
        self.assertEqual(len(ev), 2, "0.1 低于阈值必须被丢弃")
        self.assertNotIn(0.1, [e.score for e in ev])
        self.assertNotIn("内容2内容2", ctx)

    def test_all_below_threshold_returns_empty(self):
        import dashscope
        pipe = self._pipe()
        docs = self._docs(2)
        scores = [{"index": 0, "relevance_score": 0.05},
                  {"index": 1, "relevance_score": 0.08}]
        with patch.object(dashscope.TextReRank, "call",
                          staticmethod(lambda **kw: self._resp(scores))):
            ev, ctx = pipe.rerank("q", docs, final_top_n=3)
        self.assertEqual(ev, [])
        self.assertEqual(ctx, "", "全部不合格时应返回空，让上层说未找到")

    def test_empty_recall_skips_rerank_api(self):
        import dashscope
        pipe = self._pipe()
        calls = {"n": 0}

        def boom(**kw):
            calls["n"] += 1
            raise AssertionError("召回为空时不该调用精排")

        with patch.object(dashscope.TextReRank, "call", staticmethod(boom)):
            ev, ctx = pipe.rerank("q", [], final_top_n=3)
        self.assertEqual((ev, ctx), ([], ""))
        self.assertEqual(calls["n"], 0)

    def test_rerank_exception_degrades(self):
        import dashscope
        pipe = self._pipe()
        docs = self._docs(3)

        def boom(**kw):
            raise RuntimeError("network down")

        with patch.object(dashscope.TextReRank, "call", staticmethod(boom)):
            ev, ctx = pipe.rerank("q", docs, final_top_n=2)
        self.assertEqual(len(ev), 2, "降级后应按召回顺序返回")
        self.assertTrue(ctx, "降级后仍要能拼出上下文")

    def test_length_filter(self):
        pipe = self._pipe()
        short = _doc("太短", page=0)
        ok = _doc("合" * 50, page=1)
        kept = pipe._RerankPipeline__filter([short, ok])
        self.assertEqual(len(kept), 1)
        self.assertTrue(kept[0].page_content.startswith("合"))


# ============================================================
# 4. 意图快判 + 实体抽取（不构造 Agent，避免依赖 API key）
# ============================================================
class TestIntentRules(unittest.TestCase):
    "意图三级快判与实体抽取（不构造 Agent，避免依赖 API Key）"

    @staticmethod
    def _agent():
        from agent.orchestrator import Agent
        return object.__new__(Agent)      # 跳过 __init__，只用类级规则

    def test_chitchat(self):
        self.assertEqual(self._agent()._quick_intent("你好"), "chitchat")

    def test_complaint_wins(self):
        self.assertEqual(self._agent()._quick_intent("这个报销流程太麻烦了，我要投诉"), "complaint")

    def test_rag_by_hints(self):
        a = self._agent()
        self.assertEqual(a._quick_intent("年假有几天"), "rag")
        self.assertEqual(a._quick_intent("请假审批要走几级"), "rag")

    def test_ambiguous_returns_none(self):
        """修复点：以前无条件 return "rag"，模型兜底分支永远不可达"""
        a = self._agent()
        self.assertIsNone(a._quick_intent("E1017"))
        self.assertIsNone(a._quick_intent("ABC"))

    def test_need_rewrite(self):
        a = self._agent()
        self.assertTrue(a._need_rewrite("那这个呢"))
        self.assertFalse(a._need_rewrite("年假有几天"))

    def test_extract_entities(self):
        a = self._agent()
        self.assertEqual(a._extract_entities("HR-2026-001 是什么文件")["doc_no"], "HR-2026-001")
        self.assertEqual(a._extract_entities("E1017 是什么故障")["error_code"].upper(), "E1017")
        self.assertIn("第3条", a._extract_entities("员工手册第3条写的是什么")["clause"])
        self.assertEqual(a._extract_entities("今天天气怎么样"), {})


# ============================================================
# 5. 向量库空集合自愈
# ============================================================
class TestVectorStore(unittest.TestCase):
    "向量库：空集合自愈"

    def test_empty_collection_without_chunks_does_not_raise(self):
        """修复点：以前这里 raise ValueError，该文档后续每次提问都 500"""
        import indexing.vector_storage as vsmod
        from config.setting import QaConfig

        tmp = tempfile.mkdtemp()
        try:
            cfg = QaConfig()
            cfg.chroma_root_dir = os.path.join(tmp, "chroma")
            mgr = vsmod.VectorStoreManager(cfg, embeddings=None)

            class FakeVS:
                _collection = types.SimpleNamespace(count=lambda: 0)

            with patch.object(vsmod, "Chroma", lambda **kw: FakeVS()):
                got = mgr.load_or_build("deadbeef" * 4)      # 不传 chunks
            self.assertIsNotNone(got, "空集合应返回空库而不是抛异常")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# 6. 记忆持久化（重启不丢）
# ============================================================
class TestPersistence(DbIsolated):
    "记忆持久化：重连不丢、会话隔离、清空"

    def test_memory_survives_reconnect(self):
        """模拟「服务重启」：丢掉连接、重开同一个文件，三层记忆都还在"""
        from agent.memory import Memory
        import agent.persistence as p

        m = Memory("p-1")
        m.add_user("年假有几天", None)
        m.add_ai("年假五天", None)
        m.update_facts({"doc_no": "HR-2026-001"})
        m._summary = "问过年假"
        m.summary.set("问过年假")

        # 关连接、用同一个库文件重开 —— 等价于进程重启
        p.set_db_path(self._db.path)

        m2 = Memory("p-1")
        msgs = m2.history.get_all_messages()
        self.assertEqual([x.content for x in msgs], ["年假有几天", "年假五天"])
        self.assertEqual(m2.facts.get_all(), {"doc_no": "HR-2026-001"})
        self.assertEqual(m2.summary.get(), "问过年假")
        self.assertEqual(m2._summary, "问过年假", "新实例构造时应自动读回摘要")

    def test_different_sessions_are_isolated(self):
        from agent.memory import Memory
        a = Memory("iso-a")
        b = Memory("iso-b")
        a.add_user("A 的话", None)
        b.add_user("B 的话", None)
        a.update_facts({"k": "va"})
        b.update_facts({"k": "vb"})
        self.assertEqual([x.content for x in a.history.get_all_messages()], ["A 的话"])
        self.assertEqual([x.content for x in b.history.get_all_messages()], ["B 的话"])
        self.assertEqual(a.facts.get_all(), {"k": "va"})
        self.assertEqual(b.facts.get_all(), {"k": "vb"})

    def test_clear_all_wipes_db_rows(self):
        from agent.memory import Memory
        import agent.persistence as p
        m = Memory("wipe-1")
        m.add_user("x", None)
        m.update_facts({"k": "v"})
        m.summary.set("s")
        m.clear_all()
        self.assertEqual(p.db().get_messages("session：wipe-1"), [])
        self.assertEqual(p.db().get_facts("session：wipe-1"), {})
        self.assertEqual(p.db().get_summary("session：wipe-1"), "")


# ============================================================
# 7. 多文档检索
# ============================================================
class TestMultiDoc(unittest.TestCase):
    "多文档：绑定去重、跨文档融合、无证据短路"

    def _agent_stub(self):
        from agent.orchestrator import Agent
        a = object.__new__(Agent)          # 跳过 __init__（不构造模型）
        a._session_docs = {}
        return a

    def test_bind_accumulates_and_dedups(self):
        """一次会话可以绑多份文档，同一份重复上传只留一条"""
        a = self._agent_stub()
        a.bind_document("s1", "h1")
        a.bind_document("s1", "h2")
        a.bind_document("s1", "h1")        # 重复上传
        self.assertEqual(a.get_bound_documents("s1"), ["h1", "h2"])
        self.assertEqual(a.get_bound_documents("s2"), [])

    def test_unbind_keeps_session_entry_clean(self):
        a = self._agent_stub()
        a.bind_document("s1", "h1")
        a.unbind_documents("s1")
        self.assertEqual(a.get_bound_documents("s1"), [])

    def _pipe_stub(self):
        """不跑 __init__：避免构造 ChatTongyi（那需要 DASHSCOPE_API_KEY）"""
        from config.setting import QaConfig
        from querying.pipeline import RagPipeline
        pipe = object.__new__(RagPipeline)
        pipe._config = QaConfig()
        pipe._vectorstores = {}
        pipe._hybrid_cache = {}
        return pipe

    def test_query_multi_no_evidence_says_not_found_without_calling_model(self):
        """全部证据低于阈值时：直接回「未找到相关资料」，不调模型"""
        pipe = self._pipe_stub()

        class FakeRewriter:
            def rewrite(self, q, h):
                return q

        class FakeHybrid:
            def retrieve(self, q, k):
                return [Document(page_content="某段内容" * 20,
                                 metadata={"source": "a.pdf", "page": 0})]

        class BoomAnswer:
            def generate(self, *a, **kw):
                raise AssertionError("没有证据时不该调模型生成")

        pipe._rewriter = FakeRewriter()
        pipe._load_vectorstore = lambda h: object()
        pipe._get_hybrid = lambda vs, h: FakeHybrid()
        pipe._rerank_pipeline = types.SimpleNamespace(rerank=lambda **kw: ([], ""))
        pipe._answer_gen = BoomAnswer()

        out = pipe.query_multi(["h1", "h2"], "年终奖怎么算", [])
        self.assertEqual(out["answer"], "未找到相关资料。")
        self.assertEqual(out["doc_total"], 2)
        self.assertEqual(out["evidence"], [])

    def test_query_multi_fuses_across_documents(self):
        """两份文档各召回 1 条 → 融合后两条都在，且都进入精排输入"""
        pipe = self._pipe_stub()
        seen = {}

        class FakeRewriter:
            def rewrite(self, q, h):
                return q

        class FakeHybrid:
            def __init__(self, tag):
                self.tag = tag

            def retrieve(self, q, k):
                return [Document(page_content="内容" * 30 + self.tag,
                                 metadata={"source": self.tag + ".pdf", "page": 0})]

        def fake_rerank(query, recalled_docs, final_top_n=4):
            seen["n"] = len(recalled_docs)
            seen["srcs"] = sorted({d.metadata.get("source") for d in recalled_docs})
            return [], ""

        pipe._rewriter = FakeRewriter()
        pipe._load_vectorstore = lambda h: object()
        pipe._get_hybrid = lambda vs, h: FakeHybrid(h)
        pipe._rerank_pipeline = types.SimpleNamespace(rerank=fake_rerank)

        pipe.query_multi(["h1", "h2"], "问题", [])
        self.assertEqual(seen["n"], 2, "两份文档的召回结果都应进入精排")
        self.assertEqual(seen["srcs"], ["h1.pdf", "h2.pdf"])


# ============================================================
# 7. 接口层兜底（/api/chat 不再把异常抛成 500）
# ============================================================
class TestChatRoute(DbIsolated):
    "接口层：异常兜底、空问题 400、无 Key 健康检查"

    def test_agent_exception_returns_readable_reply(self):
        """修复点：以前模型超时/503 直接 500，用户只看到「服务异常」"""
        from fastapi.testclient import TestClient
        import api.main as appmod
        import api.routes.chat as chat_route

        class BoomAgent:
            def chat(self, session_id, question):
                raise RuntimeError("model timeout")

        with patch.object(chat_route, "get_agent", lambda: BoomAgent()):
            client = TestClient(appmod.app)
            r = client.post("/api/chat", json={"session_id": "s-x", "question": "年假有几天"})

        self.assertEqual(r.status_code, 200, "不应再抛 500")
        body = r.json()
        self.assertEqual(body["route"], "error")
        self.assertIn("稍后重试", body["answer"])
        self.assertEqual(body["evidence"], [])

    def test_empty_question_still_400(self):
        from fastapi.testclient import TestClient
        import api.main as appmod
        client = TestClient(appmod.app)
        r = client.post("/api/chat", json={"session_id": "s-x", "question": "   "})
        self.assertEqual(r.status_code, 400)

    def test_health_works_without_api_key(self):
        """修复点：deps 改懒加载后，没配 Key 也能起服务、能自检"""
        from fastapi.testclient import TestClient
        import api.main as appmod
        client = TestClient(appmod.app)
        r = client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")


# ============================================================
# 8. 流式输出
# ============================================================
class TestStreaming(DbIsolated):
    "流式：事件序列、固定答复、SSE 帧格式"

    def test_event_sequence(self):
        """chat_stream 的事件顺序：meta → evidence → delta* → done"""
        from agent.orchestrator import Agent
        from config.setting import QaConfig

        a = object.__new__(Agent)
        a._config = QaConfig()
        a.prepare = lambda sid, q: {
            "kind": "stream", "route": "chitchat", "intent": "chitchat",
            "confidence": 1.0, "evidence": [], "search_query": q,
            "context": "", "session_id": sid,
        }
        a._stream_chain = lambda plan, q, mem: iter(["你", "好"])
        a._remember = lambda sid, q, ans: None
        a._get_memory = lambda sid: None

        events = list(a.chat_stream("s", "你好"))
        self.assertEqual([e["event"] for e in events],
                         ["meta", "evidence", "delta", "delta", "done"])
        self.assertEqual(events[-1]["answer"], "你好")

    def test_fixed_answer_still_streams_once(self):
        """不需要模型的分支（未找到 / 未传文档）也要走同一套事件"""
        from agent.orchestrator import Agent
        from config.setting import QaConfig

        a = object.__new__(Agent)
        a._config = QaConfig()
        a.prepare = lambda sid, q: {
            "kind": "answer", "answer": "未找到相关资料。", "route": "rag",
            "intent": "rag", "confidence": 0.9, "evidence": [],
            "search_query": q, "context": "", "session_id": sid,
        }
        a._remember = lambda sid, q, ans: None

        events = list(a.chat_stream("s", "年终奖怎么算"))
        self.assertEqual(events[-1]["event"], "done")
        self.assertEqual(events[-1]["answer"], "未找到相关资料。")

    def test_sse_endpoint_emits_frames(self):
        from fastapi.testclient import TestClient
        import api.main as appmod
        import api.routes.chat as chat_route

        class FakeAgent:
            def chat_stream(self, sid, q):
                yield {"event": "meta", "intent": "rag", "route": "rag",
                       "confidence": 0.9, "search_query": q}
                yield {"event": "delta", "text": "年假"}
                yield {"event": "delta", "text": "五天"}
                yield {"event": "done", "answer": "年假五天"}

        with patch.object(chat_route, "get_agent", lambda: FakeAgent()):
            client = TestClient(appmod.app)
            r = client.post("/api/chat/stream",
                            json={"session_id": "s", "question": "年假有几天"})

        self.assertEqual(r.status_code, 200)
        self.assertIn("text/event-stream", r.headers["content-type"])
        body = r.text
        self.assertIn('"meta"', body)
        self.assertIn('"delta"', body)
        self.assertIn("年假五天", body, "中文不应被转义")
        self.assertIn("[DONE]", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
