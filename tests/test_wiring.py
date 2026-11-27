"""接线测试：验证「零件真的接进主链路了」，而不是只存在于仓库里。

这一组用例的共同点：**不用模型、不用向量库、不用网络**，
全部用假对象替换掉外部依赖，只验证接线本身的正确性。
"""

import unittest

from langchain_core.documents import Document

from config.setting import QaConfig
from core.cache import build_cache
from indexing.clean import clean_pages


class TestCleanPages(unittest.TestCase):
    """逐页清洗：页眉靠整篇识别，页码 metadata 必须保住"""

    def _pages(self, n=4):
        pages = []
        for i in range(n):
            pages.append(
                Document(
                    page_content=(
                        f"某某公司内部规范\n第 {i + 1} 页 共 {n} 页\n"
                        f"正文第{i + 1}段：报销须在３０日内提交，逾期不予受理。"
                    ),
                    metadata={"source": "a.pdf", "page": i},
                )
            )
        return pages

    def test_header_footer_removed_but_pages_kept(self):
        pages, stats = clean_pages(self._pages())

        self.assertEqual(len(pages), 4)
        # 页码 metadata 不能丢，否则答案的「第 N 页」出处就废了
        self.assertEqual([p.metadata["page"] for p in pages], [0, 1, 2, 3])
        # 页眉（每页都有的公司名）应该被识别并去掉
        self.assertNotIn("某某公司内部规范", pages[0].page_content)
        # 页码行也不该留下
        self.assertNotIn("第 1 页", pages[0].page_content)
        self.assertGreaterEqual(stats["removed_header_footer"], 4)

    def test_fullwidth_digits_normalized(self):
        pages, _ = clean_pages(self._pages(1))
        # 全角３０ → 半角 30，否则「30 日」永远检索不到
        self.assertIn("30", pages[0].page_content)
        self.assertNotIn("３０", pages[0].page_content)

    def test_empty_input(self):
        pages, stats = clean_pages([])
        self.assertEqual(pages, [])
        self.assertEqual(stats["pages"], 0)


class TestHybridParallel(unittest.TestCase):
    """两路并行：谁的错都不该拖垮另一路"""

    def _retriever(self, dense, sparse):
        from querying.hybrid_retriever import HybridRetriever

        r = object.__new__(HybridRetriever)
        r._config = QaConfig()
        r.last_degraded = []
        r._dense = lambda q, k: dense() if callable(dense) else dense
        r._sparse = lambda q, k: sparse() if callable(sparse) else sparse
        return r

    def test_both_branches_fused(self):
        d = [Document(page_content="dense-1"), Document(page_content="dense-2")]
        s = [Document(page_content="sparse-1")]
        r = self._retriever(d, s)

        docs, scores = r.retrieve_with_scores("q", k=5)
        self.assertEqual(len(docs), 3)
        self.assertEqual(len(scores), 3)
        self.assertEqual(r.last_degraded, [])

    def test_sparse_failure_degrades_to_dense(self):
        def boom():
            raise RuntimeError("bm25 索引坏了")

        d = [Document(page_content="only-dense")]
        r = self._retriever(d, boom)

        docs, scores = r.retrieve_with_scores("q", k=5)
        # 只剩一路时必须还能返回结果，而不是整站不可用
        self.assertEqual(len(docs), 1)
        self.assertEqual(r.last_degraded, ["sparse"])
        self.assertTrue(all(s > 0 for s in scores))

    def test_both_failed_returns_empty(self):
        def boom():
            raise RuntimeError("down")

        r = self._retriever(boom, boom)
        docs, scores = r.retrieve_with_scores("q", k=5)
        self.assertEqual(docs, [])
        self.assertEqual(len(r.last_degraded), 2)


class TestPipelineWiring(unittest.TestCase):
    """pipeline 的缓存、预算、校验三处接线"""

    def _pipe(self):
        from querying.pipeline import RagPipeline

        pipe = object.__new__(RagPipeline)
        pipe._config = QaConfig()
        pipe._vectorstores = {}
        pipe._hybrid_cache = {}
        pipe._cache = build_cache(None)
        pipe._calls = {"rerank": 0}

        class FakeRewriter:
            def rewrite(self, q, h):
                return q

        class FakeRerank:
            def __init__(self, outer):
                self.outer = outer

            def rerank(self, query, recalled_docs, final_top_n=4):
                from querying.retrank import RetrievedEvidence

                self.outer._calls["rerank"] += 1
                evs = [
                    RetrievedEvidence(content=f"证据内容{i}", source="a.pdf", page=i, score=0.9)
                    for i in range(1, 3)
                ]
                return evs, ""

        pipe._rewriter = FakeRewriter()
        pipe._rerank_pipeline = FakeRerank(pipe)

        class FakeHybrid:
            def retrieve(self, q, k=None):
                return [Document(page_content="x")]

        pipe._get_hybrid = lambda vs, key: FakeHybrid()
        pipe._load_vectorstore = lambda h: object()
        return pipe

    def test_second_call_hits_cache(self):
        pipe = self._pipe()
        pipe.retrieve_multi(["h1"], "报销多久内提交", [])
        first = pipe._calls["rerank"]
        pipe.retrieve_multi(["h1"], "报销多久内提交", [])
        # 第二次不应再走精排（缓存直接返回）
        self.assertEqual(pipe._calls["rerank"], first)

    def test_corpus_fingerprint_invalidates_cache(self):
        pipe = self._pipe()
        pipe.retrieve_multi(["h1"], "报销多久内提交", [])
        before = pipe._calls["rerank"]
        # 换一份文档（语料指纹变化）→ 必须重新检索，不能拿旧结果
        pipe.retrieve_multi(["h2"], "报销多久内提交", [])
        self.assertGreater(pipe._calls["rerank"], before)

    def test_budget_drops_evidence_and_keeps_numbering(self):

        pipe = self._pipe()
        pipe._config.context_budget = 60  # 故意压得很小

        from querying.retrank import RetrievedEvidence

        evs = [
            RetrievedEvidence(content="第一段证据" * 5, source="a.pdf", page=0, score=0.9),
            RetrievedEvidence(content="第二段证据" * 5, source="a.pdf", page=1, score=0.8),
            RetrievedEvidence(content="第三段证据" * 5, source="a.pdf", page=2, score=0.7),
        ]
        kept, context, stats = pipe._fit_budget(evs)

        self.assertLess(len(kept), 3)  # 真的丢了
        self.assertEqual(stats["dropped"], 3 - len(kept))
        # 至少留一条：全丢等于告诉用户「没有」，而第一条可能就是答案
        self.assertGreaterEqual(len(kept), 1)
        # 编号连续，从 [证据1] 开始
        self.assertIn("[证据1", context)
        self.assertNotIn("[证据3", context)

    def test_verify_blocks_fabricated_code(self):
        from querying.retrank import RetrievedEvidence

        pipe = self._pipe()
        evs = [RetrievedEvidence(content="E1017 是任务调度超时", source="a.pdf", page=0, score=0.9)]

        # 答案里出现证据中没有的错误码 → 应被拦掉换成保守话术
        answer, report = pipe.verify("这是 E9999 错误，重启即可。", evs)
        self.assertFalse(report["ok"])
        self.assertNotIn("E9999", answer)

    def test_verify_passes_grounded_answer(self):
        from querying.retrank import RetrievedEvidence

        pipe = self._pipe()
        evs = [RetrievedEvidence(content="E1017 是任务调度超时", source="a.pdf", page=0, score=0.9)]

        answer, report = pipe.verify("E1017 是任务调度超时。[1]", evs)
        self.assertTrue(report["ok"])
        self.assertIn("E1017", answer)


class TestPostcheckNumbers(unittest.TestCase):
    """数字校验的两条边界：该拦的拦住，不该杀的别杀"""

    EVIDENCE = ["受限级文档的访问须留存审计日志。审计日志保留期限不少于三年。"]

    def _run(self, answer):
        from core.postcheck import postcheck

        return postcheck(answer, self.EVIDENCE, 1)

    def test_fabricated_number_blocked(self):
        r = self._run("保留期限为 99 年。[1]")
        self.assertFalse(r.ok)
        self.assertNotIn("99", r.answer)

    def test_chinese_and_arabic_equivalent(self):
        """证据写「三年」、答案写「3 年」，不能判成不一致（否则误杀正确答案）"""
        for a in ("保留期限为三年。[1]", "保留期限为 3 年。[1]"):
            self.assertTrue(self._run(a).ok, a)

    def test_year_not_treated_as_unit_number(self):
        """「2026 年」是年份，交给日期规则判，不能被数字规则误杀"""
        self.assertTrue(self._run("2026 年的审计要求见第 1 条。[1]").ok)

    def test_drifted_number_blocked(self):
        """证据说三年、答案说五年 —— 业务含义已变，必须拦"""
        self.assertFalse(self._run("审计日志要留 5 年。[1]").ok)


class TestRelaxedThreshold(unittest.TestCase):
    """两段式阈值：严格阈值没命中时，够格的放宽一次，不够格的就拒答。

    存在的理由：0.45 的严格阈值会砍掉「措辞和原文不同但答案确实在」的题。
    但放宽必须有底线，否则就是把无关内容喂给模型 —— 那比答不出更糟。
    """

    def _pipe(self, ranked):
        """ranked = [(doc, score)]，模拟精排过滤前的完整排序。"""
        from querying.pipeline import RagPipeline

        pipe = object.__new__(RagPipeline)
        pipe._config = QaConfig()
        pipe._config.min_score = 0.45
        pipe._config.min_score_relaxed = 0.35
        pipe._config.relaxed_top_n = 1
        pipe._config.enable_relaxed = True
        pipe._cache = build_cache(None)

        class FakeRerank:
            last_ranked = ranked

            @staticmethod
            def to_evidence(items):
                from querying.retrank import RetrievedEvidence

                evs = [
                    RetrievedEvidence(content=d.page_content, source="a.pdf", page=0, score=s)
                    for d, s in items
                ]
                return evs, ""

        pipe._rerank_pipeline = FakeRerank()
        return pipe

    def test_normal_hit_does_not_relax(self):
        pipe = self._pipe([])
        existing = ["已有一条"]
        out, relaxed = pipe._relax_if_empty(existing)
        self.assertEqual(out, existing)
        self.assertFalse(relaxed)

    def test_empty_but_high_enough_relaxes_to_top1(self):
        ranked = [(Document(page_content="候选一"), 0.43), (Document(page_content="候选二"), 0.41)]
        out, relaxed = self._pipe(ranked)._relax_if_empty([])
        self.assertTrue(relaxed)
        self.assertEqual(len(out), 1, "降级只取 top-1，不整批放回")
        self.assertEqual(out[0].content, "候选一")

    def test_empty_and_too_low_keeps_refusing(self):
        """最高分低于降级线 = 真无关，必须继续保持「没找到」。"""
        ranked = [(Document(page_content="噪声"), 0.2)]
        out, relaxed = self._pipe(ranked)._relax_if_empty([])
        self.assertEqual(out, [])
        self.assertFalse(relaxed)

    def test_switch_off_never_relaxes(self):
        ranked = [(Document(page_content="候选"), 0.43)]
        pipe = self._pipe(ranked)
        pipe._config.enable_relaxed = False
        out, relaxed = pipe._relax_if_empty([])
        self.assertEqual(out, [])
        self.assertFalse(relaxed)

    def test_no_candidates_at_all(self):
        out, relaxed = self._pipe([])._relax_if_empty([])
        self.assertEqual(out, [])
        self.assertFalse(relaxed)


if __name__ == "__main__":
    unittest.main()
