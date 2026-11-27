"""数据链路测试：清洗、缓存、后置校验，以及**大文本**压测。

最后那一组是回答「你真的用大文本测过父子分块吗」的：
生成一份约 60 万字的模拟手册（50 页、每页带页眉页脚与页码、混入全角字符与重复段），
走完整条 clean → 父子分块 → 上下文预算，记录耗时与规模并断言不退化。
"""
import os
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.cache import Cache, MemoryCacheBackend, build_cache, corpus_fingerprint  # noqa: E402
from core.context_budget import ContextPacker                                      # noqa: E402
from core.postcheck import Severity, postcheck, summarize                           # noqa: E402
from indexing.clean import clean_text, find_repeated_lines, quality_gate            # noqa: E402
from indexing.parent_child import ParentChildIndex                                  # noqa: E402


class TestClean(unittest.TestCase):
    RAW = (
        "智云平台运维手册（内部资料）\n- 1 -\n第一章 概述\n正文第一段，说明平台用途。\n\f"
        "智云平台运维手册（内部资料）\n- 2 -\n第二章 故障\n出现 E1017 时请重启任务节点。\n\f"
        "智云平台运维手册（内部资料）\n- 3 -\n第三章 附录\n目录行 ........ 12\n"
    )

    def test_header_footer_detected(self):
        repeated = find_repeated_lines(self.RAW, min_pages=2)
        self.assertTrue(any("智云平台" in r for r in repeated))

    def test_clean_removes_noise(self):
        r = clean_text(self.RAW)
        self.assertNotIn("智云平台运维手册（内部资料）", r.text)
        self.assertGreaterEqual(r.stats["removed_header_footer"], 3)
        self.assertGreaterEqual(r.stats["removed_page_number"], 3)
        self.assertEqual(r.stats["removed_toc"], 1)
        self.assertIn("E1017", r.text)

    def test_fullwidth_folded(self):
        r = clean_text("单个文档不得超过５０ＭＢ。")
        self.assertIn("50MB", r.text)

    def test_duplicate_paragraph_removed(self):
        r = clean_text("重复段。\n\n重复段。\n\n另一段。")
        self.assertEqual(r.stats["removed_duplicate_para"], 1)

    def test_quality_gate_rejects_scanned(self):
        ok, reason = quality_gate(clean_text("短"))
        self.assertFalse(ok)
        self.assertIn("扫描件", reason)


class TestCache(unittest.TestCase):
    def setUp(self):
        self.c = Cache(MemoryCacheBackend(), ttl_default=60)

    def test_hit_and_stats(self):
        key = self.c.build_key("retr", "文档多大", corpus_fp="abc")
        self.c.set_json(key, ["a", "b"])
        got, hit = self.c.get_or_set(key, lambda: ["x"])
        self.assertTrue(hit)
        self.assertEqual(got, ["a", "b"])
        self.assertEqual(self.c.stats.hit, 1)

    def test_empty_not_cached(self):
        key = self.c.build_key("retr", "空结果")
        self.c.set_json(key, [])
        self.assertIsNone(self.c.get_json(key))

    def test_key_changes_with_corpus(self):
        k1 = self.c.build_key("retr", "q", corpus_fp=corpus_fingerprint(["h1"]))
        k2 = self.c.build_key("retr", "q", corpus_fp=corpus_fingerprint(["h1", "h2"]))
        self.assertNotEqual(k1, k2)                     # 加文档后缓存自动失效

    def test_ttl_expiry(self):
        b = MemoryCacheBackend()
        b.set("k", "v", ttl=1)
        self.assertEqual(b.get("k"), "v")
        b._data["k"] = (time.time() - 1, "v")           # 手动过期，不 sleep
        self.assertIsNone(b.get("k"))

    def test_degrade_when_redis_unavailable(self):
        c = build_cache("redis://127.0.0.1:6399/0")     # 故意给错的端口
        self.assertIsInstance(c.backend, MemoryCacheBackend)


class TestPostcheck(unittest.TestCase):
    EV = ["错误码 E1017：任务调度超时。单个文档不得超过 50 MB。维护窗口 2026-11-20。"]

    def test_clean_answer_passes(self):
        r = postcheck("E1017 是任务调度超时，文档上限 50 MB。", self.EV)
        self.assertTrue(r.ok)
        self.assertIs(r.action, Severity.PASS)

    def test_fabricated_code_blocked(self):
        r = postcheck("E1017 表示磁盘满，E1035 表示网络中断。", self.EV)
        self.assertFalse(r.ok)
        self.assertIs(r.action, Severity.BLOCK)
        self.assertIn("E1035", [v.detail for v in r.violations])
        self.assertNotIn("E1035", r.answer)             # 换成保守话术

    def test_number_drift_blocked(self):
        r = postcheck("单个文档不得超过 500 MB。", self.EV)
        self.assertIs(r.action, Severity.BLOCK)

    def test_date_drift_blocked(self):
        r = postcheck("维护窗口是 2026-12-20。", self.EV)
        self.assertIs(r.action, Severity.BLOCK)

    def test_citation_out_of_range_annotated(self):
        r = postcheck("E1017 是任务调度超时[1]，另有说明[7]。", self.EV)
        self.assertIs(r.action, Severity.ANNOTATE)
        self.assertIn("[1]", r.answer)
        self.assertNotIn("[7]", r.answer)               # 越界编号被移除

    def test_summary(self):
        reports = [postcheck("E1035 是网络中断。", self.EV), postcheck("正常。", self.EV)]
        s = summarize(reports)
        self.assertEqual(s["calls"], 2)
        self.assertEqual(s["blocked"], 1)


def make_big_doc(pages: int = 800, paras_per_page: int = 12) -> str:
    """造一份「像真的」的大文档：页眉、页码、全角字符、重复段都齐。"""
    parts = []
    for p in range(1, pages + 1):
        lines = ["智云平台运维手册（内部资料）", f"- {p} -", f"# 第 {p} 章 故障处置"]
        for i in range(paras_per_page):
            lines.append(
                f"第 {p}-{i} 条：出现 E{1000 + i} 时，请在 5 分钟内重启任务节点，"
                f"并检查磁盘占用是否超过８０％，单个日志文件不得超过５０ＭＢ。"
            )
        if p % 5 == 0:                                   # 每 5 页插一段重复内容
            lines.append("本章修订记录：2026-11-20 修订。\n\n本章修订记录：2026-11-20 修订。")
        parts.append("\n".join(lines))
    return "\f".join(parts)


class TestLargeDocument(unittest.TestCase):
    """大文本压测：清洗 + 父子分块 + 上下文预算，全链路。"""

    @classmethod
    def setUpClass(cls):
        cls.raw = make_big_doc()
        cls.t0 = time.perf_counter()
        cls.clean = clean_text(cls.raw)
        t_clean = time.perf_counter() - cls.t0

        cls.t1 = time.perf_counter()
        cls.index = ParentChildIndex(child_size=220, child_overlap=40, parent_max=900)
        cls.doc = cls.index.add("manual", cls.clean.text)
        t_index = time.perf_counter() - cls.t1

        print(f"\n[大文本] 原文 {len(cls.raw):,} 字 → 清洗后 {len(cls.clean.text):,} 字"
              f"（{t_clean * 1000:.0f}ms）｜父块 {len(cls.doc.parents)} / 子块 {len(cls.doc.children)}"
              f"（{t_index * 1000:.0f}ms）")

    def test_scale_is_large_enough(self):
        self.assertGreater(len(self.raw), 500_000)        # 至少 50 万字，才算大文本
        self.assertGreater(len(self.doc.children), 300)

    def test_noise_really_removed(self):
        s = self.clean.stats
        self.assertGreaterEqual(s["removed_header_footer"], 700)
        self.assertGreaterEqual(s["removed_page_number"], 700)
        self.assertNotIn("（内部资料）", self.clean.text)
        self.assertIn("50MB", self.clean.text)            # 全角已折半角

    def test_children_cover_whole_document(self):
        """不丢内容：子块拼起来应覆盖绝大多数正文（允许 overlap 带来的重复）。"""
        joined = "".join(c.text for c in self.doc.children)
        self.assertGreater(len(joined), len(self.clean.text) * 0.9)

    def test_expand_returns_parents_not_fragments(self):
        hits = sorted(self.doc.children, key=lambda c: c.order)[:8]
        for i, h in enumerate(hits):
            h.score = 1.0 - i * 0.05
        parents = self.index.expand(hits, max_parents=3)
        self.assertLessEqual(len(parents), 3)
        self.assertTrue(all(len(p.text.strip()) > 10 for p in parents))

    def test_context_budget_holds_under_load(self):
        """100 条证据只装得下几条 —— 预算必须真的生效，且统计要说清丢了多少。"""
        packer = ContextPacker(total_budget=1500)
        packer.add_facts(["部门=风控部"])
        packer.add_evidence([(1 - i / 200, c.text) for i, c in enumerate(self.doc.children[:100])])
        packed = packer.pack()
        self.assertLessEqual(packed.used_tokens, 1500)
        self.assertGreater(packed.stats["evidence_dropped"], 80)

    def test_performance_budget(self):
        """再跑一次，确认耗时稳定（不是首跑偶然快）。"""
        t0 = time.perf_counter()
        idx = ParentChildIndex()
        idx.add("manual", self.clean.text)
        elapsed = time.perf_counter() - t0
        print(f"[大文本] 重复分块 {elapsed * 1000:.0f}ms")
        self.assertLess(elapsed, 10.0)                    # 10 秒上限，宽松防机器抖动


if __name__ == "__main__":
    unittest.main(verbosity=2)
