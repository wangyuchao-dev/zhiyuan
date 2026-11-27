"""新增基础设施的离线单测。

覆盖：错误分类与重试、上下文预算、父子分块、并行召回与 RRF、
工具注册表与 MCP 互转、SKILL 懒加载、显式计划执行、记忆策略、版本对比。

约束：**不联网、不花钱、不依赖 langchain / 模型 / 数据库**，
全部用假对象与内存字典，`python -m unittest` 直接能跑。
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.memory_policy import Item, Kind, MemoryPolicy  # noqa: E402
from agent.plan import build_plan, execute, summarize_trace  # noqa: E402
from core.context_budget import ContextPacker, estimate_tokens, truncate_to_tokens  # noqa: E402
from core.errors import ErrorKind, ToolError, classify, next_delay, to_model_message  # noqa: E402
from eval.compare import compare  # noqa: E402
from indexing.parent_child import ParentChildIndex, split_parents  # noqa: E402
from querying.parallel import merge_with_fallback, rrf_merge, run_parallel  # noqa: E402
from skills.registry import SkillRegistry, parse_front_matter  # noqa: E402
from tools.mcp_adapter import filter_tools, from_mcp_tools  # noqa: E402
from tools.registry import ToolRegistry  # noqa: E402


class TestErrors(unittest.TestCase):
    def test_classify_http(self):
        self.assertIs(classify(status=503).kind, ErrorKind.TRANSIENT)
        self.assertIs(classify(status=403).kind, ErrorKind.PERMISSION)
        self.assertIs(classify(status=400).kind, ErrorKind.PARAM)
        self.assertIs(classify(empty=True).kind, ErrorKind.EMPTY)

    def test_permission_not_retryable(self):
        err = classify(status=403)
        self.assertFalse(err.retryable)
        self.assertIsNone(next_delay(err, 1))

    def test_backoff_sequence_and_cap(self):
        err = classify(status=500)
        self.assertEqual(next_delay(err, 1), 0.5)
        self.assertEqual(next_delay(err, 2), 1.0)
        self.assertIsNone(next_delay(err, 4))  # 超过序列长度即放弃

    def test_param_retry_only_once(self):
        err = classify(status=400)
        self.assertIsNotNone(next_delay(err, 1))
        self.assertIsNone(next_delay(err, 2))

    def test_model_message_actionable(self):
        msg = to_model_message(classify(status=500), attempts=2)
        self.assertIn("不要重复调用", msg)
        self.assertIn("2", msg)


class TestContextBudget(unittest.TestCase):
    def test_estimate(self):
        self.assertGreater(estimate_tokens("你好世界"), 0)
        self.assertGreater(estimate_tokens("hello world 123"), 0)

    def test_truncate_prefers_sentence(self):
        text = "短句。第二句话很长很长很长。"
        cut, truncated = truncate_to_tokens(text, 8)
        self.assertTrue(truncated)
        self.assertTrue(cut.endswith("。"))

    def test_pack_priority_and_stats(self):
        p = ContextPacker(total_budget=80, evidence_item_budget=40, history_item_budget=20)
        p.add_facts(["部门=风控部"])
        p.add_evidence([(0.9, "证据A" * 20), (0.8, "证据B" * 20), (0.1, "证据C" * 20)])
        p.add_history(["第一轮问答" * 10, "第二轮问答" * 10])
        packed = p.pack()
        self.assertTrue(packed.facts)  # 事实必带
        self.assertLessEqual(packed.used_tokens, 80)
        self.assertGreaterEqual(packed.stats["evidence_dropped"], 1)
        self.assertGreater(packed.stats["utilization"], 0)


class TestParentChild(unittest.TestCase):
    DOC = "# 一、文档管理\n\n上传后 5 分钟内可检索。超过 50MB 需要拆分。\n\n# 二、权限\n\n仅管理员可删除文档。"

    def test_split_parents(self):
        parents = split_parents("d1", self.DOC)
        self.assertEqual(len(parents), 2)
        self.assertIn("文档管理", parents[0].title)

    def test_children_point_to_parent(self):
        idx = ParentChildIndex(child_size=30, child_overlap=5)
        doc = idx.add("d1", self.DOC)
        self.assertTrue(doc.children)
        self.assertTrue(all(c.parent_id for c in doc.children))
        self.assertTrue(all(c.parent_id in {p.id for p in doc.parents} for c in doc.children))

    def test_expand_dedupes_by_parent(self):
        idx = ParentChildIndex(child_size=20, child_overlap=4)
        doc = idx.add("d1", self.DOC)
        hits = doc.children[:3]
        for i, h in enumerate(hits):
            h.score = 0.9 - i * 0.1
        parents = idx.expand(hits, max_parents=2)
        self.assertLessEqual(len(parents), 2)
        self.assertEqual(len({p.id for p in parents}), len(parents))


class TestParallel(unittest.TestCase):
    def test_one_branch_fails_other_still_works(self):
        def bad():
            raise RuntimeError("boom")

        res = run_parallel({"vec": lambda: ["a", "b"], "bm25": bad}, timeout=1.0)
        self.assertTrue(res["vec"].ok)
        self.assertFalse(res["bm25"].ok)
        merged, failed = merge_with_fallback(res, {"vec": 0.7, "bm25": 0.3})
        self.assertEqual([m[0] for m in merged][:2], ["a", "b"])
        self.assertEqual(failed, ["bm25"])

    def test_rrf_prefers_agreement(self):
        ranked = {"vec": ["x", "a", "b"], "bm25": ["a", "x", "c"]}
        merged = dict(rrf_merge(ranked, {"vec": 0.7, "bm25": 0.3}))
        self.assertGreater(merged["a"] + merged["x"], merged["b"] + merged["c"])

    def test_single_branch_no_weighting(self):
        res = run_parallel({"vec": lambda: ["p", "q"]}, timeout=1.0)
        merged, failed = merge_with_fallback(res)
        self.assertEqual([m[0] for m in merged], ["p", "q"])
        self.assertEqual(failed, [])


class TestTools(unittest.TestCase):
    def setUp(self):
        self.reg = ToolRegistry()
        self.reg.register_simple(
            "lookup",
            "查文档",
            {"code": "string"},
            lambda code: {"code": code, "meaning": "调度超时"},
        )

    def test_specs_shape(self):
        spec = self.reg.specs()[0]
        self.assertEqual(spec["type"], "function")
        self.assertEqual(spec["function"]["name"], "lookup")
        self.assertIn("parameters", spec["function"])

    def test_missing_arg(self):
        r = self.reg.call("lookup", {})
        self.assertFalse(r.ok)
        self.assertEqual(r.error.code, "MISSING_ARG")

    def test_unknown_arg(self):
        r = self.reg.call("lookup", {"code": "E1", "extra": 1})
        self.assertFalse(r.ok)
        self.assertEqual(r.error.code, "UNKNOWN_ARG")

    def test_type_mismatch(self):
        r = self.reg.call("lookup", {"code": 123})
        self.assertFalse(r.ok)
        self.assertEqual(r.error.code, "TYPE_MISMATCH")

    def test_permission_denied(self):
        self.reg.register_simple(
            "drop", "删表", {"table": "string"}, lambda table: "ok", requires_perm=("admin",)
        )
        r = self.reg.call("drop", {"table": "chat_msg"})
        self.assertFalse(r.ok)
        self.assertIs(r.error.kind, ErrorKind.PERMISSION)
        r2 = self.reg.call("drop", {"table": "chat_msg"}, granted_perms=("admin",))
        self.assertTrue(r2.ok)

    def test_handler_exception_becomes_tool_error(self):
        self.reg.register_simple(
            "bad", "会炸", {"x": "string"}, lambda x: (_ for _ in ()).throw(TimeoutError("slow"))
        )
        r = self.reg.call("bad", {"x": "1"})
        self.assertFalse(r.ok)
        self.assertIs(r.error.kind, ErrorKind.TRANSIENT)


class TestMcp(unittest.TestCase):
    def test_from_mcp(self):
        specs = from_mcp_tools(
            [
                {
                    "name": "amap_geo",
                    "description": "地理编码",
                    "inputSchema": {"type": "object", "properties": {"addr": {"type": "string"}}},
                }
            ]
        )
        self.assertEqual(specs[0].name, "amap_geo")
        self.assertEqual(specs[0].parameters["type"], "object")

    def test_filter_prefix_and_cap(self):
        specs = from_mcp_tools(
            [{"name": f"amap_{i}", "inputSchema": {"type": "object"}} for i in range(20)]
            + [{"name": "junk_open_app", "inputSchema": {"type": "object"}}]
        )
        kept = filter_tools(specs, allow_prefix=("amap_",), deny_prefix=("junk_",), max_tools=5)
        self.assertEqual(len(kept), 5)
        self.assertTrue(all(s.name.startswith("amap_") for s in kept))


class TestSkills(unittest.TestCase):
    def test_front_matter(self):
        meta, body = parse_front_matter("---\nname: a\nsummary: b\n---\n正文")
        self.assertEqual(meta["name"], "a")
        self.assertEqual(body.strip(), "正文")

    def test_index_and_lazy_body(self):
        reg = SkillRegistry(os.path.join(ROOT, "skills"))
        metas = reg.index()
        self.assertTrue(metas)
        self.assertEqual(reg.stats()["loaded"], 0)  # 只读目录，不读正文
        out = reg.render("E1017 这个错误码怎么处理")
        self.assertIn("错误码", out)
        self.assertEqual(reg.stats()["loaded"], 1)  # 命中后才加载

    def test_no_match_returns_empty(self):
        reg = SkillRegistry(os.path.join(ROOT, "skills"))
        reg.index()
        self.assertEqual(reg.render("今天天气怎么样"), "")


class TestPlan(unittest.TestCase):
    def test_chitchat_skips_retrieval(self):
        plan = build_plan("chitchat", need_rewrite=False, has_documents=True)
        self.assertNotIn("retrieve", plan.step_names())

    def test_no_doc_short_circuit(self):
        plan = build_plan("rag", need_rewrite=False, has_documents=False)
        self.assertEqual(plan.route, "error")

    def test_retry_then_success(self):
        calls = {"n": 0}

        def flaky(ctx):
            calls["n"] += 1
            if calls["n"] < 2:
                raise TimeoutError("slow")
            return ["doc#1"]

        res = execute(
            build_plan("rag", False, True),
            {"retrieve": flaky, "compose": lambda c: "答案"},
            sleep=lambda s: None,
        )
        self.assertTrue(res.ok)
        self.assertEqual(calls["n"], 2)
        self.assertGreaterEqual(summarize_trace(res.trace)["retries"], 1)

    def test_optional_step_degrades(self):
        def broken(ctx):
            raise RuntimeError("rewrite failed")

        res = execute(
            build_plan("rag", need_rewrite=True, has_documents=True),
            {"rewrite": broken, "retrieve": lambda c: ["d"], "compose": lambda c: "a"},
            sleep=lambda s: None,
        )
        self.assertTrue(res.ok)  # 改写失败不影响主链路
        self.assertTrue(res.outputs.get("rewrite_degraded"))
        self.assertIn("rewrite", summarize_trace(res.trace)["degraded"])

    def test_required_step_stops(self):
        res = execute(
            build_plan("rag", False, True),
            {
                "retrieve": lambda c: (_ for _ in ()).throw(RuntimeError("x")),
                "compose": lambda c: "a",
            },
            sleep=lambda s: None,
        )
        self.assertFalse(res.ok)
        self.assertEqual(res.failed_step, "retrieve")

    def test_permission_error_not_retried(self):
        calls = {"n": 0}

        def deny(ctx):
            calls["n"] += 1
            raise ToolError(ErrorKind.PERMISSION, "NO_PERM", "无权")

        res = execute(
            build_plan("rag", False, True),
            {"retrieve": deny, "compose": lambda c: "a"},
            sleep=lambda s: None,
        )
        self.assertFalse(res.ok)
        self.assertEqual(calls["n"], 1)


class TestMemoryPolicy(unittest.TestCase):
    def setUp(self):
        self.p = MemoryPolicy()
        self.store = {}

    def test_reject_question_and_filler(self):
        self.assertFalse(self.p.should_write(Kind.FACT, "文档多大？")[0])
        self.assertFalse(self.p.should_write(Kind.FACT, "谢谢")[0])
        self.assertTrue(self.p.should_write(Kind.FACT, "部门是风控部")[0])

    def test_version_increments_and_idempotent(self):
        item, wrote = self.p.upsert(self.store, Item("dept", Kind.FACT, "风控部"))
        self.assertTrue(wrote)
        self.assertEqual(item.version, 1)
        _, wrote2 = self.p.upsert(self.store, Item("dept", Kind.FACT, "风控部"))
        self.assertFalse(wrote2)  # 值没变不写
        item3, wrote3 = self.p.upsert(self.store, Item("dept", Kind.FACT, "合规部"))
        self.assertTrue(wrote3)
        self.assertEqual(item3.version, 2)

    def test_fact_lookup_is_exact(self):
        self.p.upsert(self.store, Item("dept", Kind.FACT, "风控部"))
        self.assertIsNotNone(self.p.get(self.store, "dept"))
        self.assertIsNone(self.p.get(self.store, "部门"))  # 不做相似匹配

    def test_preference_search_is_strict(self):
        self.p.upsert(self.store, Item("style", Kind.PREFERENCE, "回答要简短，用表格"))
        hit = self.p.search(self.store, "回答简短一点")
        self.assertEqual(len(hit), 1)
        self.assertEqual(self.p.search(self.store, "帮我写一封邮件"), [])

    def test_ephemeral_gets_ttl_and_expires(self):
        item = Item("doc", Kind.EPHEMERAL, "刚才那份文档")
        self.p.upsert(self.store, item)
        self.assertIsNotNone(self.store["doc"].ttl_s)
        self.store["doc"].updated_at -= 10_000
        self.assertEqual(self.p.purge_expired(self.store), 1)


class TestCompare(unittest.TestCase):
    def _payload(self, rows):
        return {"records": [{"case_id": cid, "ok": ok, "run": 0} for cid, ok in rows]}

    def test_repair_and_regression(self):
        prev = self._payload([("a", False), ("b", True), ("c", True)])
        curr = self._payload([("a", True), ("b", False), ("c", True)])
        c = compare(prev, curr)
        self.assertEqual(c["fixed"], ["a"])
        self.assertEqual(c["regressions"], ["b"])
        self.assertAlmostEqual(c["repair_rate"], 1.0)
        self.assertAlmostEqual(c["regression_rate"], 0.5)

    def test_no_failure_means_none_not_zero(self):
        prev = self._payload([("a", True)])
        curr = self._payload([("a", True)])
        self.assertIsNone(compare(prev, curr)["repair_rate"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
