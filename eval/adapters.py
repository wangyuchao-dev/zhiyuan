"""两种适配器：把「被测系统」抽成一个 ask() 接口。

为什么要抽象：
  · mock  —— 不联网、不花钱，用来验证评测链路本身对不对（CI 和本地都能跑）；
  · live  —— 真跑项目的 Agent，需要 DASHSCOPE_API_KEY 与已入库的向量库。

关键约定：适配器返回的是**原始记录**（答案、证据、耗时、trace），
不在适配器里判对错 —— 判定全部交给 metrics，避免「谁跑谁判」导致口径漂移。
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence

# ---------- 通用 ----------


def _h(*parts: str) -> int:
    return int(hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:8], 16)


def _base_record(case: Dict[str, Any], answer: str, evidence: Sequence[str],
                 evidence_ids: Sequence[str], latency_ms: float, tokens: int,
                 trace: Optional[List[Dict[str, Any]]] = None,
                 error: str = "") -> Dict[str, Any]:
    return {
        "case_id": case["id"],
        "suite": case.get("suite", ""),
        "question": case.get("question", ""),
        "answer": answer,
        "evidence": list(evidence),
        "retrieved_ids": list(evidence_ids),
        "expected_evidence": case.get("expected_evidence", []),
        "k_range": case.get("k_range", [0, 99]),
        "latency_ms": latency_ms,
        "tokens": tokens,
        "trace": trace or [],
        "error": error,
        "needs_review": bool(case.get("needs_review")),
    }


# ---------- Mock ----------

class MockAdapter:
    """按 variant 模拟不同版本的通过率，用于验证评测框架。

    ⚠️ 这里的数字是**造出来的**，只用来证明「指标能分辨好坏」，
       不能当真实效果写进简历或答辩。真实数字请用 --adapter live 跑。
    """

    PROFILES = {
        "vector-baseline": {"static": 0.60, "dynamic": 0.45, "security": 0.85, "evidence": 0.55,
                            "latency": (5200, 9000), "tokens": (2600, 3600)},
        "hybrid-v1":       {"static": 0.90, "dynamic": 0.78, "security": 1.00, "evidence": 0.88,
                            "latency": (1900, 3200), "tokens": (1200, 1900)},
    }

    def __init__(self, variant: str = "hybrid-v1") -> None:
        if variant not in self.PROFILES:
            raise ValueError(f"未知 variant: {variant}，可选 {list(self.PROFILES)}")
        self.variant = variant
        self.p = self.PROFILES[variant]

    def ask(self, case: Dict[str, Any], turn: int = 0, session: str = "",
            run: int = 0) -> Dict[str, Any]:
        suite = case.get("suite", "static")
        seed = _h(self.variant, case["id"], str(turn), str(run))

        # 故障注入：模拟后端异常，答案照样给（「表面对」的典型）
        fault = case.get("fault")
        if fault:
            return self._fault(case, fault, seed)

        # 安全类：按 profile 决定拦住还是漏放
        if suite == "security":
            blocked = (seed % 1000) / 1000 < self.p["security"]
            answer = ("这个问题不在知识库范围内，我无法回答。" if blocked
                      else "好的，我这就去查，稍等。")
            return _base_record(case, answer, [], [], 800 + seed % 400, 300)

        # 正常问答：答案对不对 + 证据有没有（分开模拟，才能看出两个指标的区别）
        pass_rate = self.p["static"] if suite == "static" else self.p["dynamic"]
        ok = (seed % 1000) / 1000 < pass_rate
        ev_text = case.get("evidence_text", "")
        ev_id = (case.get("expected_evidence") or [""])[0]

        if ok:
            answer = f"根据《{ev_id.split('#')[0] or '知识库'}》：{ev_text}"
        else:
            answer = "根据制度要求，这种情况一般需要联系管理员处理，具体以平台配置为准。"

        # 一定概率「答案对但证据没带上」——结果指标过、证据指标挂，这是「表面对」的典型
        give_evidence = (seed % 100) < int(self.p["evidence"] * 100)
        evidence = [ev_text] if (ok and give_evidence) else []
        evidence_ids = [ev_id] if give_evidence else []

        lo, hi = self.p["latency"]
        latency = lo + seed % (hi - lo)
        tlo, thi = self.p["tokens"]
        tokens = tlo + seed % (thi - tlo)
        trace = [{"step": "retrieve", "status": "ok", "n": len(evidence_ids)},
                 {"step": "generate", "status": "ok"}]
        return _base_record(case, answer, evidence, evidence_ids, latency, tokens, trace)

    def _fault(self, case: Dict[str, Any], fault: str, seed: int) -> Dict[str, Any]:
        ev_text = case.get("evidence_text", "")
        if fault == "http_500":
            return _base_record(case, "抱歉，检索服务暂时不可用，请稍后再试。", [], [],
                                400, 120, [{"step": "retrieve", "status": "error", "http_status": 500}],
                                error="HTTP 500 from vector service")
        if fault == "timeout":
            return _base_record(case, "（无响应）", [], [], 15000, 0,
                                [{"step": "retrieve", "status": "timeout"}], error="read timeout after 15s")
        if fault == "slow_response":
            return _base_record(case, f"根据知识库：{ev_text}", [ev_text],
                                case.get("expected_evidence", []), 9800, 1500,
                                [{"step": "generate", "status": "slow"}])
        if fault == "empty_evidence":
            return _base_record(case, "材料里没有找到相关内容。", [], [], 1200, 600,
                                [{"step": "retrieve", "status": "ok", "n": 0}])
        # truncated
        return _base_record(case, "同名文档重新上传会覆", [ev_text],
                            case.get("expected_evidence", []), 2400, 900,
                            [{"step": "generate", "status": "truncated"}], error="answer truncated")


# ---------- Live ----------

class LiveAdapter:
    """真跑项目里的 Agent。

    懒加载：只有 first ask 时才 import 项目模块 —— 这样没装 langchain 也能跑 mock。
    每个 case 用独立 session，动态用例的多轮在同一 session 内完成。
    """

    def __init__(self, document_hash: Optional[str] = None) -> None:
        self._agent = None
        self._doc = document_hash

    def _ensure(self):
        if self._agent is None:
            import os
            import sys
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, root)
            from agent.orchestrator import Agent
            from config.setting import QaConfig
            self._agent = Agent(QaConfig())
        return self._agent

    def ask(self, case: Dict[str, Any], turn: int = 0, session: str = "",
            run: int = 0) -> Dict[str, Any]:
        agent = self._ensure()
        session = session or f"eval-{case['id']}"
        if self._doc:
            agent.bind_document(session, self._doc)

        question = case["question"] if turn == 0 else case.get("follow_ups", [])[turn - 1]
        t0 = time.perf_counter()
        error = ""
        try:
            out = agent.chat(session, question)
        except Exception as exc:  # 异常也要记录，不能中断整批
            out = {"answer": "", "evidence": []}
            error = f"{type(exc).__name__}: {exc}"
        latency = (time.perf_counter() - t0) * 1000

        evidence = out.get("evidence") or []
        ev_texts, ev_ids = [], []
        for e in evidence:
            if isinstance(e, dict):
                ev_texts.append(str(e.get("content") or e.get("text") or ""))
                ev_ids.append(str(e.get("id") or e.get("source") or ""))
            else:
                ev_texts.append(str(e))
                ev_ids.append(str(e))

        # token 粗估：中文按 1.6 字/token 折算，仅用于比较版本间相对变化
        tokens = int((len(question) + len(out.get("answer", ""))) / 1.6)
        return _base_record(case, out.get("answer", ""), ev_texts, ev_ids, latency, tokens,
                            [{"step": "chat", "status": "ok" if not error else "error"}], error)
