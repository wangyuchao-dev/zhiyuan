"""上下文预算：先决定「带什么」，再决定「带多长」。

问题：多轮之后 prompt 里塞了历史 + 事实 + 5 段证据 + 摘要，模型开始忽略中间部分，
      延迟和成本一起涨，但答对率反而掉。
做法：分级 —— 事实必带、证据按分数带并逐条截断、历史只带完整轮次且从新到旧。
原则：**先减少「带什么」，再考虑「压多短」**。摘要最贵（一次额外调用、且不可逆），
      放在最后才用。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_CJK = re.compile(r"[\u4e00-\u9fa5]")
_WORD = re.compile(r"[A-Za-z0-9_]+")


def estimate_tokens(text: str) -> int:
    """粗估 token：中文约 1 字 = 1 token，英文/数字按 1 词 ≈ 1.3 token，其它符号算 1。

    只用于「预算分配」，不用于计费。真实计费口径各模型不同，这里要的是相对准确。
    """
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    words = len(_WORD.findall(text))
    others = max(0, len(text) - cjk - sum(len(w) for w in _WORD.findall(text)))
    return int(cjk + words * 1.3 + others * 0.5)


def truncate_to_tokens(text: str, budget: int) -> Tuple[str, bool]:
    """按预算截断，尽量在句号处断开，避免切出半句话。"""
    if budget <= 0:
        return "", bool(text)
    if estimate_tokens(text) <= budget:
        return text, False
    # 逐字符累加（文本量小，简单实现足够；不要在这里搞复杂的分词对齐）
    out, used = [], 0
    for ch in text:
        cost = estimate_tokens(ch)
        if used + cost > budget:
            break
        out.append(ch)
        used += cost
    cut = "".join(out)
    for sep in ("。", "；", "！", "？", "\n"):
        pos = cut.rfind(sep)
        if pos >= max(1, int(len(cut) * 0.3)):          # 断开点不能太靠前，否则丢太多
            cut = cut[:pos + 1]
            break
    return cut, True


@dataclass
class PackedContext:
    facts: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    history: List[str] = field(default_factory=list)
    used_tokens: int = 0
    stats: Dict[str, Any] = field(default_factory=dict)

    def as_text(self) -> str:
        parts = []
        if self.facts:
            parts.append("【已知事实】\n" + "\n".join(f"- {f}" for f in self.facts))
        if self.evidence:
            parts.append("【参考资料】\n" + "\n\n".join(
                f"[{i}] {e}" for i, e in enumerate(self.evidence, 1)))
        if self.history:
            parts.append("【对话记录】\n" + "\n".join(self.history))
        return "\n\n".join(parts)


class ContextPacker:
    """按预算逐级装填。

    优先级：facts（必带，短）> evidence（按分数）> history（从新到旧，整轮带）。
    任何一级装不下就停，并在 stats 里记下丢了多少 —— 丢掉多少必须可见，
    否则「答案变差了」永远查不出是上下文被裁了。
    """

    def __init__(self, total_budget: int = 3000, evidence_item_budget: int = 600,
                 history_item_budget: int = 300, fact_budget: int = 120) -> None:
        self.total_budget = total_budget
        self.evidence_item_budget = evidence_item_budget
        self.history_item_budget = history_item_budget
        self.fact_budget = fact_budget
        self._facts: List[str] = []
        self._evidence: List[Tuple[float, str]] = []
        self._history: List[str] = []

    # ---- 装填 ----
    def add_facts(self, facts: Sequence[str]) -> "ContextPacker":
        self._facts.extend(f for f in facts if f)
        return self

    def add_evidence(self, items: Sequence[Tuple[float, str]]) -> "ContextPacker":
        """items = [(score, text)]，分数越高越先装。"""
        self._evidence.extend(items)
        return self

    def add_history(self, turns: Sequence[str]) -> "ContextPacker":
        """turns 按时间正序传入，内部反转为「从新到旧」装填。"""
        self._history.extend(reversed([t for t in turns if t]))
        return self

    # ---- 打包 ----
    def pack(self) -> PackedContext:
        out = PackedContext()
        left = self.total_budget
        dropped_ev, truncated_ev = 0, 0
        dropped_hist, truncated_hist = 0, 0
        truncated_facts = 0

        # 1) 事实：全带，但单条有上限（事实写太长本身就是问题）
        for f in self._facts:
            text, cut = truncate_to_tokens(f, self.fact_budget)
            cost = estimate_tokens(text)
            if cost > left:
                dropped_ev += 0
                break
            out.facts.append(text)
            left -= cost
            truncated_facts += int(cut)

        # 2) 证据：按分数降序，单条截断，装不下就停（宁可少一条，不塞残句）
        for score, text in sorted(self._evidence, key=lambda x: -x[0]):
            text, cut = truncate_to_tokens(text, self.evidence_item_budget)
            cost = estimate_tokens(text)
            if cost > left:
                dropped_ev += 1
                continue
            out.evidence.append(text)
            left -= cost
            truncated_ev += int(cut)

        # 3) 历史：整轮带，装不下就停（半轮历史比没有更糟）
        for turn in self._history:
            text, cut = truncate_to_tokens(turn, self.history_item_budget)
            cost = estimate_tokens(text)
            if cost > left:
                dropped_hist += 1
                continue
            out.history.append(text)
            left -= cost
            truncated_hist += int(cut)

        out.history.reverse()               # 还原成时间正序，模型读着顺
        out.used_tokens = self.total_budget - left
        out.stats = {
            "budget": self.total_budget,
            "used": out.used_tokens,
            "facts": len(out.facts),
            "evidence_in": len(out.evidence),
            "evidence_dropped": dropped_ev,
            "evidence_truncated": truncated_ev,
            "history_in": len(out.history),
            "history_dropped": dropped_hist,
            "history_truncated": truncated_hist,
            "facts_truncated": truncated_facts,
            "utilization": round(out.used_tokens / self.total_budget, 3) if self.total_budget else 0.0,
        }
        return out


def should_summarize(packed: PackedContext, history_total: int) -> bool:
    """要不要触发摘要：历史被丢掉超过一半才值得花一次模型调用。"""
    kept = packed.stats.get("history_in", 0)
    if history_total <= 0:
        return False
    return kept / history_total < 0.5
