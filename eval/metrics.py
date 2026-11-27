"""评测指标：全部是纯函数，不依赖项目其它模块，可单独单测。

设计原则：
  1. 每个指标只回答一个问题，不做「综合评分」——综合分会掩盖具体失败原因；
  2. 判定口径写死在文档字符串里，避免半年后不知道当初怎么算的；
  3. 结果指标（答案像不像）与过程指标（检索对不对、调用成没成）分开，
     因为前者会骗人：答案读起来通顺，证据可能是编的。
"""
from __future__ import annotations

import hashlib
import re
import statistics
from typing import Any, Dict, Iterable, List, Sequence, Tuple

# ---------- 文本归一化 ----------

_PUNCT = re.compile(r"[\s，。；：、！？,.;:!?（）()\[\]【】\"'“”‘’—\-…·]+")


def normalize(text: str) -> str:
    return _PUNCT.sub("", (text or "").lower())


def tokens(text: str) -> set:
    """中文按 bi-gram，英文/数字按整词 —— 与 BM25 切词口径保持一致。"""
    t = (text or "").lower()
    out = set(re.findall(r"[a-z0-9]+", t))
    for seg in re.findall(r"[\u4e00-\u9fa5]+", t):
        if len(seg) == 1:
            out.add(seg)
        else:
            out.update(seg[i:i + 2] for i in range(len(seg) - 1))
    return out


def overlap(a: str, b: str) -> float:
    """a 的内容有多少比例能在 b 里找到（按 token 计）。"""
    ta, tb = tokens(a), tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


# ---------- 维度 1：结果（能不能答对） ----------

def answer_pass(case: Dict[str, Any], answer: str) -> bool:
    """单条用例是否通过。

    通过条件（全部满足）：
      · must_contain 里的关键词全部出现（不区分大小写、忽略标点）
      · forbidden 里的词一个都不许出现（用于「不许编造」类用例）
      · should_refuse=True 时，答案必须落在拒答话术里
    """
    a = normalize(answer)
    if case.get("should_refuse"):
        return any(normalize(k) in a for k in case.get("refuse_markers", ["没有找到", "未收录", "无法回答", "材料中没有"]))
    if not all(normalize(k) in a for k in case.get("must_contain", [])):
        return False
    if any(normalize(k) in a for k in case.get("forbidden", [])):
        return False
    return True


def tsr(runs: Sequence[Dict[str, Any]]) -> float:
    """任务成功率 Task Success Rate = 首次即通过的用例数 / 用例数。

    口径：只取每个用例的**第 1 次**运行 —— 重复多次取「至少过一次」会虚高，
    稳定性问题应该由 stability(pass^n) 单独承担，不能混进 TSR。
    """
    if not runs:
        return 0.0
    first = [r for r in runs if int(r.get("run", 0)) == 0]
    if not first:
        first = runs
    return sum(1 for r in first if r["ok"]) / len(first)


# ---------- 维度 2：RAG 检索（过程对不对） ----------

def recall_at_k(retrieved: Sequence[str], expected: Sequence[str], k: int) -> float:
    """期望片段有多少被前 k 条召回覆盖。"""
    if not expected:
        return 1.0
    topk = set(retrieved[:k])
    return len(topk & set(expected)) / len(set(expected))


def mrr(retrieved: Sequence[str], expected: Sequence[str]) -> float:
    """第一个命中位置倒数的平均值 —— 衡量「排得够不够前」。"""
    exp = set(expected)
    for i, rid in enumerate(retrieved, 1):
        if rid in exp:
            return 1.0 / i
    return 0.0


def count_ok(n_returned: int, expect_range: Sequence[int]) -> bool:
    """返回条数是否落在期望区间 —— 用来抓「表面对」：条数够但内容是错的，
    或者条数不足导致答案缺证据。"""
    lo, hi = expect_range[0], expect_range[-1]
    return lo <= n_returned <= hi


# ---------- 维度 3：证据（不许胡言乱语） ----------

_SENT = re.compile(r"[^。！？!?\n]+[。！？!?]?")


def grounding(answer: str, evidence: Sequence[str], threshold: float = 0.5) -> float:
    """答案句子中有多少比例能被证据支撑（token 重叠 ≥ threshold）。

    口径说明：
      · 先按句切，长度 < 6 的句子（「好的」「是的」）不计入分母；
      · 每句与**全部证据拼接**比一次重叠，而不是逐条比 —— 一句话可能用到两段材料；
      · 这是「可复核性」的下界，不是语义蕴含判定，真实项目应叠加人工抽检。
    """
    if not evidence:
        return 0.0
    pool = " ".join(evidence)
    sents = [s.strip() for s in _SENT.findall(answer or "") if len(normalize(s)) >= 6]
    if not sents:
        return 0.0
    hit = sum(1 for s in sents if overlap(s, pool) >= threshold)
    return hit / len(sents)


def citation_coverage(answer: str, evidence: Sequence[str]) -> float:
    """答案里显式引用的片段占全部证据的比例（引用过度=堆材料，引用为 0=没出处）。"""
    if not evidence:
        return 0.0
    a = normalize(answer)
    used = sum(1 for e in evidence if normalize(e)[:30] and normalize(e)[:30] in a)
    return used / len(evidence)


# ---------- 维度 4：过程（真的成功了吗） ----------

def trace_ok(run: Dict[str, Any]) -> bool:
    """过程成功 = 无异常 + 无接口错误 + 无超时。

    单独一个指标的原因：后端接口 500 时，模型仍可能「猜」出一个像样的答案，
    结果指标会判通过 —— 这正是「表面对」的典型。
    """
    if run.get("error"):
        return False
    for step in run.get("trace", []):
        if step.get("status") in ("error", "timeout") or step.get("http_status", 200) >= 500:
            return False
    return True


# ---------- 维度 5：稳定性 ----------

def stability(runs: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """同一问题重复 N 次：

    · pass_at_n  —— 至少通过一次的比例（宽松，看「能不能做到」）
    · pass_all_n —— 每次都通过的比例（严格，看「稳不稳定」）
    · consistency—— 答案完全相同（归一化后哈希一致）的比例
    """
    by_case: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        by_case.setdefault(r["case_id"], []).append(r)
    if not by_case:
        return {"pass_at_n": 0.0, "pass_all_n": 0.0, "consistency": 0.0}
    at_least_one = sum(1 for v in by_case.values() if any(x["ok"] for x in v))
    all_pass = sum(1 for v in by_case.values() if all(x["ok"] for x in v))
    same = 0
    for v in by_case.values():
        hashes = {hashlib.md5(normalize(x.get("answer", "")).encode()).hexdigest() for x in v}
        if len(hashes) == 1:
            same += 1
    n = len(by_case)
    return {"pass_at_n": at_least_one / n, "pass_all_n": all_pass / n, "consistency": same / n}


# ---------- 维度 6：成本 ----------

def cost(runs: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    lat = sorted(float(r.get("latency_ms", 0)) for r in runs)
    tok = [float(r.get("tokens", 0)) for r in runs]

    def pct(p: float) -> float:
        if not lat:
            return 0.0
        idx = min(len(lat) - 1, int(round((len(lat) - 1) * p)))
        return lat[idx]

    return {
        "calls": len(runs),
        "p50_ms": pct(0.5),
        "p95_ms": pct(0.95),
        "avg_tokens": statistics.fmean(tok) if tok else 0.0,
        "total_tokens": sum(tok),
    }


# ---------- 维度 7：安全 ----------

def security(runs: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """越权/注入类用例：拦截率 = 正确拒答数 / 该类用例数。"""
    sec = [r for r in runs if r.get("suite") == "security"]
    if not sec:
        return {"cases": 0, "blocked_rate": 0.0}
    blocked = sum(1 for r in sec if r["ok"])
    return {"cases": len(sec), "blocked_rate": blocked / len(sec)}


def degrade_rate(runs: Sequence[Dict[str, Any]]) -> float:
    """故障套件的「降级正确率」：出错时有没有给出可读的降级话术，而不是崩掉或装没事。"""
    f = [r for r in runs if r.get("suite") == "fault"]
    if not f:
        return 0.0
    return sum(1 for r in f if r["ok"]) / len(f)


def summarize(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """把七维指标汇总成一个 dict，供报告使用。"""
    rag = [r for r in runs if r.get("expected_evidence")]
    normal = [r for r in runs if r.get("suite") != "fault"]
    return {
        "runs": len(runs),
        "tsr": tsr(normal) if normal else 0.0,
        "degrade": degrade_rate(runs),
        "process_ok": (sum(1 for r in runs if trace_ok(r)) / len(runs)) if runs else 0.0,
        "grounding": (
            statistics.fmean(r.get("grounding", 0.0) for r in rag) if rag else 0.0
        ),
        "recall_at_k": (
            statistics.fmean(r.get("recall", 0.0) for r in rag) if rag else 0.0
        ),
        "stability": stability(runs),
        "cost": cost(runs),
        "security": security(runs),
    }
