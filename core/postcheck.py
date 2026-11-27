"""后置校验：答案出去之前，用程序核对一遍「有没有编」。

为什么要做（而不是只靠提示词）：
    提示词里写「不许编造」是**约束**，不是**保证**。模型会：
      · 引用一个不存在的片段编号（[7] 但只有 3 条证据）；
      · 把 `E1017` 的处置写成 `E1035`（码变了）；
      · 把「5 分钟」写成「15 分钟」（数字漂移）；
      · 把证据里的日期改一个月。
    这些都能用字符串比对抓出来，成本几乎为零。抓不住的是同义改写 —— 所以要
    配合 eval 的 grounding 指标和人工抽检，本模块只负责「确定性错误」。

三种处置（不做「一刀切拦掉」，那会把正常回答也杀掉）：
    · BLOCK：编造了错误码 / 数字 / 日期 → 换成保守话术（宁可说不知道）；
    · ANNOTATE：引用编号越界 → 重新编号到真实证据，并在末尾标「出处已校正」；
    · PASS：没问题。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

_CODE = re.compile(r"\b([A-Z]{1,4}\d{2,6})\b")  # E1017 / HTTP500 / AL2026
_NUM = re.compile(r"\d+(?:\.\d+)?")
_DATE = re.compile(r"(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})")
_CITE = re.compile(r"\[(\d{1,2})\]")
# 带单位的数字：单位必须是「换了就不是同一件事」的那种。
# 「年」也在内 —— 「保留三年」写成「保留 99 年」是典型的数字漂移。
# 值允许到 4 位（500MB、2026 年），但 4 位数的「年」在 check 里按日期处理，不在这里判。
_UNIT_NUM = re.compile(
    r"(\d+(?:\.\d+)?)\s*(年|个月|周|天|小时|分钟|秒|MB|GB|KB|厘米|米|类|项|倍|%)"
)

# 中文数字 → 阿拉伯：证据里写「三年」、答案写「3 年」时不该判为不一致。
# 只转「单字数字 + 单位」这一种形态，避免把「第一」「三、流程」也改掉。
_CN_DIGIT = str.maketrans("一二两三四五六七八九", "1223456789")
_CN_UNIT = re.compile(r"([一二两三四五六七八九十])\s*(年|个月|周|天|小时|分钟|秒|类|项|倍)")
_CN_TEN = re.compile(r"十")


class Severity(StrEnum):
    BLOCK = "block"
    ANNOTATE = "annotate"
    PASS = "pass"


@dataclass
class Violation:
    kind: str  # code / number / date / citation
    detail: str
    severity: Severity


@dataclass
class CheckReport:
    ok: bool = True
    action: Severity = Severity.PASS
    violations: list[Violation] = field(default_factory=list)
    answer: str = ""  # 校正后的答案（可能等于原答案）
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action.value,
            "violations": [
                {"kind": v.kind, "detail": v.detail, "severity": v.severity.value}
                for v in self.violations
            ],
            "notes": self.notes,
        }


def _cn_num(s: str) -> int:
    """中文数字（一~九十九）转整数。只用于「数字 + 单位」的比对，不追求通用性。"""
    if s == "十":
        return 10
    if "十" in s:
        head, _, tail = s.partition("十")
        tens = int(head.translate(_CN_DIGIT)) if head else 1
        ones = int(tail.translate(_CN_DIGIT)) if tail else 0
        return tens * 10 + ones
    return int(s.translate(_CN_DIGIT))


def _norm(text: str) -> str:
    """归一化：中文数字转阿拉伯 + 去空白 + 大写。

    为什么要转中文数字：证据写「保留三年」、答案写「保留 3 年」，
    不归一化就会被判成「数字不在证据里」，直接误杀正确答案。
    只转「数字 + 单位」这一种形态 —— 不碰「第一」「三、流程」这类序数。
    """
    t = re.sub(
        r"([一二两三四五六七八九十]+)\s*(年|个月|周|天|小时|分钟|秒|类|项|倍)",
        lambda m: f"{_cn_num(m.group(1))}{m.group(2)}",
        text or "",
    )
    return re.sub(r"\s+", "", t).upper()


def check_codes(answer: str, evidence: Sequence[str]) -> list[Violation]:
    """答案里的「类错误码」必须出现在证据里。

    只查形如 E1017 的强特征 —— 纯数字太容易误报（年份、金额、页码都像）。
    """
    pool = _norm(" ".join(evidence))
    out: list[Violation] = []
    for code in set(_CODE.findall(answer or "")):
        if _norm(code) not in pool:
            out.append(Violation("code", code, Severity.BLOCK))
    return out


def check_unit_numbers(answer: str, evidence: Sequence[str]) -> list[Violation]:
    """带单位的数字（5 分钟 / 50 MB / 90 秒）必须能在证据里找到。

    为什么只查带单位的：裸数字误报太多（「第 1 条」「2026 年」）。
    带单位的数字一旦漂移，业务含义就变了（50MB 写成 500MB 会直接误导）。
    """
    pool = _norm(" ".join(evidence))
    out: list[Violation] = []
    for value, unit in set(_UNIT_NUM.findall(answer or "")):
        # 4 位数的「年」是年份（2026 年），交给 check_dates 判，
        # 否则每句「2026 年」都要在证据里找一遍，误报率高
        if unit == "年" and len(value.split(".")[0]) >= 4:
            continue
        if _norm(value + unit) not in pool:
            out.append(Violation("number", f"{value}{unit}", Severity.BLOCK))
    return out


def check_dates(answer: str, evidence: Sequence[str]) -> list[Violation]:
    """日期必须与证据一致（只比对年月日三段，不管写法）。"""
    pool = _norm(" ".join(evidence))
    out: list[Violation] = []
    for y, m, d in set(_DATE.findall(answer or "")):
        token = f"{y}{int(m)}{int(d)}"
        if token not in pool and f"{y}{m}{d}" not in pool:
            out.append(Violation("date", f"{y}-{m}-{d}", Severity.BLOCK))
    return out


def check_citations(answer: str, evidence_count: int) -> list[Violation]:
    """引用编号不能越界：答案写 [5]，但实际只有 3 条证据。"""
    out: list[Violation] = []
    for idx in {int(n) for n in _CITE.findall(answer or "")}:
        if idx < 1 or idx > max(evidence_count, 0):
            out.append(
                Violation("citation", f"[{idx}] 超出证据数 {evidence_count}", Severity.ANNOTATE)
            )
    return out


def renumber_citations(answer: str, evidence_count: int) -> str:
    """把越界的引用去掉，而不是全篇重排 —— 重排可能把引用指错。"""

    def repl(m: re.Match[str]) -> str:
        idx = int(m.group(1))
        return m.group(0) if 1 <= idx <= evidence_count else ""

    cleaned = _CITE.sub(repl, answer or "")
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def postcheck(
    answer: str,
    evidence: Sequence[str],
    evidence_count: int | None = None,
    conservative_reply: str = "知识库中没有找到能支撑该结论的内容，我不做推测。",
) -> CheckReport:
    """一次跑完四类校验，返回处置建议。

    注意：**不做「自动改写数字」**。把 15 分钟改回 5 分钟看起来聪明，
    实际上是在用程序替模型编第二个答案；正确做法是拦掉，让上层重答或降级。
    """
    evidence = [e for e in (evidence or []) if e]
    count = len(evidence) if evidence_count is None else evidence_count

    violations: list[Violation] = []
    violations += check_codes(answer, evidence)
    violations += check_unit_numbers(answer, evidence)
    violations += check_dates(answer, evidence)
    violations += check_citations(answer, count)

    report = CheckReport(answer=answer, violations=violations)
    if not violations:
        return report

    report.ok = False
    if any(v.severity is Severity.BLOCK for v in violations):
        report.action = Severity.BLOCK
        report.answer = conservative_reply
        report.notes.append("检测到无法在证据中找到的错误码/数字/日期，已替换为保守回答")
        return report

    # 只剩引用越界：去掉越界引用，保留内容
    report.action = Severity.ANNOTATE
    report.answer = renumber_citations(answer, count)
    report.notes.append("引用了不存在的片段编号，已移除越界编号")
    return report


def summarize(reports: Sequence[CheckReport]) -> dict[str, Any]:
    """给评测/日志用：拦截率与主要问题类型分布。"""
    if not reports:
        return {"calls": 0, "blocked": 0, "annotated": 0, "by_kind": {}}
    by_kind: dict[str, int] = {}
    for r in reports:
        for v in r.violations:
            by_kind[v.kind] = by_kind.get(v.kind, 0) + 1
    return {
        "calls": len(reports),
        "blocked": sum(1 for r in reports if r.action is Severity.BLOCK),
        "annotated": sum(1 for r in reports if r.action is Severity.ANNOTATE),
        "by_kind": by_kind,
    }
