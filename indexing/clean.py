"""入库前的清洗：不洗干净，后面全白搭。

为什么要单独一层（而不是在切分里顺手 replace）：
    · 清洗规则会被反复调（PDF 来源不同，脏法不同），放在切分里改一次就要重测切分；
    · 清洗的收益必须可量化（去掉了多少行、多少重复段），否则不知道值不值得；
    · 有些脏东西（页眉页脚）**只有在整篇文字里才能识别**，单页看不到。

五类必清（都是踩过的）：
    1. 页眉页脚 —— 每页重复，进了索引会让「公司名」成为高频词，检索被污染；
    2. 页码 / 「第 x 页 共 y 页」—— 纯噪声，还会让「3」变成关键词；
    3. 目录行 —— 「第一章 ........ 12」这种，切分后是半句话；
    4. 全角数字与字母 —— 「５０ＭＢ」检索不到「50MB」；
    5. 连续重复段落 —— 扫描 PDF 的双栏重叠、复制粘贴的残留。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

_PAGE_NO = re.compile(r"^\s*[-—–－]?\s*(?:第\s*)?\d{1,4}\s*(?:页)?\s*(?:[/／共]\s*\d{1,4}\s*页?)?\s*[-—–－]?\s*$")
_PAGE_TOTAL = re.compile(r"第\s*\d+\s*页\s*[，,]?\s*共\s*\d+\s*页")
_TOC_LINE = re.compile(r"[.．·…]{4,}\s*\d{1,4}\s*$")
_ZW = re.compile(r"[\u200b-\u200f\u2028\u2029\ufeff]")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAIL = re.compile(r"[ \t]+$", re.M)


@dataclass
class CleanResult:
    text: str
    stats: Dict[str, Any] = field(default_factory=dict)


def _pages(text: str) -> List[str]:
    """按换页符切页；没有换页符就按 40 行粗略切（页眉页脚检测需要「跨页重复」）。"""
    if "\f" in text:
        return text.split("\f")
    lines = text.split("\n")
    return ["\n".join(lines[i:i + 40]) for i in range(0, len(lines), 40)] or [text]


def find_repeated_lines(text: str, min_pages: int = 3, ratio: float = 0.3,
                        max_len: int = 40) -> List[str]:
    """找跨页重复的短行（页眉页脚）。

    判据：出现在 ≥ max(min_pages, ratio×页数) 个页面上，且长度 ≤ max_len。
    为什么要两条：只按次数会把「目录」误伤（目录页本身重复），只按比例在小文档上不生效。
    """
    pages = _pages(text)
    if len(pages) < min_pages:
        return []
    need = max(min_pages, int(len(pages) * ratio))
    counter: Dict[str, int] = {}
    for page in pages:
        seen = set()
        for line in page.split("\n"):
            s = line.strip()
            if 2 <= len(s) <= max_len and s not in seen:
                seen.add(s)
                counter[s] = counter.get(s, 0) + 1
    return [s for s, c in counter.items() if c >= need]


def clean_text(raw: str, *, drop_header_footer: bool = True,
               drop_toc: bool = True) -> CleanResult:
    """主入口。返回清洗后的文本 + 统计（统计必须能说清「去了多少」）。"""
    raw = raw or ""
    stats: Dict[str, Any] = {"in_chars": len(raw)}

    # ---- 0. 先在原文上找跨页重复行 ----
    # 必须在任何字符处理之前做：换页符一旦被当控制符删掉，页与页就会粘成一行，
    # 页眉也跟着粘到正文尾部，再也识别不出来（这是实测踩到的）。
    # 注意：repeated 必须与正文用同一套归一化（NFKC 会把全角括号变成半角），
    # 否则「原文找到的页眉」和「清洗后的行」永远对不上 —— 这是第二次踩。
    repeated = ({unicodedata.normalize("NFKC", r).strip()
                 for r in find_repeated_lines(raw)} if drop_header_footer else set())

    # ---- 1. 字符级 ----
    text = unicodedata.normalize("NFKC", raw)       # 全角 → 半角（５０ＭＢ → 50MB）
    text = text.replace("\f", "\n\n")              # 换页 → 空行，保住行边界
    text = _ZW.sub("", text)
    text = _CTRL.sub("", text)
    text = text.replace("\ufffd", "")               # 解码失败留下的替换字符

    # ---- 2. 页眉页脚 ----
    removed_hf = 0
    if repeated:
        if repeated:
            kept = []
            for line in text.split("\n"):
                if line.strip() and line.strip() in repeated:
                    removed_hf += 1
                    continue
                kept.append(line)
            text = "\n".join(kept)

    # ---- 3. 行级：页码 / 目录 ----
    kept_lines: List[str] = []
    removed_page, removed_toc = 0, 0
    for line in text.split("\n"):
        s = line.rstrip()
        if _PAGE_NO.match(s) or _PAGE_TOTAL.search(s):
            removed_page += 1
            continue
        if drop_toc and _TOC_LINE.search(s):
            removed_toc += 1
            continue
        kept_lines.append(s)
    text = "\n".join(kept_lines)

    # ---- 4. 段落级：去掉连续重复段 ----
    paras = re.split(r"\n\s*\n", text)
    deduped: List[str] = []
    removed_dup = 0
    prev_norm = None
    for p in paras:
        key = re.sub(r"\s+", "", p)
        if key and key == prev_norm:          # 只去「紧邻重复」，避免误删正文里的重复句
            removed_dup += 1
            continue
        deduped.append(p)
        prev_norm = key if key else prev_norm

    # ---- 5. 收尾：多余空白 ----
    text = "\n\n".join(deduped)
    text = _TRAIL.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = _MULTI_BLANK.sub("\n\n", text).strip()

    stats.update({
        "out_chars": len(text),
        "kept_ratio": round(len(text) / max(1, len(raw)), 3),
        "removed_header_footer": removed_hf,
        "removed_page_number": removed_page,
        "removed_toc": removed_toc,
        "removed_duplicate_para": removed_dup,
        "pages": len(_pages(raw)),
    })
    return CleanResult(text, stats)


def quality_gate(result: CleanResult, min_ratio: float = 0.3,
                 min_chars: int = 50) -> Tuple[bool, str]:
    """入库前的门槛：清完太短或丢太多，多半是 PDF 本身有问题（扫描件/加密）。

    宁可拒收并提示「可能是扫描件」，也不要把半页乱码灌进向量库 ——
    脏数据进了库，后面所有评测都是白跑。
    """
    if len(result.text) < min_chars:
        return False, f"清洗后仅 {len(result.text)} 字，疑似扫描件或加密文档"
    if result.stats.get("kept_ratio", 1) < min_ratio:
        return False, f"清洗后仅保留 {result.stats['kept_ratio']:.0%}，请检查原文质量"
    return True, "ok"
