"""父子分块：检索用小碎块，喂模型用大块。

为什么需要：
    切太碎 → 检索准，但模型拿到的上下文是断句，答不完整；
    切太大 → 上下文完整，但一个块里混了五件事，向量被稀释，检索不准。
父子分块把两个目标拆开：**用 child 检索，命中后回溯 parent 塞给模型**。

实现要点：
    · parent 按标题切（一级/二级标题天然是一个语义单元），没有标题时按段落聚合；
    · child 在 parent 内部按句子打包，句间留 overlap；
    · child 的 id 里带 parent 指针，检索命中后按 parent 去重，避免同一段进两次上下文。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

try:  # 单独跑这个文件时也能 import
    from core.context_budget import estimate_tokens
except ImportError:  # pragma: no cover

    def estimate_tokens(text: str) -> int:  # 退化实现，仅保证可运行
        return len(text or "")


_HEADING = re.compile(r"^(#{1,3})\s+(.+)$", re.M)
_SENT = re.compile(r"[^。！？!?\n]+[。！？!?]?")


@dataclass
class Parent:
    id: str
    doc_id: str
    title: str
    text: str
    order: int


@dataclass
class Child:
    id: str
    parent_id: str
    doc_id: str
    text: str
    order: int
    score: float = 0.0  # 检索时回填


def _hash(*parts: str) -> str:
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:10]


def split_parents(doc_id: str, text: str, max_tokens: int = 900) -> list[Parent]:
    """按标题切父块；单块超过 max_tokens 时按段落再切一次。"""
    text = (text or "").strip()
    if not text:
        return []

    marks = [(m.start(), m.group(2).strip()) for m in _HEADING.finditer(text)]
    blocks: list[tuple] = []
    if not marks:
        blocks = [("正文", text)]
    else:
        if marks[0][0] > 0:  # 第一个标题之前的内容
            head = text[: marks[0][0]].strip()
            if head:
                blocks.append(("前言", head))
        for i, (pos, title) in enumerate(marks):
            end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
            blocks.append((title, text[pos:end].strip()))

    parents: list[Parent] = []
    for title, body in blocks:
        if estimate_tokens(body) <= max_tokens:
            parents.append(
                Parent(_hash(doc_id, title, str(len(parents))), doc_id, title, body, len(parents))
            )
            continue
        # 太长：按空行分段，攒到接近上限就切一个父块
        buf, order = [], 0
        for para in re.split(r"\n\s*\n", body):
            para = para.strip()
            if not para:
                continue
            if buf and estimate_tokens("\n\n".join(buf + [para])) > max_tokens:
                parents.append(
                    Parent(
                        _hash(doc_id, title, str(order)),
                        doc_id,
                        f"{title}（{order + 1}）",
                        "\n\n".join(buf),
                        len(parents),
                    )
                )
                buf, order = [], order + 1
            buf.append(para)
        if buf:
            parents.append(
                Parent(
                    _hash(doc_id, title, str(order)),
                    doc_id,
                    f"{title}（{order + 1}）" if order else title,
                    "\n\n".join(buf),
                    len(parents),
                )
            )
    return parents


def split_children(parent: Parent, size: int = 220, overlap: int = 40) -> list[Child]:
    """父块内部按句子打包成子块。"""
    sents = [s.strip() for s in _SENT.findall(parent.text) if s.strip()]
    if not sents:
        return []
    children: list[Child] = []
    buf: list[str] = []
    for s in sents:
        candidate = buf + [s]
        if buf and estimate_tokens("".join(candidate)) > size:
            text = "".join(buf)
            children.append(
                Child(
                    _hash(parent.id, str(len(children))),
                    parent.id,
                    parent.doc_id,
                    text,
                    len(children),
                )
            )
            # overlap：保留尾部若干字符，避免答案被切在两句之间
            tail = text[-overlap:] if overlap and len(text) > overlap else ""
            buf = [tail, s] if tail else [s]
        else:
            buf = candidate
    if buf:
        children.append(
            Child(
                _hash(parent.id, str(len(children))),
                parent.id,
                parent.doc_id,
                "".join(buf),
                len(children),
            )
        )
    return children


@dataclass
class IndexedDoc:
    doc_id: str
    parents: list[Parent] = field(default_factory=list)
    children: list[Child] = field(default_factory=list)


class ParentChildIndex:
    """内存索引：只负责切分与回溯，向量化与存储交给上层。

    为什么要独立出来：切分规则变了（比如调整 child 大小），
    检索侧完全不用动 —— 只要 child 还能指回 parent。
    """

    def __init__(
        self, child_size: int = 220, child_overlap: int = 40, parent_max: int = 900
    ) -> None:
        self.child_size = child_size
        self.child_overlap = child_overlap
        self.parent_max = parent_max
        self.docs: dict[str, IndexedDoc] = {}
        self._parent_by_id: dict[str, Parent] = {}

    def add(self, doc_id: str, text: str) -> IndexedDoc:
        parents = split_parents(doc_id, text, self.parent_max)
        children: list[Child] = []
        for p in parents:
            children.extend(split_children(p, self.child_size, self.child_overlap))
            self._parent_by_id[p.id] = p
        doc = IndexedDoc(doc_id, parents, children)
        self.docs[doc_id] = doc
        return doc

    def children_of(self, doc_id: str) -> list[Child]:
        return self.docs.get(doc_id, IndexedDoc(doc_id)).children

    def expand(self, hits: Sequence[Child], max_parents: int = 3) -> list[Parent]:
        """把命中的 child 回溯成 parent，按最高分排序、去重。

        注意：这里按 parent 去重，所以同一段被多个 child 命中时只出现一次 ——
        否则上下文里会出现三段几乎一样的文字，白占预算还干扰模型。
        """
        best: dict[str, float] = {}
        for h in hits:
            best[h.parent_id] = max(best.get(h.parent_id, 0.0), getattr(h, "score", 0.0))
        ordered = sorted(best.items(), key=lambda kv: -kv[1])[:max_parents]
        return [self._parent_by_id[pid] for pid, _ in ordered if pid in self._parent_by_id]

    def stats(self) -> dict[str, int]:
        return {
            "docs": len(self.docs),
            "parents": sum(len(d.parents) for d in self.docs.values()),
            "children": sum(len(d.children) for d in self.docs.values()),
        }
