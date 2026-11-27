"""SKILL 注册表：只把「名字 + 一句简介」放进提示词，命中才加载正文。

为什么这么做（和 MCP 的分工）：
    MCP 解决「能不能碰到外部系统」，SKILL 解决「碰到之后怎么干」。
    一个技能的正文明明有几百行操作手册，全塞进提示词会直接挤爆上下文，
    而且会稀释注意力。所以只暴露目录，模型判断相关了再取正文 —— 渐进式披露。

加载时机：
    index()      启动时读 front-matter（name / summary），几十个技能也只占几行；
    load_body()  真正用到时才读全文，进程内缓存。

格式（SKILL.md）：
    ---
    name: fault-code-lookup
    summary: 根据错误码查原因与处置步骤，适用于「E1017 是什么」这类问题
    keywords: 错误码, 报错, E1, E2, 排查
    ---
    # 正文（操作步骤 / 判断表 / 输出格式）
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

_FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)
_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fa5]{2,}")


@dataclass
class SkillMeta:
    name: str
    summary: str
    path: str
    keywords: list[str] = field(default_factory=list)

    def line(self) -> str:
        """给提示词的一行目录项 —— 越短越好。"""
        return f"- {self.name}：{self.summary}"


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """解析极简 front-matter（key: value），不引入 yaml 依赖。

    只支持单行值，够用；复杂结构（列表）用逗号分隔。
    """
    m = _FM.match(text or "")
    if not m:
        return {}, text or ""
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip()
    return meta, text[m.end() :]


def tokens(text: str) -> set:
    """粗糙关键词切分：英文词 + 中文连续段（不做分词，避免引入 jieba）。"""
    return {t.lower() for t in _TOKEN.findall((text or "").lower())}


class SkillRegistry:
    def __init__(self, root: str) -> None:
        self.root = root
        self._metas: dict[str, SkillMeta] = {}
        self._bodies: dict[str, str] = {}

    # ---- 目录 ----
    def index(self) -> list[SkillMeta]:
        """扫描目录，只读 front-matter，不读正文。"""
        self._metas.clear()
        if not os.path.isdir(self.root):
            return []
        for entry in sorted(os.listdir(self.root)):
            path = os.path.join(self.root, entry, "SKILL.md")
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as f:
                meta, _ = parse_front_matter(f.read())
            name = meta.get("name") or entry
            self._metas[name] = SkillMeta(
                name=name,
                summary=meta.get("summary", "")[:120],
                path=path,
                keywords=[k.strip() for k in meta.get("keywords", "").split(",") if k.strip()],
            )
        return list(self._metas.values())

    def manifest(self) -> str:
        """提示词里的技能目录（只有名字与简介）。"""
        if not self._metas:
            return ""
        return "可用技能（需要时再展开）：\n" + "\n".join(m.line() for m in self._metas.values())

    # ---- 匹配 ----
    def match(self, query: str, top: int = 2) -> list[SkillMeta]:
        """按关键词重合度排序，返回最相关的几个。

        为什么不用向量：技能数量少、描述短，关键词匹配足够且零成本；
        真到上百个技能再上 embedding，别提前优化。
        """
        q = tokens(query)
        if not q:
            return []
        scored: list[tuple[float, SkillMeta]] = []
        for m in self._metas.values():
            kw = set()
            for k in m.keywords:
                kw |= tokens(k)
            doc = tokens(m.summary) | tokens(m.name.replace("-", " ")) | kw
            if not doc:
                continue
            hit = len(q & doc)
            if hit == 0:
                continue
            # 关键词命中权重更高：它是人工标注过的强信号
            score = hit / max(1, len(q)) + 0.5 * len(q & kw)
            scored.append((score, m))
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored[:top]]

    # ---- 正文 ----
    def load_body(self, name: str) -> str:
        """按需读正文，带缓存。调用方负责决定要不要真的展开。"""
        if name in self._bodies:
            return self._bodies[name]
        meta = self._metas.get(name)
        if not meta:
            return ""
        with open(meta.path, encoding="utf-8") as f:
            _, body = parse_front_matter(f.read())
        self._bodies[name] = body.strip()
        return self._bodies[name]

    def render(self, query: str, max_skills: int = 1) -> str:
        """匹配 + 展开：返回要插进提示词的那一段。

        max_skills 默认 1 —— 两个技能同时展开会互相干扰（指令冲突），
        这是踩过的坑：一个说「先查错误码」，一个说「先问用户现象」。
        """
        picked = self.match(query, top=max_skills)
        if not picked:
            return ""
        parts = []
        for m in picked:
            parts.append(f"### 技能：{m.name}\n{self.load_body(m.name)}")
        return "\n\n".join(parts)

    def stats(self) -> dict[str, int]:
        return {"skills": len(self._metas), "loaded": len(self._bodies)}
