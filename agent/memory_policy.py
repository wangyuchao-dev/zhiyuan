"""记忆策略：先分类，再谈存。四类东西的生命周期完全不同，混在一起就是垃圾场。

四类（对应"必须区分事实/偏好/临时状态"）：

| 类型 | 例子 | 生命周期 | 检索方式 |
|---|---|---|---|
| FACT 事实 | 部门=风控部、岗位=柜员 | 长期，几乎不变 | **按 key 精确取**，不做相似度 |
| PREFERENCE 偏好 | 喜欢表格、回答要简短 | 中长期，可被覆盖 | 关键词匹配 + 严格门槛 |
| EPHEMERAL 临时 | 刚才那个文档、这轮的城市 | 分钟级，会话内 | 只从会话上下文取，**不落长期库** |
| SUMMARY 摘要 | 早前对话的压缩 | 长期，会更新 | 仅长对话时带一段 |

三条铁律：
  1. **写入门槛**：疑问句、客气话、"嗯"这类不进库 —— 一进去就是永久噪音；
  2. **版本号**：同 key 覆盖时 version+1，值没变不写（避免无意义写入）；
  3. **检索从紧**：宁可少带一条，也不放一条不确定的。事实类根本不用相似度，
     用相似度找"部门"会匹配到"部门经理"。
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

_QUESTION = re.compile(r"(吗|呢|怎么|如何|什么|为什么|多少|哪[个些里]|\?|？)\s*$")
_FILLER = {"嗯", "好的", "谢谢", "在吗", "你好", "收到", "ok", "嗯嗯", "哈哈哈"}
_WORD = re.compile(r"[a-z0-9]+|[\u4e00-\u9fa5]{2,}")


class Kind(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    EPHEMERAL = "ephemeral"
    SUMMARY = "summary"


@dataclass
class Item:
    key: str
    kind: Kind
    value: str
    version: int = 1
    updated_at: float = field(default_factory=time.time)
    ttl_s: float | None = None  # None = 不过期（仅 FACT / PREFERENCE 可设）
    source: str = ""

    def expired(self, now: float | None = None) -> bool:
        if self.ttl_s is None:
            return False
        return (now or time.time()) - self.updated_at > self.ttl_s


def tokens(text: str) -> set:
    """中文按 bi-gram 切 —— 中文没有空格，按整段切出来的是一个长 token，
    永远匹配不上，这是最初版本的真实 bug。"""
    t = (text or "").lower()
    out = set(re.findall(r"[a-z0-9]+", t))
    for seg in re.findall(r"[\u4e00-\u9fa5]+", t):
        out.add(seg)
        out.update(seg[i : i + 2] for i in range(len(seg) - 1))
    return out


class MemoryPolicy:
    """纯策略：不碰数据库，只做判断与合并。存储由 persistence 负责。

    这样拆的好处：策略可以单测（不依赖 SQLite），换存储也不影响规则。
    """

    def __init__(
        self,
        max_value_len: int = 200,
        search_min_overlap: float = 0.3,
        search_max_items: int = 1,
        ephemeral_ttl: float = 600,
        search_min_hits: int = 2,
    ) -> None:
        self.max_value_len = max_value_len
        self.search_min_overlap = search_min_overlap  # 覆盖率门槛
        self.search_min_hits = search_min_hits  # 至少 2 个共同词才算命中
        self.search_max_items = search_max_items  # 默认只放一条
        self.ephemeral_ttl = ephemeral_ttl

    # ---- 写入门槛 ----
    def should_write(self, kind: Kind, text: str) -> tuple[bool, str]:
        """返回 (能不能写, 原因)。原因会进日志，方便回查"为什么没记住"。"""
        text = (text or "").strip()
        if not text:
            return False, "empty"
        if text.lower() in _FILLER:
            return False, "filler"
        if _QUESTION.search(text):
            return False, "question"  # 提问不是事实，写进去会污染检索
        if len(text) > self.max_value_len:
            return False, "too_long"  # 长文本该进摘要，不该进事实库
        if kind is Kind.EPHEMERAL:
            # 临时状态只在会话里用；真写库也要带 TTL，否则三天后还在影响回答
            return True, "ephemeral_ttl"
        return True, "ok"

    # ---- 写入 ----
    def upsert(self, store: dict[str, Item], item: Item) -> tuple[Item, bool]:
        """同 key 覆盖：值没变就不写，变了 version+1。返回 (最终项, 是否真的写了)。"""
        ok, reason = self.should_write(item.kind, item.value)
        if item.kind is not Kind.EPHEMERAL and not ok:
            # 不允许写时返回原值（可能是 None 场景，由调用方判断）
            return item, False
        old = store.get(item.key)
        if old is not None:
            if old.value == item.value and old.kind is item.kind:
                return old, False  # 幂等：重复说同一件事不该涨版本
            item.version = old.version + 1
            item.updated_at = time.time()
        if item.kind is Kind.EPHEMERAL and item.ttl_s is None:
            item.ttl_s = self.ephemeral_ttl
        store[item.key] = item
        return item, True

    # ---- 读取 ----
    def get(self, store: dict[str, Item], key: str) -> Item | None:
        """事实类只按 key 取。不要"相似 key 兜底"——那是最容易串味的地方。"""
        item = store.get(key)
        if item and item.expired():
            store.pop(key, None)
            return None
        return item

    def search(
        self,
        store: dict[str, Item],
        query: str,
        kinds: Sequence[Kind] = (Kind.PREFERENCE,),
        min_overlap: float | None = None,
        max_items: int | None = None,
    ) -> list[Item]:
        """只在允许的类型里做关键词匹配，门槛默认 0.6、最多 1 条。

        为什么门槛这么高：偏好类误召回会直接改变回答风格（把"简短"当成"详细"），
        比漏一条的代价大得多。
        """
        q = tokens(query)
        if not q:
            return []
        need = self.search_min_overlap if min_overlap is None else min_overlap
        limit = self.search_max_items if max_items is None else max_items
        scored: list[tuple[float, Item]] = []
        for item in store.values():
            if item.kind not in kinds or item.expired():
                continue
            t = tokens(item.value) | tokens(item.key)
            if not t:
                continue
            shared = q & t
            if len(shared) < self.search_min_hits:  # 只共用一个词，误召回风险太高
                continue
            score = len(shared) / max(1, min(len(q), len(t)))
            if score >= need:
                scored.append((score, item))
        scored.sort(key=lambda x: -x[0])
        return [i for _, i in scored[:limit]]

    # ---- 维护 ----
    def purge_expired(self, store: dict[str, Item], now: float | None = None) -> int:
        dead = [k for k, v in store.items() if v.expired(now)]
        for k in dead:
            store.pop(k, None)
        return len(dead)

    def prompt_lines(self, items: Iterable[Item]) -> list[str]:
        """给提示词用的短行：`- key: value`，不带版本号（版本是给排查用的）。"""
        return [f"- {i.key}：{i.value}" for i in items]
