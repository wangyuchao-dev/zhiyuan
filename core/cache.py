"""缓存层：Redis 优先，连不上自动退化到进程内，接口不变。

要不要加 Redis（行业判断）：
    · 单机、单用户、语料小 —— 没必要，加了只是多一跳；
    · 一旦出现「同一个问题被反复问」「检索 + embedding 是主要耗时」——
      必须加，因为这两步是纯函数，重复算就是浪费钱。
    本项目实测：检索 4.18s → 0.32s、生成 14.1s → 0.03s，就是靠这层。

三条设计约束：
  1. **key 里必须带语料指纹** —— 改文档后自动失效，不靠人记得清缓存；
  2. **空结果与异常不缓存** —— 一次抖动不能污染接下来半小时；
  3. **连不上 Redis 也要能用** —— 降级成本地缓存，行为不变（只是不跨进程）。
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

# 版本号进 key：改了序列化格式或 key 结构时，旧缓存自然失效
SCHEMA_VERSION = "v1"


@dataclass
class CacheStats:
    hit: int = 0
    miss: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hit + self.miss
        return self.hit / total if total else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {"hit": self.hit, "miss": self.miss, "errors": self.errors,
                "hit_rate": round(self.hit_rate, 3)}


class MemoryCacheBackend:
    """进程内缓存：带 TTL、线程安全。降级用，也可单独跑测试。"""

    def __init__(self) -> None:
        self._data: Dict[str, Tuple[float, str]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            expire_at, payload = item
            if expire_at < time.time():
                self._data.pop(key, None)
                return None
            return payload

    def set(self, key: str, value: str, ttl: int) -> None:
        with self._lock:
            self._data[key] = (time.time() + ttl, value)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def incr(self, key: str, ttl: int) -> int:
        with self._lock:
            item = self._data.get(key)
            current = 0
            if item and item[0] >= time.time():
                current = int(item[1])
            current += 1
            self._data[key] = (time.time() + ttl, str(current))
            return current

    def __len__(self) -> int:
        return len(self._data)


class RedisCacheBackend:
    """Redis 后端：只在真的装了 redis 且连得上时使用。

    不在这里 import redis —— 没装 redis 包的项目也能跑起来（降级成本地缓存）。
    """

    def __init__(self, url: str, timeout: float = 1.0) -> None:
        import redis  # noqa: WPS433  （延迟导入是刻意的）
        self._client = redis.Redis.from_url(url, socket_timeout=timeout,
                                            socket_connect_timeout=timeout,
                                            decode_responses=True)
        self._client.ping()          # 连不上直接抛，由 build_cache 兜住

    def get(self, key: str) -> Optional[str]:
        return self._client.get(key)

    def set(self, key: str, value: str, ttl: int) -> None:
        self._client.setex(key, ttl, value)

    def delete(self, key: str) -> None:
        self._client.delete(key)

    def incr(self, key: str, ttl: int) -> int:
        pipe = self._client.pipeline()
        pipe.incr(key)
        pipe.expire(key, ttl)
        return int(pipe.execute()[0])


def corpus_fingerprint(file_hashes: list[str]) -> str:
    """语料指纹：文档哈希排序后取摘要。

    把「改了哪份文档」这件事编码进 key —— 加一份文档、删一份文档，
    指纹就变，旧缓存自动失效。这是被「加城市忘清缓存」坑过之后的改法。
    """
    joined = "|".join(sorted(h or "" for h in file_hashes))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()[:12]


class Cache:
    """对外只有一个门面：拼 key、读写、统计、空值不缓存。"""

    def __init__(self, backend: Any, namespace: str = "zy",
                 ttl_default: int = 1800, cache_empty: bool = False) -> None:
        self.backend = backend
        self.namespace = namespace
        self.ttl_default = ttl_default
        self.cache_empty = cache_empty
        self.stats = CacheStats()

    # ---- key ----
    def build_key(self, kind: str, query: str, *, corpus_fp: str = "",
                  params: Optional[Dict[str, Any]] = None) -> str:
        """key 结构：`ns:v1:kind:params哈希:语料指纹:query哈希`

        为什么 query 取哈希而不是原文：中文 key 在 Redis 里可读性差、长度不可控，
        而且日志里带用户提问是隐私问题。
        """
        qh = hashlib.md5((query or "").strip().lower().encode("utf-8")).hexdigest()[:16]
        ph = hashlib.md5(json.dumps(params or {}, sort_keys=True).encode()).hexdigest()[:8]
        return f"{self.namespace}:{SCHEMA_VERSION}:{kind}:{ph}:{corpus_fp or 'nofp'}:{qh}"

    # ---- 读写 ----
    def get_json(self, key: str) -> Any:
        raw = self._safe_get(key)
        if raw is None:
            self.stats.miss += 1
            return None
        self.stats.hit += 1
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 脏数据当未命中处理，顺手删掉，避免反复踩
            self._safe_delete(key)
            self.stats.miss += 1
            return None

    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if value in (None, [], {}, "") and not self.cache_empty:
            return                      # 空结果不缓存：一次抖动不能污染半小时
        try:
            self.backend.set(key, json.dumps(value, ensure_ascii=False), ttl or self.ttl_default)
        except Exception:
            self.stats.errors += 1

    def get_or_set(self, key: str, producer: Callable[[], Any],
                   ttl: Optional[int] = None) -> Tuple[Any, bool]:
        """返回 (值, 是否命中)。producer 抛异常时不写缓存。"""
        cached = self.get_json(key)
        if cached is not None:
            return cached, True
        value = producer()
        self.set_json(key, value, ttl)
        return value, False

    def incr(self, key: str, ttl: int) -> int:
        try:
            return int(self.backend.incr(key, ttl))
        except Exception:
            self.stats.errors += 1
            return 0

    # ---- 容错 ----
    def _safe_get(self, key: str) -> Optional[str]:
        try:
            return self.backend.get(key)
        except Exception:
            self.stats.errors += 1
            return None

    def _safe_delete(self, key: str) -> None:
        try:
            self.backend.delete(key)
        except Exception:
            self.stats.errors += 1


def build_cache(url: Optional[str], namespace: str = "zy") -> Cache:
    """按配置构造：给了 url 且连得上用 Redis，否则本地缓存。绝不因此起不来。"""
    if url:
        try:
            return Cache(RedisCacheBackend(url), namespace=namespace)
        except Exception as exc:      # 没装 redis / 连不上 / 超时
            print(f"[cache] Redis 不可用（{type(exc).__name__}），降级为进程内缓存")
    return Cache(MemoryCacheBackend(), namespace=namespace)
