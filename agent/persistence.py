# -*- coding: utf-8 -*-
"""
会话记忆的持久化层（标准库 sqlite3，不引入新依赖）。

为什么要有这一层：
    改造前，历史 / 事实 / 摘要都存在模块级 dict 里，属于「进程内存」——
    服务一重启，所有会话记忆全丢；多个 worker 之间也不共享。
    把这三份数据落到 SQLite，进程重启后仍能接着聊，且单机部署够用。

为什么选 SQLite 而不是 Redis：
    这是课程/单机项目，SQLite 零部署、零依赖、单文件可拷走；
    接口层（ChatHistoryManager / FactManager / SummaryManager）不需要感知底层，
    以后要换 Redis，只改这个文件里的三个读写方法即可。

表结构：
    messages(session_id, seq, role, content)   对话原话，只追加
    facts(session_id, key, value)              事实槽，同 key 覆盖
    summaries(session_id, summary)             增量摘要，每会话一条

并发：
    sqlite3 连接默认禁止跨线程使用，而 FastAPI 的同步路由跑在线程池里，
    所以连接用 check_same_thread=False，并配一把进程内锁串行化写操作。
"""
import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

from config.setting import QaConfig

_LOCK = threading.Lock()
_DB: Optional["MemoryDB"] = None


class MemoryDB:
    """SQLite 封装：只管三张表的读写，不含任何业务语义"""

    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with _LOCK:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    session_id TEXT NOT NULL,
                    seq        INTEGER NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    PRIMARY KEY (session_id, seq)
                );
                CREATE TABLE IF NOT EXISTS facts (
                    session_id TEXT NOT NULL,
                    key        TEXT NOT NULL,
                    value      TEXT NOT NULL,
                    PRIMARY KEY (session_id, key)
                );
                CREATE TABLE IF NOT EXISTS summaries (
                    session_id TEXT PRIMARY KEY,
                    summary    TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, seq);
                """
            )
            self._conn.commit()

    # ---------------- 对话历史 ----------------
    def append_message(self, session_id: str, role: str, content: str) -> None:
        """追加一条消息。seq 用「当前最大 + 1」，保证顺序稳定。"""
        with _LOCK:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS m FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            next_seq = int(row["m"]) + 1
            self._conn.execute(
                "INSERT INTO messages(session_id, seq, role, content) VALUES (?, ?, ?, ?)",
                (session_id, next_seq, role, content),
            )
            self._conn.commit()

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        取消息，按 seq 升序。

        limit: 只取最近 N 条（None = 全部）。SQL 层用倒序 + LIMIT 再翻回来，
                避免把整段历史读进内存再切片。
        """
        with _LOCK:
            if limit is None or limit <= 0:
                rows = self._conn.execute(
                    "SELECT role, content FROM messages WHERE session_id = ? ORDER BY seq",
                    (session_id,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT role, content FROM messages WHERE session_id = ? "
                    "ORDER BY seq DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
                rows = list(reversed(rows))
        return [(r["role"], r["content"]) for r in rows]

    def clear_messages(self, session_id: str) -> None:
        """清空某会话的全部对话记录"""
        with _LOCK:
            self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self._conn.commit()

    # ---------------- 事实槽 ----------------
    def set_fact(self, session_id: str, key: str, value: Any) -> None:
        """写入/覆盖一个事实槽"""
        with _LOCK:
            self._conn.execute(
                "INSERT INTO facts(session_id, key, value) VALUES (?, ?, ?) "
                "ON CONFLICT(session_id, key) DO UPDATE SET value = excluded.value",
                (session_id, key, json.dumps(value, ensure_ascii=False)),
            )
            self._conn.commit()

    def get_facts(self, session_id: str) -> Dict[str, Any]:
        """取某会话的全部事实槽（新建字典，改它不影响库）"""
        with _LOCK:
            rows = self._conn.execute(
                "SELECT key, value FROM facts WHERE session_id = ?", (session_id,)
            ).fetchall()
        out: Dict[str, Any] = {}
        for r in rows:
            try:
                out[r["key"]] = json.loads(r["value"])
            except (TypeError, ValueError):
                out[r["key"]] = r["value"]
        return out

    def delete_fact(self, session_id: str, key: str) -> None:
        """删除单个事实槽"""
        with _LOCK:
            self._conn.execute(
                "DELETE FROM facts WHERE session_id = ? AND key = ?", (session_id, key)
            )
            self._conn.commit()

    def clear_facts(self, session_id: str) -> None:
        """清空某会话的全部事实槽"""
        with _LOCK:
            self._conn.execute("DELETE FROM facts WHERE session_id = ?", (session_id,))
            self._conn.commit()

    # ---------------- 摘要 ----------------
    def set_summary(self, session_id: str, summary: str) -> None:
        """写入/覆盖某会话的摘要"""
        with _LOCK:
            self._conn.execute(
                "INSERT INTO summaries(session_id, summary) VALUES (?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET summary = excluded.summary",
                (session_id, summary or ""),
            )
            self._conn.commit()

    def get_summary(self, session_id: str) -> str:
        """取某会话的摘要，没有则返回空串"""
        with _LOCK:
            row = self._conn.execute(
                "SELECT summary FROM summaries WHERE session_id = ?", (session_id,)
            ).fetchone()
        return row["summary"] if row else ""

    def clear_summary(self, session_id: str) -> None:
        """删除某会话的摘要"""
        with _LOCK:
            self._conn.execute("DELETE FROM summaries WHERE session_id = ?", (session_id,))
            self._conn.commit()

    # ---------------- 整体 ----------------
    def clear_session(self, session_id: str) -> None:
        """一个会话的三份数据一起清（/api/reset 用它）"""
        self.clear_messages(session_id)
        self.clear_facts(session_id)
        self.clear_summary(session_id)

    def close(self) -> None:
        "关闭连接（单元测试切库路径时调用；正常服务运行期不必关）"
        with _LOCK:
            self._conn.close()


def db() -> MemoryDB:
    """进程内单例。第一次访问时才建库建表。"""
    global _DB
    if _DB is None:
        _DB = MemoryDB(QaConfig().memory_db_path)
    return _DB


def set_db_path(path: Optional[str] = None) -> MemoryDB:
    """
    切换数据库文件（单元测试用，也方便以后把记忆指到别的盘）。

    path=None 时回退到 QaConfig 里的默认路径。
    """
    global _DB
    if _DB is not None:
        try:
            _DB.close()
        except Exception:
            pass
        _DB = None
    _DB = MemoryDB(path or QaConfig().memory_db_path)
    return _DB
