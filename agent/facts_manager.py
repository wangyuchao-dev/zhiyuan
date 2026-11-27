"""事实槽管理器：按会话存结构化关键事实（制度编号、条款号、错误码等）。

写入是同 key 覆盖（只保留当前值），和历史「只追加」互补；
值以 JSON 序列化，落 SQLite。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 事实槽管理器：保存会话内关键业务事实（制度编号、条款号、错误码、文件名等）
#
# 和历史的核心区别（一句话）：历史是「流水账」，事实是「当前值」。
#   历史 —— 只追加，客户改三次地址就留三条；
#   事实 —— 同 key 覆盖，只留最后一次的值，给代码用、精确可读。
#
# 存储位置：SQLite（agent/persistence.py）。值用 JSON 序列化，
# 所以槽位里放 dict / list / 数字都可以，不限于字符串。
from typing import Any, Optional

from agent.persistence import db


class FactManager:
    """事实槽管理器，维护单会话结构化业务槽位"""

    def __init__(self, chat_session_id: str):
        self.chat_session_id = chat_session_id

    def put(self, key: str, value: Any) -> None:
        """写入一条事实槽（同 key 覆盖）"""
        db().set_fact(self.chat_session_id, key, value)

    def get(self, key: str) -> Optional[Any]:
        """读取槽，不存在返回 None"""
        return db().get_facts(self.chat_session_id).get(key)

    def remove(self, key: str) -> None:
        """删除指定槽位"""
        db().delete_fact(self.chat_session_id, key)

    def get_all(self) -> dict[str, Any]:
        """获取全部槽（每次从库里读，返回的是新字典，改它不会影响库）"""
        return db().get_facts(self.chat_session_id)

    def clear(self) -> None:
        """清空当前会话所有事实槽"""
        db().clear_facts(self.chat_session_id)


# 第三题：事实槽管理器
