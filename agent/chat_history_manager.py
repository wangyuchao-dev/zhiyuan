"""对话历史管理器：按会话存「原话」，只追加不修改。

存储落在 SQLite（agent/persistence.py），服务重启后历史仍在。
和历史相对的是事实槽（FactManager）：历史是流水账，事实是当前值。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 对话历史管理器：维护单会话消息列表
#
# 存储位置：SQLite（agent/persistence.py），不再是进程内字典。
# 这样服务重启后历史还在，多个 worker 之间也能看到同一份记忆。
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from agent.persistence import db


def _to_message(role: str, content: str) -> BaseMessage:
    """把库里的 (role, content) 还原成 LangChain 消息对象"""
    if role == "human":
        return HumanMessage(content=content)
    return AIMessage(content=content)


class ChatHistoryManager:
    """对话历史管理器，针对单个会话，支持追加、读取、清空消息"""

    def __init__(self, chat_session_id: str):
        self.chat_session_id = chat_session_id

    def add_message(self, msg: BaseMessage) -> None:
        """追加一条消息，支持 HumanMessage / AIMessage"""
        if isinstance(msg, HumanMessage):
            db().append_message(self.chat_session_id, "human", msg.content)
        elif isinstance(msg, AIMessage):
            db().append_message(self.chat_session_id, "ai", msg.content)
        # 其他类型（SystemMessage / ToolMessage）不进历史：
        # 系统注入的内容由 Memory.build_context 现拼，不需要落库。

    def get_all_messages(self) -> List[BaseMessage]:
        """获取当前会话全部消息（按时间顺序）"""
        return [_to_message(r, c) for r, c in db().get_messages(self.chat_session_id)]

    def get_recent_messages(self, n: int) -> List[BaseMessage]:
        """
        获取当前会话最近 n 条消息
        n 比总数大 → 返回全部
        n <= 0     → 返回空列表（注意 [-0:] 会变成 [0:]，所以必须单独判断）
        """
        if n <= 0:
            return []
        return [_to_message(r, c) for r, c in db().get_messages(self.chat_session_id, limit=n)]

    def clear(self) -> None:
        """清空当前会话历史"""
        db().clear_messages(self.chat_session_id)

    def to_plain_text(self, limit: int = None) -> str:
        """把消息拼接成纯文本，供给摘要、意图识别使用

        limit: 只取最近 N 条（None = 全部）。
               意图识别和改写不需要全量历史，截断能显著降低延迟。
        """
        rows = db().get_messages(self.chat_session_id, limit=limit)
        lines = []
        for role, content in rows:
            lines.append(f"{role}: {content}")
        return "\n".join(lines)


# 第二题：对话历史管理器
