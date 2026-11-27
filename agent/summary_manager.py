"""增量摘要：旧摘要 + 新消息 → 新摘要，长度不随对话轮次增长。

滚动压缩而不是全量重压；按阈值触发（见 Memory.SUMMARY_TRIGGER），
因为每压一次就是一次完整的模型调用。摘要落 SQLite。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 会话摘要管理器：负责生成和存储用户会话中的对话摘要
#
# 存储位置：SQLite（agent/persistence.py）。每会话只保留一条最新摘要，
# 采用「滚动压缩」：旧摘要 + 新消息 → 新摘要，长度不随轮次增长。
from typing import List

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config.prompts import SUMMARY_PROMPT

from agent.persistence import db

str_output_parser = StrOutputParser()


class SummaryManager:
    """负责会话摘要的生成与存储，仅使用大模型生成摘要"""

    def __init__(self, chat_session_id: str) -> None:
        self.chat_session_id = chat_session_id

    # ---- 摘要的读写（Memory 通过这两个方法和它对接）----
    def get(self) -> str:
        """读本次会话当前摘要；没有则返回空串"""
        return db().get_summary(self.chat_session_id)

    def set(self, summary: str) -> None:
        """写摘要（覆盖）。清空会话时传空串即可。"""
        db().set_summary(self.chat_session_id, summary or "")

    def generate_incremental_summary(
        self,
        old_summary: str,
        new_messages: List[BaseMessage],
        llm: ChatTongyi
    ) -> str:
        """
        生成增量摘要，依赖传入大模型。

        要点：把「旧摘要」和「这一批新消息」一起喂给模型，产出新的摘要。
        这样每次只处理「一小段 + 几条新的」，上下文长度不随对话轮次增长；
        而不是每轮都把全量历史重新压一遍（那样又慢又贵）。
        """
        if not new_messages:
            return old_summary

        messages_text = ""
        for msg in new_messages:
            if isinstance(msg, HumanMessage):
                messages_text += f"human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                messages_text += f"ai: {msg.content}\n"

        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", SUMMARY_PROMPT),
            ("human", f"旧摘要：{old_summary}\n新消息：{messages_text}")
        ])
        chain = summary_prompt | llm | str_output_parser
        result = chain.invoke(input={})
        new_summary = result.strip()
        return new_summary


# 第四题：会话摘要管理器：负责生成和存储用户会话中的对话摘要
