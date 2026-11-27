"""查询改写：把「那这个呢」这类带指代的追问，还原成能独立检索的查询。

只在问题含指代词时才调用（见 Agent._need_rewrite），省一次模型调用。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from config.prompts import REWRITE_PROMPT

class QueryRewriter:
    """查询改写器：消除指代，生成独立检索查询"""

    def __init__(self, llm) -> None:
        self._llm = llm
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", REWRITE_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "用户问题：{question}\n改写后的查询：")
        ])
        self._chain = self._prompt | llm | StrOutputParser()

    def rewrite(
            self,
            question: str,
            chat_history: Optional[List[BaseMessage]] = None
    ) -> str:

        """返回改写后的独立查询；问题本身已清晰时模型会原样返回"""
        question = (question or "").strip()
        chat_history = chat_history or []

        rewrite = self._chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
        rewrite = rewrite.strip()
        return rewrite
