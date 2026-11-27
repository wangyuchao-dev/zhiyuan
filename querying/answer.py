"""答案生成器：按 ANSWER_PROMPT 约束，只依据上下文作答并标注引用编号。

提供 generate()（一次性）与 stream()（流式）两条路径，入参结构共用 _payload()。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config.prompts import ANSWER_PROMPT

# 职责：基于检索到的上下文，生成最终答案（带引用标注）


class AnswerGenerator:
    """答案生成器：根据问题、对话历史、检索上下文生成答案"""

    def __init__(self, llm: ChatTongyi):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ANSWER_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.__llm = prompt | llm | StrOutputParser()

    def generate(
        self, question: str, chat_history: list[BaseMessage], context: str, skills: str = ""
    ) -> str:
        """
        生成答案（一次性返回）
        :param question: 用户原始问题
        :param chat_history: 对话历史对象
        :param context: 格式化后的上下文字符串
        :param skills: 命中技能的操作指引（空串表示没有命中技能）
        :return: 答案字符串
        """
        return self.__llm.invoke(self._payload(question, chat_history, context, skills))

    def stream(
        self, question: str, chat_history: list[BaseMessage], context: str, skills: str = ""
    ):
        """
        流式生成答案（逐 token 产出），供 SSE 接口使用。

        用的是 LangChain 链自带的 .stream()：模型侧本来就是流式返回的，
        以前用 .invoke() 只是把它攒成一个字符串再吐出来，白等了首字延迟。
        """
        for chunk in self.__llm.stream(self._payload(question, chat_history, context, skills)):
            if chunk:
                yield chunk

    @staticmethod
    def _payload(
        question: str, chat_history: list[BaseMessage], context: str, skills: str = ""
    ) -> dict:
        """一次性 / 流式两条路径共用的入参，避免两处写不一样的 key"""
        return {
            "input": question,
            "chat_history": chat_history,
            "context": context,
            # 没命中技能时给一句明确的「无」，而不是空串 ——
            # 空串会让模型以为占位符没填上，有时会自己编一段流程出来
            "skills": skills or "（本次没有命中的技能，按通用要求作答）",
        }
