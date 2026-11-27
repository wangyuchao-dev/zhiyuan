"""意图识别：把一句自然语言变成「意图 + 槽位 + 置信度」。

整条链路里只有这一步需要理解自然语言，其余都是确定性代码，
所以单独抽成一个类，好测试、好替换。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 意图识别器：通过大语言模型识别用户输入的意图，输出结构化数据
from typing import Any

from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config.prompts import INTENT_PROMPT


class IntentResult(BaseModel):
    """意图识别结果：通过大语言模型识别用户输入的意图，输出结构化数据。"""

    intents: list[str] = Field(description="意图列表，每个元素为一个意图名称")
    slots: dict[str, Any] = Field(description="slot值字典，键为slot名称，值为slot值")
    confidence: float = Field(description="置信度分数，取值范围0~1，越大表示越确信该意图")


class IntentRecognizer:
    """
    意图识别器：通过大语言模型识别用户输入的意图，输出 IntentResult 对象
    """

    def __init__(self, llm: ChatTongyi):
        self.__prompt = ChatPromptTemplate.from_messages(
            [
                ("system", INTENT_PROMPT),
                ("ai", "上下文内容: {chat_history}"),
                ("human", "用户输入: {user_input}"),
            ]
        )
        self.__llm = llm.with_structured_output(IntentResult)
        self.__chain = self.__prompt | self.__llm

    def recognize(self, user_input: str, chat_history: str | None = None) -> IntentResult:
        """识别一句话的意图与槽位；传 chat_history 能显著提高短回答的判定准确率（如「大连甘井子」）"""
        chat_history = chat_history if chat_history else ""
        result = self.__chain.invoke(input={"chat_history": chat_history, "user_input": user_input})
        if result is None:
            result = IntentResult(intents=["general"], slots={}, confidence=0.0)
        return result
