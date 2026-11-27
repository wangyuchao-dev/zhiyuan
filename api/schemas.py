"""
接口的数据契约（Pydantic 模型）。

为什么单独一个文件：
    1. FastAPI 靠它做自动校验 —— 少字段/类型错立刻返回 422，不用在业务里手写 if；
    2. 自动生成 OpenAPI 文档（/docs 页面），前后端对字段时有唯一依据；
    3. 前端要展示哪些信息，看这个文件就够了，不用翻路由代码。

命名约定：
    XxxRequest  = 请求体
    XxxResponse = 响应体
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from pydantic import BaseModel


class ChatRequest(BaseModel):
    """一次提问：必须带会话 ID（记忆按它隔离）和问题文本"""

    session_id: str
    question: str


class EvidenceItem(BaseModel):
    """
    一条证据片段。前端把它渲染成可展开的「原文出处」。

    content 只回传前 200 字（见 routes/chat.py 的截断）：
    完整片段可能有上千字，全部塞给前端既没必要也拖慢响应；
    需要全文时前端按 source + page 去原文里定位。
    """

    content: str
    source: str
    page: int | None = None
    score: float  # 精排相关性，0~1；低于 QaConfig.min_score 的不会出现在这里


class ChatResponse(BaseModel):
    """一次问答的完整结果"""

    answer: str
    intent: str  # rag / chitchat / complaint / error
    confidence: float  # 规则判定是固定 0.9，模型判定用模型自评
    route: str  # 实际走的分支，便于排查误判
    search_query: str | None = None  # 若发生改写，这里是改写后的检索词
    evidence: list[EvidenceItem] = []


class DocumentUploadResponse(BaseModel):
    """上传结果"""

    file_hash: str
    filename: str
    message: str
    # 当前会话累计绑定了多少份文档（多文档检索：上传多份即可跨文档提问）
    bound_count: int = 1


class ResetResponse(BaseModel):
    """清空会话的结果"""

    ok: bool
    message: str
