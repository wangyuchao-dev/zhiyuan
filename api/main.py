"""
FastAPI 应用装配层（只做三件事：建 app、加中间件、挂路由）。

设计约束：
    · 这个文件不写任何业务逻辑 —— 业务在 agent/ 与 querying/ 里，
      接口层只负责「收请求 → 调 agent → 转成响应结构」。
    · 依赖（Agent / IndexingPipeline）全部通过 api/deps.py 拿，且是懒加载，
      所以没配 DASHSCOPE_API_KEY 时进程也能起来，至少 /api/health 能自检。

启动：
    python api/main.py           # 默认 127.0.0.1:8000
    # 或用 uvicorn 起多 worker：
    # uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
    #   注意：多 worker 时进程内缓存的会话对象不共享，
    #   记忆已经落 SQLite 所以不丢，但 _session_docs（文档绑定）仍在进程内存里，
    #   多 worker 下建议把文档绑定也挪到库里，或改用 sticky session。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware import RequestContextMiddleware
from api.routes import chat, documents, ops
from config.setting import QaConfig
from config.version import __version__
from core.logging import setup_logging

_config = QaConfig.load()
setup_logging(env=_config.env, level=os.getenv("LOG_LEVEL", "INFO"))

app = FastAPI(
    title="知渊 · 企业知识智能中枢",
    description="RAG 知识库问答 Agent：意图识别 + 三层记忆（历史/事实/摘要）+ "
    "混合检索（稠密向量 + BM25 + RRF）+ 精排 + 阈值过滤，支持多文档与流式输出",
    version=__version__,
)

# 请求上下文中间件必须在 CORS 之前加（后加的先执行），
# 这样 4xx/5xx 的响应也能带上 X-Request-ID
app.add_middleware(RequestContextMiddleware)

# 开发期前端是 http://127.0.0.1:8080、后端是 8000，属于跨源，必须放行。
# 生产环境应把 allow_origins 收窄成具体域名，不要留 "*"。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# 路由挂载（三组职责边界：运维 / 文档离线建库 / 在线问答）
app.include_router(ops.router)  # GET /api/health、/api/ready、/api/version
app.include_router(documents.router)  # POST/GET/DELETE /api/documents
app.include_router(chat.router)  # POST /api/chat、/api/chat/stream、POST /api/reset

# 前端单页：挂在根路径，打开 http://127.0.0.1:8000 就是聊天界面。
# 必须放在 include_router 之后 —— mount("/") 会兜住所有未匹配路径，
# api 路由先注册就先匹配，不会被它抢走。
_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
if os.path.isdir(_WEB_DIR):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
