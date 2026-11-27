"""对话相关路由：/api/chat（一次性）、/api/chat/stream（SSE）、/api/reset、/api/health。

两个对话接口共用 Agent 的同一套路由逻辑，区别只在「怎么把答案吐出去」。
所有路由都做了异常兜底，不让外部服务抖动变成 500。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import StreamingResponse

from api.deps import get_agent
from api.schemas import ChatRequest, ChatResponse, EvidenceItem, ResetResponse

router = APIRouter(prefix="/api", tags=["chat"])


def _sse(payload: dict) -> str:
    """
    把一个事件包成 SSE 帧。

    SSE 的格式就是 `data: <一行文本>` 加空行；这里用 JSON 承载结构，
    前端 JSON.parse 后按 event 字段分支处理（meta / evidence / delta / done / error）。
    必须 ensure_ascii=False，否则中文会被转成 Unicode 转义码，前端虽能解但日志不可读。
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.get("/health")
def health():
    """健康检查：不依赖任何外部服务，用来确认进程活着"""
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """对话：意图识别 → 路由 → 检索/闲聊 → 记忆回写"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 兜底：模型超时 / 503 / token 超限 / prompt 渲染失败，任何一种都会抛异常。
    # 不兜的话用户只看到「服务异常」，不知道能不能重试；这里转成一条可读的 reply，
    # 同时把完整堆栈打进服务端日志，方便定位。
    try:
        result = get_agent().chat(req.session_id, req.question)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[chat] 处理失败 session={req.session_id}: {type(e).__name__}: {e}")
        return ChatResponse(
            answer="服务暂时不可用，请稍后重试。如果反复出现，把这个问题的截图反馈给管理员。",
            intent="error",
            confidence=0.0,
            route="error",
            search_query=None,
            evidence=[],
        )

    return ChatResponse(
        answer=result["answer"],
        intent=result.get("intent", ""),
        confidence=result.get("confidence", 0.0),
        route=result.get("route", ""),
        search_query=result.get("search_query"),
        evidence=[
            EvidenceItem(
                content=ev.content[:200],
                source=ev.source,
                page=ev.page,
                score=round(ev.score, 4),
            )
            for ev in result.get("evidence", [])
        ],
    )


@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    流式对话（SSE）。

    和 /api/chat 的区别只有一个：答案逐字吐出来，首字延迟从「等整段生成完」
    降到「第一个 token 到达」，体感差别很大。

    事件类型（前端按 event 字段分支）：
        meta     意图 / 路由 / 检索词
        evidence 证据列表
        delta    文本片段
        done     结束（此时服务端已把这一轮写回记忆）
        error    出错

    注意：不要再套一层 @router.post("/chat") 的 response_model —— 流式返回
    不是 JSON，走 StreamingResponse，media_type 必须是 text/event-stream。
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    def event_source():
        """把 Agent 的事件流包成 SSE 帧，异常与结束标记都在这里兜住"""
        try:
            for ev in get_agent().chat_stream(req.session_id, req.question):
                yield _sse(ev)
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield _sse({"event": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            # 显式结束标记：前端读到它就 close，不会一直等
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",       # 别让浏览器缓存流
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",         # 有反向代理时禁用缓冲，否则流会被攒住
        },
    )


@router.post("/reset", response_model=ResetResponse)
def reset(session_id: str = Form(...)):
    """清空某个会话的记忆与文档绑定"""
    try:
        get_agent().reset_session(session_id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=503, detail=f"重置失败：{type(e).__name__}")
    return ResetResponse(ok=True, message=f"会话 {session_id} 已重置")
