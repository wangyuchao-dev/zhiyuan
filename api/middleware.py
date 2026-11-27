"""HTTP 中间件：请求 ID + 访问日志 + 耗时。

**为什么不用 Starlette 的 BaseHTTPMiddleware**：
它会在应用外面再包一层，流式响应（本项目 /api/chat/stream 的 SSE）
可能被攒起来一起发，前端的打字机效果就没了。
纯 ASGI 中间件只包 send，不碰 body，对 SSE 安全。
"""

import logging
import time

from core.logging import new_request_id, request_id_var

log = logging.getLogger("api.access")


class RequestContextMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        raw = dict(scope.get("headers") or {})
        rid = (raw.get(b"x-request-id") or b"").decode("latin-1") or new_request_id()
        token = request_id_var.set(rid)
        start = time.perf_counter()
        status = {"code": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                # 把 request_id 回给调用方，前端/网关出问题时报这个 id 就能定位
                message.setdefault("headers", [])
                message["headers"].append((b"x-request-id", rid.encode("latin-1")))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            log.exception("请求未捕获异常")
            raise
        finally:
            log.info(
                "access",
                extra={
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status": status["code"],
                    "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                },
            )
            request_id_var.reset(token)
