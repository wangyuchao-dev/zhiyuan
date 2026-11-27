"""结构化日志与请求上下文。

两条设计约束：
  1. **生产输出 JSON**，测试/开发输出可读文本 —— 同一份代码，不靠人记得切换。
  2. **request_id 贯穿一次请求**：用 contextvar，业务代码不用一路透传参数，
     日志里自动带上；出问题时按 id 一把捞全链路。
"""

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar

# 一次请求的追踪 ID（中间件写入，日志格式化器读取）
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """一行一条 JSON：便于 Loki / ELK 采集，不用再写正则解析。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        # 允许 logging.info("...", extra={"route": "rag"}) 这样附加业务字段
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """开发期用：人眼扫得快，带 request_id 前 8 位。"""

    def format(self, record: logging.LogRecord) -> str:
        rid = request_id_var.get()
        rid = "-" if rid == "-" else rid[:8]
        base = (
            f"{time.strftime('%H:%M:%S', time.localtime(record.created))} "
            f"{record.levelname:<5} [{rid}] {record.name}: {record.getMessage()}"
        )
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging(env: str = "development", level: str = "INFO") -> None:
    """初始化根 logger。重复调用是安全的（幂等替换 handler）。"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if env == "production" else TextFormatter())

    root = logging.getLogger()
    root.handlers.clear()  # 避免 uvicorn 与自己的 handler 重复输出
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # uvicorn 自己的访问日志与我们的中间件重复，交给中间件统一记
    logging.getLogger("uvicorn.access").disabled = True
    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True


def new_request_id() -> str:
    return uuid.uuid4().hex
