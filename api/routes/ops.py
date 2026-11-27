"""运维端点：存活检查、就绪检查、版本。

三个不要混：
  · **liveness**（/health）只回答「进程还在吗」。依赖挂了它也应该是 200，
    否则编排系统会不停重启一个本来没坏的进程。
  · **readiness**（/ready）回答「现在能接流量吗」。缺 Key、向量库没挂上 → 503，
    让负载均衡先把流量摘走，而不是把错误返回给用户。
  · **version** 给发布与问题排查用。
"""

import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.deps import get_config
from config.version import __version__

router = APIRouter(prefix="/api", tags=["ops"])


@router.get("/health", summary="存活检查")
def health():
    """不碰任何外部依赖：进程能响应就算活着。"""
    return {"status": "ok"}


@router.get("/ready", summary="就绪检查")
def ready():
    """检查「能不能开工」：密钥、向量库目录、运行目录。

    刻意不真的调一次模型：就绪检查会被负载均衡每几秒打一次，
    每次都调模型既费钱又会被限流。这里只检查必要条件。
    """
    cfg = get_config()
    checks = {}

    checks["api_key"] = bool(os.getenv("DASHSCOPE_API_KEY"))
    checks["chroma_dir"] = os.path.isdir(cfg.chroma_root_dir)
    checks["runtime_dir"] = os.path.isdir(os.path.dirname(cfg.memory_db_path))
    checks["env"] = cfg.env

    ok = checks["api_key"] and checks["chroma_dir"]
    body = {"status": "ready" if ok else "not_ready", "version": __version__, "checks": checks}
    return JSONResponse(body, status_code=200 if ok else 503)


@router.get("/version", summary="版本")
def version():
    return {"name": "zhiyuan", "version": __version__}
