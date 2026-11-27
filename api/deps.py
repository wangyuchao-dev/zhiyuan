"""FastAPI 依赖注入：全局单例的懒加载。

为什么不写在模块顶层：那样 import 的一刻就会构造 ChatTongyi，
没配 DASHSCOPE_API_KEY 时连 /api/health 都返回不了，服务根本起不来。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.orchestrator import Agent
from config.setting import QaConfig
from indexing.indexing_pipeline import IndexingPipeline

# 全局单例：单 worker 部署用内存即可；多 worker 时这里的会话状态要挪到 Redis
#
# 为什么改成「懒加载」：
#   原来这三行写在模块顶层，等于 import 的那一刻就去构造 ChatTongyi 和 Chroma。
#   后果是只要环境里没有 DASHSCOPE_API_KEY（或者网络不通），连 /api/health
#   都返回不了 —— 服务根本起不来，也没法做健康检查定位问题。
#   现在改成第一次真正用到时才创建，进程可以正常启动。
_config: QaConfig | None = None
_agent: Agent | None = None
_ingestor: IndexingPipeline | None = None


class ServiceNotReady(RuntimeError):
    """依赖没配好（最典型的是缺 DASHSCOPE_API_KEY）。

    单独定义异常，是为了和「跑起来以后才出错」区分开：
    前者应该返回 503 + 一句能照着做的提示，后者才走重试/降级。
    """


# 依赖构造失败的原始异常里，这些关键字说明是「配置缺失」而不是「代码 bug」
_CONFIG_HINTS = ("dashscope_api_key", "DASHSCOPE_API_KEY")


def _wrap(exc: Exception, what: str) -> ServiceNotReady:
    """把构造失败翻译成一句能照着做的提示，原始异常留在 __cause__ 里。"""
    msg = f"{what}初始化失败：{type(exc).__name__}: {exc}"
    if any(h in str(exc) for h in _CONFIG_HINTS):
        msg = (
            "服务未就绪：缺少环境变量 DASHSCOPE_API_KEY。"
            "在项目根目录的 .env 里填入百炼 API Key 后重启服务"
            "（或直接设置环境变量后启动）。"
        )
    return ServiceNotReady(msg)


def get_config() -> QaConfig:
    """取全局配置（首次调用时创建，按 APP_ENV 套用环境变量覆盖）"""
    global _config
    if _config is None:
        _config = QaConfig.load()
    return _config


def get_agent() -> Agent:
    """取 Agent 单例（首次调用时才构造模型，避免没 Key 就起不来）"""
    global _agent
    if _agent is None:
        try:
            _agent = Agent(get_config())
        except Exception as exc:  # 不吞异常：转成可读的「未就绪」
            raise _wrap(exc, "对话 Agent ") from exc
    return _agent


def get_ingestor() -> IndexingPipeline:
    """取建库流水线单例"""
    global _ingestor
    if _ingestor is None:
        try:
            _ingestor = IndexingPipeline(get_config())
        except Exception as exc:
            raise _wrap(exc, "建库流水线 ") from exc
    return _ingestor
