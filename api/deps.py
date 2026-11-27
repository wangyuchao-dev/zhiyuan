"""FastAPI 依赖注入：全局单例的懒加载。

为什么不写在模块顶层：那样 import 的一刻就会构造 ChatTongyi，
没配 DASHSCOPE_API_KEY 时连 /api/health 都返回不了，服务根本起不来。"""
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.setting import QaConfig
from agent.orchestrator import Agent
from indexing.indexing_pipeline import IndexingPipeline

# 全局单例：单 worker 部署用内存即可；多 worker 时这里的会话状态要挪到 Redis
#
# 为什么改成「懒加载」：
#   原来这三行写在模块顶层，等于 import 的那一刻就去构造 ChatTongyi 和 Chroma。
#   后果是只要环境里没有 DASHSCOPE_API_KEY（或者网络不通），连 /api/health
#   都返回不了 —— 服务根本起不来，也没法做健康检查定位问题。
#   现在改成第一次真正用到时才创建，进程可以正常启动。
_config: Optional[QaConfig] = None
_agent: Optional[Agent] = None
_ingestor: Optional[IndexingPipeline] = None


def get_config() -> QaConfig:
    """取全局配置（首次调用时创建）"""
    global _config
    if _config is None:
        _config = QaConfig()
    return _config


def get_agent() -> Agent:
    """取 Agent 单例（首次调用时才构造模型，避免没 Key 就起不来）"""
    global _agent
    if _agent is None:
        _agent = Agent(get_config())
    return _agent


def get_ingestor() -> IndexingPipeline:
    """取建库流水线单例"""
    global _ingestor
    if _ingestor is None:
        _ingestor = IndexingPipeline(get_config())
    return _ingestor
