"""全局配置：改这里就能改行为，不用翻业务代码。

原则：只放两类参数 ——
  1. 跨多个文件重复出现的（模型名等）
  2. 需要按场景调的（chunk_size、min_score 等）
只在某个函数里用一次的局部常量留在原处，避免「配置地狱」。"""
from dataclasses import dataclass
from pathlib import Path

# 项目根目录（AI_project）；不管从哪个目录运行都算得对
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class QaConfig:
    """
    全局配置：改这里就能改行为，不用翻业务代码。

    原则 —— 只放两类参数：
      1. 跨多个文件重复出现的（比如模型名，原来散在 5 个文件里）
      2. 需要按场景调的（比如 chunk_size、min_score）
    只在某个函数里用一次的局部常量（比如流式读的 4096、并发的 max_workers）
    留在原处，不进这里，否则会变成「配置地狱」。
    """

    # ===== 模型 =====
    chat_model: str = "deepseek-v4-flash-0731"      # 意图识别 / 改写 / 生成 / 摘要
    rerank_model: str = "qwen3-rerank"              # 精排
    embedding_model: str = "text-embedding-v1"      # 向量化

    # ===== 文本切分 =====
    chunk_size: int = 200
    chunk_overlap: int = 20

    # ===== 检索 =====
    recall_k: int = 8                               # 每路召回的条数（向量 / BM25 各取这么多）
    rrf_k: int = 60                                 # RRF 融合常数，论文默认 60
    # 向量权重略高于 BM25：BM25 会因高频词（如「请假」「工资」）把错误片段顶上来，
    # 实测 0.7/0.3 比等权好 —— 员工手册从 7/9 恢复到 8/9，技术手册保持 8/8
    dense_weight: float = 0.7
    bm25_weight: float = 0.3
    rerank_top_n: int = 3                           # 精排后保留条数
    min_score: float = 0.3                          # 相关性下限，低于它丢弃

    # ===== 记忆 =====
    history_limit: int = 6                          # 喂给模型的最近消息条数
    # 记忆落库路径：历史 / 事实槽 / 摘要都存在这个 SQLite 文件里，进程重启不丢。
    # 想换成 Redis，只改 agent/persistence.py，上层三个 Manager 不用动。
    memory_db_path: str = str(_PROJECT_ROOT / "runtime" / "memory.db")

    # ===== 向量库 =====
    # 用绝对路径，避免 "../" 把数据建到项目外面去
    chroma_root_dir: str = str(_PROJECT_ROOT / "indexing" / "chroma_db")
    collection_prefix: str = 'pdf_qa'
