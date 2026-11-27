"""向量库管理：一份 PDF 一个 Chroma 集合，集合名 = 文件指纹。

建库时显式指定 hnsw:space=cosine —— L2 距离在 1536 维里没有上界，
无法定义「多近算相关」；cosine 固定在 0~2，阈值才有意义。"""
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

from config.setting import QaConfig


class VectorStoreManager:
    """向量库管理（企业主流：一个库，一个 PDF 一个集合）"""

    def __init__(self, config: QaConfig, embeddings: DashScopeEmbeddings) -> None:
        self._config = config
        self._embeddings = embeddings
        self._persist_dir = Path(self._config.chroma_root_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _count(vs: Chroma) -> int:
        """
        取集合里已有的向量条数。

        不用 vs._collection.count() 直接读私有属性 —— 那是 Chroma 内部实现，
        版本一升就可能改名。这里做一层兜底：公开 API 失败时再退回私有属性。
        """
        try:
            return vs._collection.count()
        except AttributeError:
            try:
                return len(vs.get(include=[])["ids"])
            except Exception:
                return 0

    def load_or_build(self, file_hash: str, chunks: Optional[List[Document]] = None) -> Chroma:

        """取集合：有数据直接复用；为空且给了 chunks 就建；为空又没 chunks 按空库返回"""
        persist_dir = str(self._persist_dir)

        vs = Chroma(
            persist_directory=persist_dir,
            collection_name=file_hash, # 集合名字就是文档的hash值
            embedding_function=self._embeddings,
            collection_metadata={"name": "AI文档", "hnsw:space": "cosine"}
        )

        if self._count(vs) > 0:
            return vs

        if chunks is None:
            # 走到这里说明：集合存在但里面是空的（上次入库中途失败留下的），
            # 而调用方只是来「查」的，手里没有 chunks。
            # 以前这里直接抛 ValueError，导致这一份文档之后每次提问都 500；
            # 现在退化成「空库」返回，检索拿到空结果，回答变成「未找到相关资料」，
            # 前端还能正常工作，用户重新上传一次就会自动建好。
            print(f"[vectorstore] 集合 {file_hash[:8]} 为空且未提供 chunks，按空库处理")
            return vs

        vs.add_documents(
            documents=chunks,
            ids=[f"doc-{idx}" for idx in range(1, len(chunks) + 1)]
        )

        return vs
