"""离线建库流水线：算指纹 → 加载切分 → 向量化入库。

三步串起来对外只暴露 build_from_source()，接口层不用关心细节。"""

from langchain_community.embeddings import DashScopeEmbeddings

from indexing.hash_calculate import HashCalculate
from indexing.upload_split import PdfIngestor
from config.setting import QaConfig
from indexing.vector_storage import VectorStoreManager


class IndexingPipeline:
    """索引构建流水线：PDF → 切分 → Embedding → 持久化Chroma"""

    def __init__(self, config: QaConfig) -> None:
        self._config = config
        self._hash_calculator = HashCalculate()
        self._ingestor = PdfIngestor(self._config)
        self._vector_store = VectorStoreManager(self._config, DashScopeEmbeddings(
            model=self._config.embedding_model,
        ))

    def build_from_source(self, pdf_source: str):
        # 1. 计算文件哈希
        """算指纹 → 切分 → 入库；已入过库的会复用，返回文件指纹"""
        file_hash = self._hash_calculator.compute_hash_from_file(pdf_source)

        # 2. 加载并切分文档
        chunks = self._ingestor.ingest(pdf_source)

        # 3. 构建向量库
        self._vector_store.load_or_build(file_hash, chunks)

        return file_hash