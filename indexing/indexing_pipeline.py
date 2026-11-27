"""离线建库流水线：算指纹 → 清洗切分 → 向量化入库。

三步串起来对外只暴露 build_from_source()，接口层不用关心细节。

清洗是硬门槛，不是「尽力而为」：清洗后字数太少、或丢了 70% 以上，
说明这份 PDF 本身有问题（扫描件 / 加密 / 乱码），直接拒收。
放进去的代价比拒收大得多 —— 脏数据一旦进向量库，后面所有评测都是白跑。
"""

from langchain_community.embeddings import DashScopeEmbeddings

from config.setting import QaConfig
from indexing.clean import quality_gate_stats
from indexing.hash_calculate import HashCalculate
from indexing.upload_split import PdfIngestor
from indexing.vector_storage import VectorStoreManager


class IngestionRejected(ValueError):
    """文档质量不达标，拒绝入库（扫描件 / 加密 / 清洗后几乎没剩下内容）。"""


class IndexingPipeline:
    """索引构建流水线：PDF → 清洗切分 → Embedding → 持久化 Chroma"""

    def __init__(self, config: QaConfig) -> None:
        self._config = config
        self._hash_calculator = HashCalculate()
        self._ingestor = PdfIngestor(self._config)
        self._vector_store = VectorStoreManager(
            self._config,
            DashScopeEmbeddings(
                model=self._config.embedding_model,
            ),
        )

    def build_from_source(self, pdf_source: str) -> str:
        """算指纹 → 清洗切分 → 入库；已入过库的会复用，返回文件指纹"""
        file_hash = self._hash_calculator.compute_hash_from_file(pdf_source)

        chunks, stats = self._ingestor.ingest_with_stats(pdf_source)
        ok, reason = quality_gate_stats(stats)
        if not ok:
            raise IngestionRejected(reason)

        self._vector_store.load_or_build(file_hash, chunks)
        return file_hash
