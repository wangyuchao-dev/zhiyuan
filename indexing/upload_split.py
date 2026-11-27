"""PDF 加载与切分：清洗 → 按句号/问号/分号后切，保证不腰斩句子。

按换行硬切会把「年假须提前三个工作日提交」切成两半，两半都检索不到完整语义。

清洗放在切分之前（这里就是唯一正确的位置）：
页眉页脚是「跨页重复」的，切完就只剩碎片，再也认不出来。
"""

from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.setting import QaConfig
from indexing.clean import clean_pages


class PdfIngestor:
    """PDF文档加载与切分"""

    def __init__(self, config: QaConfig) -> None:
        """初始化PDF加载器"""
        self._config = config
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
            is_separator_regex=True,
            separators=["(?<=。)", "(?<=！)", "(?<=？)", "(?<=；)", "(?<=，)", " ", ""],
        )
        # 最近一次清洗的统计，供建库流水线做质量门槛与日志
        self.last_clean_stats: dict = {}

    def ingest(self, pdf_source: str) -> list[Document]:
        """从 PDF 文件路径加载、清洗并切分（只要结果，不要统计）"""
        chunks, _ = self.ingest_with_stats(pdf_source)
        return chunks

    def ingest_with_stats(self, pdf_source: str) -> tuple[list[Document], dict]:
        """加载 → 清洗 → 切分，同时返回清洗统计。

        统计必须能带出去：入库了多少字、去掉多少页眉，是判断「这份 PDF 值不值得入库」
        的唯一依据，只留在日志里等于没有。
        """
        return self._ingest_from_file(pdf_source)

    # PDF 加载器
    def _ingest_from_file(self, file_path: str) -> tuple[list[Document], dict]:
        """从本地文件加载 PDF 并清洗"""
        path = Path(file_path)
        documents = PyMuPDFLoader(str(path)).load()

        # 逐页清洗（页眉按整篇识别，页码 metadata 保留）
        documents, stats = clean_pages(documents)
        self.last_clean_stats = stats

        return self._split_documents(documents), stats

    # 递归分割器
    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """切分文档为文本块"""
        chunks = self._splitter.split_documents(documents)
        # 清洗后为空的块没必要入库：它在向量空间里会随机命中任何问题
        return [c for c in chunks if (c.page_content or "").strip()]
