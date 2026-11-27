"""PDF 加载与切分：用正则后行断言在句号/问号/分号后切，保证不腰斩句子。

按换行硬切会把「年假须提前三个工作日提交」切成两半，两半都检索不到完整语义。"""

from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.setting import QaConfig


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

    def ingest(self, pdf_source: str) -> list[Document]:
        """从 PDF 文件路径加载并切分文档"""
        # 比如：本地文档、网上文档
        # if 本地文档 调用本地加载
        # if 网上文档 调用Web加载
        return self._ingest_from_file(pdf_source)

    # PDF 加载器
    def _ingest_from_file(self, file_path: str) -> list[Document]:
        """从本地文件加载 PDF"""
        path = Path(file_path)
        documents = PyMuPDFLoader(str(path)).load()
        return self._split_documents(documents)

    # 递归分割器
    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """切分文档为文本块"""
        chunks = self._splitter.split_documents(documents)
        return chunks
