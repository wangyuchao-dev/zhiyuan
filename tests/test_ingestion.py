"""入库冒烟测试：把 uploads/ 下的 PDF 全走一遍建库流程。

这是脚本式测试（不是单元测试），用来快速确认「能不能入库」。
逻辑级回归测试见 tests/test_core.py。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
入库冒烟测试：把 uploads/ 下的 PDF 全部走一遍「算 hash → 切分 → 向量化 → 入库」。

用途：改完建库代码后跑一次，确认没坏。
用法：python tests/test_ingestion.py
"""

from config.setting import QaConfig
from indexing.indexing_pipeline import IndexingPipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")


def main():
    """遍历 uploads/ 下的 PDF，逐个建库并打印结果"""
    if not os.path.isdir(UPLOAD_DIR):
        print(f"没有 {UPLOAD_DIR} 目录，先上传一份文档")
        return

    pdfs = sorted(f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf"))
    if not pdfs:
        print("uploads/ 下没有 PDF")
        return

    pipeline = IndexingPipeline(QaConfig())
    print(f"共 {len(pdfs)} 份文档：\n")

    for name in pdfs:
        path = os.path.join(UPLOAD_DIR, name)
        try:
            file_hash = pipeline.build_from_source(path)
            print(f"  ✓ {name}\n    hash = {file_hash}")
        except Exception as e:
            print(f"  ✗ {name}\n    {type(e).__name__}: {e}")

    print("\n完成。重复执行不会重复向量化（文件 hash 做幂等）。")


if __name__ == "__main__":
    main()
