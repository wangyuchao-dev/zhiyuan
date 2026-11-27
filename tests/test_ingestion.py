"""入库冒烟：把 samples/ 下的 PDF 全走一遍「算 hash → 清洗 → 切分 → 向量化 → 入库」。

两种用法：
  · 脚本：python tests/test_ingestion.py        （打印每份的结果，人工看）
  · pytest：pytest tests/test_ingestion.py -v   （有指纹就断言，CI 没 Key 时自动跳过）

它需要真实的 embedding 调用，所以**不属于离线单测**：
没有 DASHSCOPE_API_KEY 时 pytest 会 skip，不会让 CI 变红。
逻辑级回归见 tests/test_core.py 与 tests/test_wiring.py。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from config.setting import QaConfig
from indexing.indexing_pipeline import IndexingPipeline, IngestionRejected

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "samples")

# 这几份是「故意做的坏文档」，预期被质量门槛拒收；进了库才是 bug
EXPECT_REJECTED = {"05-扫描件无文字层.pdf"}


def _sample_pdfs() -> list[str]:
    if not os.path.isdir(SAMPLE_DIR):
        return []
    return sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(".pdf"))


def ingest_all(verbose: bool = True) -> dict[str, str]:
    """逐份建库，返回 {文件名: 'ok' | 'rejected' | 'failed: ...'}"""
    pdfs = _sample_pdfs()
    if not pdfs:
        raise RuntimeError(f"samples/ 下没有 PDF（预期路径 {SAMPLE_DIR}）")

    pipeline = IndexingPipeline(QaConfig())
    results: dict[str, str] = {}
    for name in pdfs:
        path = os.path.join(SAMPLE_DIR, name)
        should_reject = name in EXPECT_REJECTED
        try:
            file_hash = pipeline.build_from_source(path)
            status = "ok"
            if verbose:
                print(f"  ✓ {name}\n    hash = {file_hash}")
                if should_reject:
                    print("    !! 这份本该被拒收，却入库了")
        except IngestionRejected as e:
            status = "rejected"
            if verbose:
                mark = "✓" if should_reject else "✗"
                print(f"  {mark} {name}\n    拒收（预期{'内' if should_reject else '外'}）：{e}")
        except Exception as e:  # 网络、Key、磁盘……
            status = f"failed: {type(e).__name__}: {e}"
            if verbose:
                print(f"  ✗ {name}\n    {status}")
        results[name] = status
    return results


@pytest.mark.skipif(
    not os.getenv("DASHSCOPE_API_KEY"),
    reason="需要真实 embedding 调用，CI 无 Key 时跳过（这不是离线单测）",
)
def test_samples_ingest_as_expected():
    """正常文档都能入库；标记为坏的文档必须被拒收"""
    results = ingest_all(verbose=False)

    for name, status in results.items():
        if name in EXPECT_REJECTED:
            assert status == "rejected", f"{name} 应被拒收，实际 {status}"
        else:
            assert status == "ok", f"{name} 应入库，实际 {status}"


def main():
    pdfs = _sample_pdfs()
    print(f"samples/ 共 {len(pdfs)} 份文档：\n")
    results = ingest_all()
    ok = sum(1 for v in results.values() if v == "ok")
    rejected = sum(1 for v in results.values() if v == "rejected")
    print(f"\n完成：入库 {ok} 份，拒收 {rejected} 份。重复执行不会重复向量化（hash 幂等）。")


if __name__ == "__main__":
    main()
