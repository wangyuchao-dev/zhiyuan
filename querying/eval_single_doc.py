# -*- coding: utf-8 -*-
"""
单文档三题验证：跑真实链路（召回 → 精排 → 阈值过滤），把每一题的分数打出来。

用途：
    产品说明 6.1 那一节写的是「精排最高分 + 返回几条」，
    这两个数必须在**阈值生效之后**重新测一遍 —— 老数字是阈值还没实现时测的，已经过时。

它和 eval_retrieval.py 的分工：
    eval_retrieval.py  → 34 题，只测召回层（不调大模型），用来比较「纯向量 vs 混合检索」
    eval_single_doc.py → 3 题，测完整检索链路（含精排与阈值），用来写文档里的分数

跑法：
    python querying/eval_single_doc.py
    python querying/eval_single_doc.py --pdf "D:/企业文档/员工手册.pdf"

需要环境变量 DASHSCOPE_API_KEY（向量模型与精排模型都要调用）。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.embeddings import DashScopeEmbeddings

from config.setting import QaConfig
from indexing.hash_calculate import HashCalculate
from indexing.vector_storage import VectorStoreManager
from querying.hybrid_retriever import HybridRetriever
from querying.retrank import RerankPipeline

# 三题：前两题文档里有答案，第三题文档里没有（用来验证阈值是否真的挡住）
QUESTIONS = [
    ("年假有几天", "有"),
    ("请假要怎么申请", "有"),
    ("年终奖怎么算", "无"),
]


def main() -> None:
    """跑三题真实链路，打印召回条数、精排分数与过阈值条数"""
    parser = argparse.ArgumentParser(description="单文档三题验证（含精排与阈值）")
    parser.add_argument("--pdf", default=r"D:/企业文档/员工手册.pdf", help="要测的 PDF 路径")
    parser.add_argument("--recall-k", type=int, default=8, help="每路召回条数")
    parser.add_argument("--top-k", type=int, default=3, help="精排后保留条数")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"找不到文档：{args.pdf}")
        return

    cfg = QaConfig()
    print(f"文档：{args.pdf}")
    print(f"阈值 min_score = {cfg.min_score}　召回 recall_k = {args.recall_k}　保留 top_k = {args.top_k}")
    print("-" * 78)

    emb = DashScopeEmbeddings(model=cfg.embedding_model)
    file_hash = HashCalculate().compute_hash_from_file(args.pdf)
    vs = VectorStoreManager(cfg, emb).load_or_build(file_hash)   # 已入库会直接复用
    hybrid = HybridRetriever(vs, cfg)
    rerank = RerankPipeline(cfg)

    rows = []
    for question, has_answer in QUESTIONS:
        recalled = hybrid.retrieve(question, k=args.recall_k)
        evidence, context = rerank.rerank(query=question, recalled_docs=recalled,
                                          final_top_n=args.top_k)
        top = evidence[0].score if evidence else 0.0
        scores = "、".join(f"{e.score:.4f}" for e in evidence) or "（全部被阈值滤掉）"
        rows.append((question, has_answer, len(recalled), top, len(evidence), scores))
        print(f"问题：{question}")
        print(f"  文档里有答案：{has_answer}")
        print(f"  召回 {len(recalled)} 条 → 过阈值 {len(evidence)} 条")
        print(f"  精排分数：{scores}")
        print(f"  最高分：{top:.4f}　→　{'✅ 保留' if evidence else '❌ 被 0.3 阈值滤掉'}")
        print()

    print("=" * 78)
    print("可直接写进产品说明的结论：")
    for question, has_answer, n_recall, top, n_keep, _ in rows:
        if has_answer == "无" and n_keep > 0:
            print(f"  · 「{question}」文档里没有，但精排最高分 {top:.4f} 仍高于阈值 "
                  f"{cfg.min_score}，未被滤掉 —— 说明 0.3 对这个库偏松，0.4 更合适。")
        elif has_answer == "无" and n_keep == 0:
            print(f"  · 「{question}」文档里没有，已被阈值滤掉（最高分 {top:.4f}）—— 0.3 生效且够严。")
    print("  注：分数会随模型版本与索引内容变化，写进文档时注明测量日期。")


if __name__ == "__main__":
    main()
