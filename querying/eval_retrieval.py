"""检索质量评测（召回层口径）：纯向量 vs 混合检索，34 题 / 4 份文档。

只测召回 top-1，不经过精排、不调大模型 —— 目的是把「两路召回」的差异单独衡量。
带精排与阈值的口径见 querying/eval_single_doc.py。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
检索质量评测：纯向量 vs 混合检索（向量 + BM25 + RRF）

四份文档、34 道题（21 精确编号类 + 13 语义类）：
  员工手册 9 / 费用报销管理办法 9 / 信息安全管理制度 8 / 技术手册（含 30 个错误码）8
判定：top-1 片段是否包含「期望关键词」。只看**召回层**，不经过精排，
      目的是让两路召回的差异能被单独衡量。
不调大模型 —— 只看检索层命中率，不受生成质量干扰。

脚本里每题都写明「问题 + 期望答案 + 所属文档 + 题目类型」，可复跑、可核查。

跑法：python querying/eval_retrieval.py
"""

from dataclasses import dataclass
from typing import List

from langchain_community.embeddings import DashScopeEmbeddings

from config.setting import QaConfig
from indexing.hash_calculate import HashCalculate
from indexing.vector_storage import VectorStoreManager
from querying.vector_retriever import recall_with_scores
from querying.hybrid_retriever import HybridRetriever


DOC_DIR = r'D://企业文档'
HR_PDF = os.path.join(DOC_DIR, '员工手册.pdf')
FIN_PDF = os.path.join(DOC_DIR, '费用报销管理办法.pdf')
SEC_PDF = os.path.join(DOC_DIR, '信息安全管理制度.pdf')
OPS_PDF = r'D://智云平台故障排查手册.pdf'   # 技术手册：含 30 个错误码，用于对比



@dataclass
class Case:
    """一道评测题：问题 / 期望关键词 / 所属文档 / 题型"""
    question: str
    expect: str      # top-1 里必须出现的关键词
    doc: str         # 'hr' / 'fin' / 'sec'
    kind: str        # 'exact' 精确编号类 / 'semantic' 语义类


CASES: List[Case] = [
    # ========== 员工手册 ==========
    Case('HR-2026-001 是什么文件', '员工手册', 'hr', 'exact'),
    Case('旷工一天扣几天工资', '旷工一天扣发三天工资', 'hr', 'exact'),
    Case('迟到多少分钟算迟到', '不超过三十分钟', 'hr', 'exact'),
    Case('单月迟到几次会扣全勤奖', '累计达到五次', 'hr', 'exact'),
    Case('年假有几天', '年休假五天', 'hr', 'exact'),
    Case('请假要怎么申请', '提前三个工作日', 'hr', 'semantic'),
    Case('工资什么时候发', '每月十五日发放', 'hr', 'semantic'),
    Case('产假休多久', '一百五十八天', 'hr', 'semantic'),
    Case('试用期员工适用这个手册吗', '试用期员工', 'hr', 'semantic'),

    # ========== 费用报销管理办法 ==========
    Case('FIN-2026-003 是什么文件', '费用报销管理办法', 'fin', 'exact'),
    Case('一线城市住宿费上限多少', '六百元', 'fin', 'exact'),
    Case('报销流程分几步', '报销流程共四步', 'fin', 'exact'),
    Case('五千元以上的报销谁审批', '分管副总审批', 'fin', 'exact'),
    Case('同一张发票能报两次吗', '不得重复报销', 'fin', 'exact'),
    Case('报销单最晚什么时候交', '三十日内提交', 'fin', 'semantic'),
    Case('出差要提前几天申请', '两个工作日', 'fin', 'semantic'),
    Case('对发票抬头有什么要求', '抬头须为公司全称', 'fin', 'semantic'),
    Case('虚开发票会怎么处理', '解除劳动合同', 'fin', 'semantic'),

    # ========== 信息安全管理制度 ==========
    Case('SEC-2026-005 是什么文件', '信息安全管理制度', 'sec', 'exact'),
    Case('密码最少要多少位', '不得少于十二位', 'sec', 'exact'),
    Case('密码多久要换一次', '九十天', 'sec', 'exact'),
    Case('公司数据分几级', '三级', 'sec', 'exact'),
    Case('机密级数据能发个人邮箱吗', '不得通过个人邮箱', 'sec', 'exact'),
    Case('账号可以借给同事用吗', '禁止将账号借予他人', 'sec', 'semantic'),
    Case('电脑能自己装软件吗', '禁止安装未经授权的软件', 'sec', 'semantic'),
    Case('发现安全事件第一时间做什么', '立即断开网络', 'sec', 'semantic'),
    # ========== 技术手册（错误码类，BM25 的主场）==========
    Case('E1017 是什么故障', '副本同步失败', 'ops', 'exact'),
    Case('E1024 报错怎么处理', '审计日志写入失败', 'ops', 'exact'),
    Case('E1013 是什么意思', '接口已被限流', 'ops', 'exact'),
    Case('E1029 是什么问题', '缓存击穿', 'ops', 'exact'),
    Case('E1020 告警', '存储节点离线', 'ops', 'exact'),
    Case('E1007 位点丢失怎么办', '增量位点丢失', 'ops', 'exact'),
    Case('接口限流阈值是多少', '每分钟六百次', 'ops', 'semantic'),
    Case('备份数据保留多少天', '保留三十天', 'ops', 'semantic'),
]


def top1_hit(content: str, expect: str) -> bool:
    """判定 top-1 片段是否包含期望关键词（忽略空白差异）"""
    flat = ''.join(content.split())
    return ''.join(expect.split()) in flat


def main():
    """跑完 34 题，按题型与文档类型分别打印命中率"""
    cfg = QaConfig()
    emb = DashScopeEmbeddings(model=cfg.embedding_model)

    stores = {}
    for key, pdf in [('hr', HR_PDF), ('fin', FIN_PDF), ('sec', SEC_PDF), ('ops', OPS_PDF)]:
        if not os.path.exists(pdf):
            print(f'缺少文档：{pdf}')
            return
        h = HashCalculate().compute_hash_from_file(pdf)
        vs = VectorStoreManager(cfg, emb).load_or_build(h)
        stores[key] = (vs, HybridRetriever(vs, cfg))

    print(f'{"":4}{"类型":10}{"纯向量":9}{"混合检索":11} 问题')
    print('-' * 82)

    stat = {'dense': 0, 'hybrid': 0}
    by_kind = {'exact': {'dense': 0, 'hybrid': 0, 'n': 0},
               'semantic': {'dense': 0, 'hybrid': 0, 'n': 0}}

    for i, c in enumerate(CASES, 1):
        vs, hybrid = stores[c.doc]

        d_docs, _ = recall_with_scores(c.question, vs, k=8, min_score=0.0)
        d_ok = bool(d_docs) and top1_hit(d_docs[0].page_content, c.expect)

        h_docs = hybrid.retrieve(c.question, k=8, rrf_top_n=8)
        h_ok = bool(h_docs) and top1_hit(h_docs[0].page_content, c.expect)

        stat['dense'] += d_ok
        stat['hybrid'] += h_ok
        by_kind[c.kind]['dense'] += d_ok
        by_kind[c.kind]['hybrid'] += h_ok
        by_kind[c.kind]['n'] += 1

        kind_cn = '精确编号' if c.kind == 'exact' else '语义'
        print(f'{i:2d}. {"✓" if d_ok else "✗":4}{kind_cn:10}'
              f'{"命中" if d_ok else "未中":9}{"命中" if h_ok else "未中":11} {c.question}')

    n = len(CASES)
    print('-' * 82)
    print(f'总计：纯向量 {stat["dense"]}/{n} = {stat["dense"]/n:.1%}　'
          f'混合检索 {stat["hybrid"]}/{n} = {stat["hybrid"]/n:.1%}')
    print()
    for kind in ('exact', 'semantic'):
        s = by_kind[kind]
        name = '精确编号类' if kind == 'exact' else '语义类'
        print(f'  {name}（{s["n"]} 题）：纯向量 {s["dense"]}/{s["n"]} = {s["dense"]/s["n"]:.0%}'
              f'　→　混合 {s["hybrid"]}/{s["n"]} = {s["hybrid"]/s["n"]:.0%}')

    # 按文档类型分组：说明混合检索的价值和文档类型强相关
    print()
    print('按文档类型：')
    for doc_key, doc_name in [('hr', '员工手册'), ('fin', '费用报销办法'), ('sec', '信息安全制度'),
                              ('ops', '技术手册（含错误码）')]:
        items = [c for c in CASES if c.doc == doc_key]
        if not items:
            continue
        d = h = 0
        for c in items:
            vs, hybrid = stores[c.doc]
            dd, _ = recall_with_scores(c.question, vs, k=8, min_score=0.0)
            hd = hybrid.retrieve(c.question, k=8, rrf_top_n=8)
            d += bool(dd) and top1_hit(dd[0].page_content, c.expect)
            h += bool(hd) and top1_hit(hd[0].page_content, c.expect)
        print(f'  {doc_name:14s}（{len(items)} 题）纯向量 {d}/{len(items)} → 混合 {h}/{len(items)}')


if __name__ == '__main__':
    main()
