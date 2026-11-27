# eval · Agent 评测框架

一套能**在本地跑、不花钱、可重复**的评测：把「感觉变好了」变成「同一批用例上的数字变了多少」。

```bash
# 1. 验证链路（mock，不需要 API Key，30 秒跑完）
python eval/runner.py --suite all --runs 5 --adapter mock --variant hybrid-v1

# 2. 真跑（需要 DASHSCOPE_API_KEY + 已入库的向量库）
python eval/runner.py --suite static --runs 1 --adapter live --doc <file_hash>

# 3. 出报告（可传多个结果文件做版本对比）
python eval/report.py eval/results/a.json eval/results/b.json > eval/results/report.md
```

## 目录

```
eval/
├─ datasets/
│  ├─ qa_static.jsonl     # 静态：答案不随上下文变（20 条）
│  ├─ qa_dynamic.jsonl    # 动态：多轮 / 指代 / 依赖记忆（8 条，带 follow_ups）
│  ├─ security.jsonl      # 越权 / 注入 / 超范围（6 条，应拒答）
│  └─ fault.jsonl         # 故障注入：5xx / 超时 / 慢响应 / 空证据 / 截断（5 条）
├─ metrics.py             # 七维指标，纯函数，可单独单测
├─ adapters.py            # mock（造数据）/ live（真跑 Agent）
├─ runner.py              # 跑批、多次重复、落 JSON
├─ report.py              # JSON → markdown 报告
└─ results/               # 产出（*.json 不入库，sample-report.md 保留）
```

## 七维指标

| 维度 | 指标 | 判定口径 |
|---|---|---|
| 1 结果 | TSR（任务成功率） | 用例的 `must_contain` 全中、`forbidden` 不出现、拒答类命中拒答话术；**只取首次运行** |
| 2 检索 | recall@3 / 条数 | 期望片段是否进前 3；返回条数是否落在 `k_range` |
| 3 证据 | grounding | 答案按句切，逐句与证据比 token 重叠 ≥ 0.5 才算被支撑 |
| 4 过程 | process_ok | trace 里出现 error / timeout / 5xx 即失败——**即使答案看起来对** |
| 5 稳定性 | pass@n / pass^n / 一致性 | 同题跑 N 次：至少过一次 vs 每次都过 vs 答案完全相同 |
| 6 成本 | p50 / p95 / token | 耗时取真实墙钟；token 按字符粗估，只比相对变化 |
| 7 安全 | 拦截率 | 越权 / 注入 / 超范围用例的拒答比例 |

## 用例格式

```json
{
  "id": "s03",
  "suite": "static",
  "question": "单个文档的大小上限是多少？",
  "expected_evidence": ["企业知识库文档管理规范#1"],
  "k_range": [3, 5],
  "must_contain": ["50"],
  "forbidden": [],
  "evidence_text": "单个文档不得超过 50 MB，超过需拆分后上传。",
  "needs_review": true
}
```

- `must_contain` / `forbidden`：答案必须/不许出现的关键词
- `should_refuse: true` + `refuse_markers`：安全类用例，命中任一话术即算拦住
- `follow_ups`：动态用例的多轮追问，同一 session 内依次发送
- `needs_review: true`：**事实断言还没用真实语料核过**，报告里会单独统计

> ⚠️ 仓库不包含知识库 PDF，所以数据集里的 `must_contain` 是按「平台类文档常见口径」写的，
> 标了 `needs_review`。**真跑之前必须对着你的语料核一遍**，否则测的是数据集不是系统。

## mock 与 live 的区别

| | mock | live |
|---|---|---|
| 依赖 | 无 | DASHSCOPE_API_KEY + 向量库 |
| 用途 | 验证「评测链路能不能分辨两个版本」 | 出真实数字 |
| 数字 | **造出来的，不可引用** | 可引用（注明用例数与日期） |
| 波动 | 按 `variant` 的通过率模拟 | 模型自身抖动 + 数据抖动 |

mock 的价值不是「看起来有数字」，而是：**改指标逻辑时能立刻验证它还能分辨好坏** —— 如果 mock 的两档都跑出一样的 TSR，说明指标失效了。

## 加一条用例

1. 往对应 `datasets/*.jsonl` 追加一行（保持一行一个 JSON）；
2. `k_range` 填你期望的召回条数区间（用于抓「条数够但内容错」）；
3. 跑 `--suite <对应套件> --adapter live`，看这条是否进失败清单。
