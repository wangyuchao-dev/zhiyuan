# 知渊 · 企业知识库问答

把几十页制度 PDF 变成「问一句话、答案带出处」的知识库。
核心不是把文档塞进向量库，而是**把检索做准**：意图先分流、双路召回、按名次融合、精排压噪、阈值兜底，每一步的收益都用同一套 34 题评测集量化过。

```
文档上传 → 切分 → 向量入库（cosine）
    ↓
提问 → 意图识别（规则快判优先）→ 查询改写（按需）
    ↓
双路召回：稠密向量 + BM25  →  RRF 按名次融合（0.7 / 0.3）
    ↓
精排（qwen3-rerank）→ 相关性阈值 0.3 → 拼上下文
    ↓
生成答案（标注出处条款）→ SSE 流式返回
```

## 效果

| 指标 | 纯向量基线 | 混合检索 | 说明 |
|---|---|---|---|
| 34 题命中率 | 28/34（82.4%） | **31/34（91.2%）** | 判定口径：召回首条命中期望关键词 |
| 编号密集型 | 5/8 | **8/8** | 纯向量召不回 `E1017` 这类纯符号，BM25 精确命中 |
| 语义类 | 11/13（85%） | 11/13（85%） | 持平——混合检索不是万灵药 |
| 典型问答耗时 | 7 s | **2.3 s** | 规则快判意图省一次模型调用 + 无指代词时跳过改写 |

单元测试 **36 个**，`python -m unittest discover -s tests` **不联网、不花钱、不依赖数据库**，0.3 秒跑完。

## 目录

```
config/       setting.py（全部可调参数）、prompts.py（6 块提示词）
indexing/     hash_calculate（MD5 指纹）· upload_split（PDF 切分）· vector_storage（写 Chroma）· indexing_pipeline
querying/     query_rewrite · vector_retriever · bm25_retriever · rrf · hybrid_retriever
              retrank（精排）· answer（生成）· pipeline（串联）· eval_retrieval / eval_single_doc（评测）
agent/        intent（意图）· chat_history_manager · facts_manager · summary_manager
              memory（中枢）· persistence（SQLite）· orchestrator（主流程）
              plan（显式计划 + 有界重试）· memory_policy（四类记忆与严格检索）
core/         errors（五类错误 + 重试策略）· context_budget（上下文分级预算）
tools/        registry（function calling 注册表）· mcp_adapter（MCP 描述互转）
skills/       registry（懒加载）· fault-code-lookup/SKILL.md（示例技能）
eval/         七维评测框架（见下方「评测」）
api/          main（装配）· deps（懒加载）· schemas · routes/chat · routes/documents
tests/        test_core（36 用例）· test_ingestion（入库冒烟）
web/          index.html（Vue 3 单页）
```

## 快速开始

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

set DASHSCOPE_API_KEY=sk-xxx                        # 对话 / embedding / 精排都走 DashScope
python api/main.py                                  # http://127.0.0.1:8000  （文档 /docs）

# 另开一个窗口起前端
python -m http.server 8080 --directory web          # http://127.0.0.1:8080
```

传一份 PDF 到 `uploads/`（或走 `POST /api/documents` 上传），系统会切分、向量化、写入 `indexing/chroma_db/`。
首次没有向量库也能起服务，`/api/health` 会告诉你缺什么。

> 仓库未附带知识库 PDF 与向量库（`.gitignore` 已排除），放任意 PDF 到 `uploads/` 即可复现。

## 四个我认为值得说的设计

### 1. 意图先用规则快判，判不了才调模型

`agent/intent.py` 里有三类意图：文档问答、闲聊、投诉。规则能判的直接走对应提示词（省一次模型调用，约 3 秒）；
判不了才让模型分类。

**为什么一个意图一块提示词**：投诉那条根本不需要查文档，走 RAG 反而会拿制度条款去回应情绪；文档问答必须严格要求「只依据材料回答」。

### 2. 三层记忆，各管一段

| 层 | 存什么 | 为什么单独一层 |
|---|---|---|
| 历史 | 最近若干轮原文 | 保证指代消解（「那它呢」能接上） |
| 事实 | 用户说过的稳定信息（部门、岗位） | 反复问同一件事不该反复说 |
| 摘要 | 更早对话的压缩摘要 | 长对话不爆上下文，又不完全丢信息 |

三张表落 SQLite，重启不丢；换 Redis 只改 `agent/persistence.py`。

### 3. 阈值 0.3 是「精排分」的阈值，不是向量相似度

建库时强制 **cosine 距离** —— 只有这样 0.3 才有明确含义（换 L2 之后同一个数值含义完全不同）。
精排分数低于 0.3 的片段直接丢，宁可答「材料里没有」，也不拿不相关段落凑上下文。

### 4. 多文档：一份文档一个集合，跨文档用 RRF 而不是分数相加

不同集合的相似度分数**不可比**（各自的分布不同），所以跨文档只能按**名次**融合 —— 这正是 RRF 的适用场景。

## 评测怎么做

`querying/eval_retrieval.py` 读 34 题（21 道精确编号类 + 13 道语义类），覆盖 4 份文档 42 个片段，
判定「召回首条是否包含期望关键词」，输出按题型分组的命中率。

> 只看总数会骗人：编号类从 5/8 到 8/8 是结构性改善，语义类持平说明混合检索解决的不是语义泛化问题。

## 评测（七维）

不报单一「准确率」，而是拆七维：**结果 / 检索 / 证据 / 过程 / 稳定性 / 成本 / 安全**。
用例 39 条（静态 20 / 动态 8 / 安全 6 / 故障 5），可离线跑、不花钱。

```bash
python eval/runner.py --suite all --runs 5 --adapter mock --variant hybrid-v1   # 验证链路
python eval/runner.py --suite static --adapter live --doc <file_hash>           # 真跑
python eval/report.py eval/results/a.json eval/results/b.json > report.md       # 版本对比
```

- 设计口径：[docs/eval-framework.md](docs/eval-framework.md)（七维怎么定义、静态/动态为什么分开、TSR 为什么只取首次）
- 跑法与用例格式：[eval/README.md](eval/README.md)
- 样例报告：[eval/results/sample-report.md](eval/results/sample-report.md) —— **mock 生成，只验证链路，数字不可引用**


## 已知边界

- 34 题样本量小，看趋势不看绝对值；扩文档后要重跑
- 单机 SQLite，没有并发压测；多 worker 下会话绑定仍在进程内存里（代码注释里写明了）
- BM25 用 bi-gram 切词（不依赖 jieba），中文长词召回略弱于分词方案
- 精排依赖 DashScope，离线环境会退化成「只用 RRF 排序」

## 配套文档

| 文档 | 内容 |
|---|---|
| [docs/agent-design.md](docs/agent-design.md) | 十个绕不开的设计问题：ReAct/Plan、Agent/Workflow、工具失败、多 Agent、MCP 与 Function Calling、记忆、召回、MCP/SKILL、评估、上下文 |
| [docs/eval-framework.md](docs/eval-framework.md) | 七维评测的口径与局限，以及「表面对」为什么会骗人 |
| [eval/README.md](eval/README.md) | 评测框架怎么跑、用例怎么写 |
