# 2026-11-10 修复记录（知渊 v1.0 → v1.1）

接手后通读全部源码（27 个 py / 1739 行），修掉「文档宣称的功能代码里没有对应实现」这一类问题。
原则：**只补实现、不改对外行为**；每处修复都有单元测试兜住。

## 一、致命问题（会导致功能不可用或直接 500）

| # | 问题 | 位置 | 修法 |
|---|---|---|---|
| 1 | `POST /api/reset` 必然 500：`Memory.clear_all()` 不存在 | `agent/memory.py` | 新增 `clear_all()`，清空历史/事实/摘要与计数器 |
| 2 | 三层记忆只有「历史」进 prompt：`build_context()` 与 `update_facts()` 从未被调用 | `agent/orchestrator.py` | 三个分支统一改用 `memory.build_context()`；新增 `update_facts()` 调用链 |
| 3 | 相关性阈值 `min_score=0.3` 定义了但从未生效（所有调用点传 0.0） | `querying/retrank.py` | 精排后按 `config.min_score` 过滤，低于阈值的片段直接丢弃 |
| 4 | 模型意图识别是死代码：`_quick_intent()` 永不返回 None，兜底分支不可达 | `agent/orchestrator.py` | 改为三级判定：强特征 → 文档特征词 → 都未命中才调模型（返回 None 走兜底） |
| 5 | 提示词仍写着「论文知识库 / 论文学习」，从练习项目抄来未改 | `config/prompts.py` | 改为企业制度知识库语境 |

## 二、影响效果

| # | 问题 | 位置 | 修法 |
|---|---|---|---|
| 6 | ANSWER_PROMPT 没有「没有依据就说未找到」，与「不会编」的宣称不符 | `config/prompts.py` | 补 4 条硬性要求（只用上下文、无依据就说未找到、标注引用、部分覆盖要说明） |
| 7 | 生成答案用的是**改写后**的问题，用户原话里的编号/时间可能被丢掉 | `querying/pipeline.py` | 检索用 `search_query`，生成用原始 `question` |
| 8 | 集合存在但为空时 `raise ValueError`，该文档之后每次提问都 500 | `indexing/vector_storage.py` | 退化为空库返回，回答降级为「未找到相关资料」；并提供 `_count()` 兜底私有属性 |

## 三、健壮性与规范

| # | 改动 | 位置 |
|---|---|---|
| 9 | 精排接口无异常保护 → 网络抖动整个问答 500；现在降级为按召回顺序返回 | `querying/retrank.py` |
| 10 | 召回为空仍调用精排接口（传空 documents 会报错）→ 提前返回 | `querying/retrank.py` |
| 11 | RRF 用 `page_content` 当融合键，不同页的同文片段会被吞掉 → 改用 `来源+页码+内容` 的哈希 | `querying/rrf.py` |
| 12 | 模块导入即构造 `ChatTongyi`，没配 Key 连 `/api/health` 都返回不了 → 改为懒加载单例 | `api/deps.py` |
| 13 | `SummaryManager` 的 `_summary_store` 形同虚设，摘要只存在 Memory 实例里 → 改为真正的读写接口 `get()/set()` | `agent/summary_manager.py` |
| 14 | 删除死代码：`SUMMARY_BATCH_SIZE`、`_unsummarized_count`、重复的 `if/elif` 分支 | `summary_manager.py` / `orchestrator.py` |
| 15 | Agent 直接访问 `self._rag._rewriter` 私有成员 → 新增公开方法 `RagPipeline.rewrite_query()` | `querying/pipeline.py` |
| 16 | 注释说「四个业务分支」，实际只有 complaint 一个 → 修正注释 | `agent/orchestrator.py` |
| 17 | 评测脚本 docstring 写「25 题（13+12）」，实际 34 题（21+13）→ 修正 | `querying/eval_retrieval.py` |

## 四、新增能力

- **事实槽真正有数据了**：`_extract_entities()` 用正则从问题里抽 `制度编号(HR-2026-001)`、`错误码(E1017)`、`条款号(第3条)`、`文件名(xxx.pdf)`，与模型返回的 `slots` 合并写入事实槽。
  好处：多轮追问「那第 3 条呢」时模型上下文里已经带着编号，不用再问一遍。
  顺带说明：上一轮我说的「`{context}` 存在模板注入风险」**是我判断错了** —— LangChain 只解析模板字符串，不重新解析变量值，PDF 里含 `{}` 不会出问题。这条没有改代码。

## 五、测试

新增 `tests/test_core.py`（标准库 unittest，22 个用例，零外部依赖、不联网）：

```
D:\Anaconda3\envs\langchain\python.exe -m unittest tests.test_core -v
Ran 22 tests ... OK
```

覆盖：RRF 加权与去重、记忆三层顺序与 `clear_all`、摘要阈值触发与落库、
精排阈值/空召回/异常降级、意图三级判定、实体抽取、向量库空集合自愈。

## 六、第二轮修复（复核后补）

第二轮复核指出了两处真问题，已修：

| # | 问题 | 修法 |
|---|---|---|
| 18 | `build_context()` 用 `AIMessage` 塞摘要和事实 → 语义上是「AI 说过的话」，且会出现连续三条 `ai`，破坏 human/ai 交替 | 改为**一条 `SystemMessage`**（摘要 + 事实合成一条，放在最后）。实测角色序列：`human → ai → human → ai → system` |
| 19 | `/api/chat` 无异常兜底：模型超时/503/token 超限 → 整个请求 500，用户只看到「服务异常」 | 包 try/except，返回可读回复（`route="error"`）+ 服务端打完整堆栈；`/api/reset` 同样加兜底 |
| 20 | `orchestrator.py:60` 注释仍写「四个业务分支」（上一轮只改了下面那处注释） | 改为「业务分支各自的链（目前只有 complaint 一条）」 |

测试从 22 个增加到 **26 个**，新增：`SystemMessage` 角色校验、无连续 AI 校验、`/api/chat` 异常路径、空问题 400、无 Key 时 `/api/health` 可用。

## 七、第三轮：功能补全（v1.0 → v1.1）

把原来写在「后续规划」里的三件事做掉了，并补上工程规范。

### 1. 多文档检索
- `Agent._session_doc`（单值）→ `_session_docs`（列表，按序去重）：一个会话可以连续上传多份 PDF
- `RagPipeline.retrieve_multi()`：逐份文档各自混合召回 → **跨文档 RRF 按排名融合** → 统一精排
  （每份文档的分数不可比，只能比排名——和单文档内「向量 vs BM25」融合是同一个道理）
- 单份文档召回异常只跳过它，不拖垮整个会话
- 接口：`POST /api/documents` 返回 `bound_count`；新增 `GET /api/documents`（查绑定）、
  `DELETE /api/documents`（解绑）
- 前端：上传后提示「本会话共 N 份文档，提问会跨这些文档检索」

### 2. 流式输出（SSE）
- `Agent.prepare()` 抽出「判意图 → 路由 → 检索」；`chat()` 与 `chat_stream()` 共用
- 新增 `POST /api/chat/stream`：事件顺序 `meta → evidence → delta* → done`，
  出错给 `error`，最后一条固定 `[DONE]`
- `AnswerGenerator.stream()` / `RagPipeline.stream_answer()`：用链自带的 `.stream()`
- 前端用 `fetch + ReadableStream` 逐字渲染（EventSource 只支持 GET，我们是 POST）
- 流式失败自动退回一次性接口，老浏览器也能用

### 3. 记忆持久化（SQLite）
- 新增 `agent/persistence.py`：`messages / facts / summaries` 三张表，标准库 sqlite3，零依赖
- 三个 Manager 全部改为读写库，对外 API 不变
- 摘要从「只存在实例里」改为真正落库，重启后 `Memory` 构造时自动读回
- 新增配置 `QaConfig.memory_db_path`（默认 `runtime/memory.db`）
- 想换 Redis 只改 `persistence.py`，上层不用动

### 4. 评测与文档取数
- 新增 `querying/eval_single_doc.py`：跑真实链路（召回 → 精排 → 阈值），
  打印三题的召回条数、精排分数、过阈值条数，直接产出产品说明 6.1 要的数字

### 5. 注释补全（可核对）
用 `ast` 做了一次覆盖率审计，逐文件补齐：

| 指标 | 补齐前 | 补齐后 |
|---|---|---|
| 缺模块级 docstring 的文件 | 24 / 30 | **0 / 30** |
| 业务模块里缺 docstring 的公开成员 | 28 处 | **0** |

新增的说明都写「为什么」而不是「是什么」，例如：
- `retrank.py`：为什么阈值要在这里生效（宁可变空也不硬塞无关内容）
- `rrf.py`：为什么不能加权求和（量纲不同）+ k=60 的作用
- `vector_storage.py`：为什么必须指定 cosine（L2 在 1536 维无上界）
- `api/deps.py`：为什么改懒加载（模块导入即构造模型 → 没 Key 起不来）
- `main.py`：多 worker 下文档绑定仍在内存，需要挪库或 sticky session

测试文件（`tests/`）只要求模块级说明，测试方法与 3 行假对象不强制加 docstring ——
方法名本身即说明（`test_below_min_score_dropped`），堆注释反而是噪音。

### 6. 测试
`tests/test_core.py` 从 26 个用例扩到 **36 个**，新增：持久化重连、会话隔离、
多文档绑定/融合、流式事件序列、SSE 帧格式。

## 八、待办（未做）

1. **产品说明 6.1 的分数需重测**：本机无 `DASHSCOPE_API_KEY`，无法代跑；
   在有 Key 的机器执行 `python querying/eval_single_doc.py`，把那两个数填回文档
2. **多 worker 下的文档绑定**：记忆已落库，但 `_session_docs` 仍在进程内存，
   多 worker 部署需要挪到库里或改 sticky session
3. **首字延迟量化**：流式已上线，但没有统计「首字耗时」，文档里只写了定性结论
4. **文档格式**：仍只支持 PDF（Word / Excel / 网页加载器未做）
5. **文档更新**：内容变更会新建集合，旧集合需手动清理，未做版本管理
