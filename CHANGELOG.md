# 变更记录

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [1.2.0] - 2026-11-27

工程化改造：分支模型、CI、代码质量门槛、容器化。功能行为与 1.1.0 一致。

### Added

- **分支模型**：`main`（生产）/ `test`（验收）/ `develop`（开发），配合 `feature/*`、`hotfix/*`
- **CI 流水线**（`.github/workflows/ci.yml`）：ruff 静态检查 + 格式检查 + 单元测试 + 覆盖率门槛 65% + 镜像可构建
- **发布流水线**（`release.yml`）：打 tag 时构建镜像并推送 `ghcr.io`
- **代码质量配置**：`pyproject.toml` 统一 ruff / pytest / coverage 配置，`fail_under = 65`
- **提交钩子**：`.pre-commit-config.yaml`，含 `detect-private-key`（拦 API Key 与私钥）
- **容器化**：`Dockerfile`（非 root、依赖分层缓存、HEALTHCHECK）、`docker-compose.yml`、`.dockerignore`
- **运维端点**：`/api/ready`（就绪检查：密钥 / 向量库 / 运行目录）、`/api/version`
- **结构化日志**：`core/logging.py`，生产输出 JSON、开发输出可读文本；`X-Request-ID` 贯穿全链路
- **环境分层**：`APP_ENV`（development / test / production），`QaConfig.load()` 支持环境变量覆盖
- **版本单一来源**：`config/version.py`，`main.py` 的 OpenAPI 版本与镜像标签共用
- **本地脚本**：`scripts/check.ps1`（与 CI 同款）、`lint.ps1`、`test.ps1`
- **团队文档**：`CONTRIBUTING.md`、`SECURITY.md`、PR 模板、Issue 模板

### Changed

- 健康检查拆成 liveness（`/api/health`）与 readiness（`/api/ready`）：依赖挂了不再让编排系统反复重启进程
- 前端单页挂载到根路径，访问 `:8000` 直接是聊天界面，不再需要单独打开 HTML
- 依赖缺少时返回 **503 + 可读提示**，原来 `/api/documents` 直接抛 500 堆栈
- 全量 `ruff format` 与 `ruff check --fix`（410 处自动修复）；枚举改用 `StrEnum`，异常补 `from` 保留调用链

### Fixed

- `except` 中 `raise` 未带 `from`，异常链丢失，线上排查只剩一条「重置失败」
- `indexing/clean.py` 中重复嵌套的 `if repeated:`（重构残留）

## [1.1.0] - 2026-11-10

### Added

- 评测框架 `eval/`：七维指标（结果 / 检索 / 证据 / 过程 / 稳定性 / 成本 / 安全）
- 四套评测集：静态 20 / 动态 8 / 安全 6 / 故障 5
- 十个 Agent 设计点的落地代码：错误分类与重试、上下文预算、父子分块、双路并行召回、
  工具注册表与 MCP 互转、SKILL 懒加载、显式计划、记忆策略
- 文档清洗与质量门槛（页眉页脚 / 页码 / 目录 / 全角 / 重复段）

### Changed

- README 重写：效果数据前置，补上「已知边界」

## [1.0.0] - 2026-10-20

首个可用版本。

### Added

- 意图识别（规则快判 + 模型兜底）、三层记忆（历史 / 事实 / 摘要）
- 混合检索：向量 + BM25 + RRF 融合 + qwen3-rerank + 阈值过滤
- 文档入库流水线：PDF 切分 → 指纹幂等 → Chroma 持久化
- FastAPI 接口层与 SSE 流式输出，Vue 3 单页前端
