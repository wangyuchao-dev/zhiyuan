# 贡献指南

本项目按「**开发 → 测试 → 生产**」三段推进，分支与流水线一一对应。
一个人开发也照这个走 —— 目的不是流程本身，而是**任何一次上线都能回答「这个版本在 test 上验过吗」**。

## 分支模型

```
feature/xxx ──┐
              ├─> develop ──> test ──> main
hotfix/xxx ───┘         (CI)     (CI)    (CI + 打 tag 出镜像)
```

| 分支 | 用途 | 谁能进 | 部署到 |
|---|---|---|---|
| `main` | 生产，随时可发版 | **只接受来自 `test` 的 PR** | 生产环境 |
| `test` | 验收环境，跑集成与压测 | 只接受来自 `develop` 的 PR | 测试环境 |
| `develop` | 日常开发主线 | 直接提交或合并 feature | 无（本地/CI） |
| `feature/<短横线描述>` | 一个功能一条 | 从 `develop` 切出 | 无 |
| `hotfix/<版本>-<描述>` | 线上急修 | 从 `main` 切出 | 修完同时回合并 `main` 与 `develop` |

### 为什么主分支叫 main 不叫 master

新仓库统一用 `main`（GitHub 默认）。老仓库若还是 `master`，改名前先确认 CI 与保护规则已指向新名字。

### 分支保护（在 GitHub 上配置，一次即可）

`main` 与 `test` 都应开启：

- Require a pull request before merging（禁止直推）
- Require status checks to pass（勾选 CI 的 `quality` / `test`）
- Require branches to be up to date before merging
- Do not allow bypassing the above settings

> 单人也别关掉 PR 流程：PR 页面留着一次「改了什么、为什么、怎么验的」，三个月后回看比 commit message 有用。

## 提交信息规范

采用 [Conventional Commits](https://www.conventionalcommits.org/)，一行标题 + 空行 + 正文（说清**为什么**，不是重复改了什么）。

```
<type>(<scope>): <一句话>

<正文：为什么这么改、权衡了什么、有什么副作用>
```

| type | 用于 |
|---|---|
| `feat` | 新功能 |
| `fix` | 修缺陷 |
| `refactor` | 不改行为的重构 |
| `perf` | 性能优化 |
| `test` | 测试相关 |
| `docs` | 文档 |
| `build` / `ci` | 依赖、构建、流水线 |
| `chore` | 杂项（版本号、忽略规则等） |

示例：

```
feat(querying): 双路召回改为并行 + 单路超时降级

串行时向量 3.1s + BM25 1.2s 全压在一次请求上；
并行后总耗时取两者较慢值，单路超时只影响自己。
RRF 融合按名次，避免相似度与 BM25 分量纲不同直接相加。
```

## 本地开发

```powershell
# 1. 环境（首次）
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt

# 2. 提交钩子（首次，之后每次 commit 自动检查）
.venv\Scripts\pre-commit install

# 3. 起服务（读 .env）
.\start.bat

# 4. 提交前自检（与 CI 同款）
powershell -ExecutionPolicy Bypass -File .\scripts\check.ps1
```

单独跑：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\lint.ps1    # ruff 自动修复 + 格式化
powershell -ExecutionPolicy Bypass -File .\scripts\test.ps1    # 测试 + 覆盖率
```

## 提交前检查清单

- [ ] `scripts/check.ps1` 通过（ruff + 测试 + 覆盖率 ≥ 65%）
- [ ] 新增逻辑有对应测试；**测试必须离线**（不调模型、不连数据库、不需要 Key）
- [ ] 没有把 `.env`、真实 Key、内网地址、真实文档内容提交进去
- [ ] 改了对外接口时，同步更新 README 与 `docs/`
- [ ] 改了版本号时，`config/version.py`、`pyproject.toml`、`CHANGELOG.md` 三处一致

## 发布流程

1. `develop` → PR 到 `test`，在测试环境验一轮（含 `pytest`、手工冒烟、`eval/` 抽样）
2. `test` → PR 到 `main`
3. 在 `main` 上打 tag 并推送：`git tag v1.2.0 && git push origin v1.2.0`
4. tag 触发 `Release` 流水线，构建镜像推到 `ghcr.io/wangyuchao-dev/zhiyuan:<版本>`
5. 在 `CHANGELOG.md` 顶部补这一版的变更

## 代码风格

- 行宽 100，双引号，4 空格缩进（`ruff format` 统一，不要手工对齐）
- 注释写**为什么**，不写**是什么**；能一眼看懂的代码不写注释
- 对外函数写 docstring，说明边界与失败行为（抛什么、返回什么表示「没有」）
- 不做「顺手重构」：一个 PR 只解决一件事，方便回滚
