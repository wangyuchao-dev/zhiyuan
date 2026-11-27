# 安全策略

## 报告漏洞

不要开公开 issue。用 GitHub 的 [Private vulnerability reporting](https://github.com/wangyuchao-dev/zhiyuan/security/advisories/new)
或直接邮件联系维护者。请附：影响范围、复现步骤、你判断的严重程度。
初步响应目标 **72 小时内**。

## 密钥管理（最容易出事的一条）

### 哪些东西绝不能进仓库

| 类型 | 例子 | 正确存放 |
|---|---|---|
| 模型 API Key | `DASHSCOPE_API_KEY=sk-...` | `.env`（已被 `.gitignore` 忽略） |
| 云账号凭据 | `LTAI...` / AccessKey Secret | 系统环境变量或密钥管理服务，**不要出现在任何文件里** |
| 数据库口令 | `password: xxx` | 环境变量，代码里只写 `${DB_PASSWORD}` |
| 内网地址 | `192.168.x.x`、内部域名 | 配置项，不写死在代码与文档中 |

### 已经泄露了怎么办（按顺序，别跳步）

1. **先吊销，再清理**。删掉本地文件没用 —— 只要推送过，就要当作已泄露。
   - 模型 Key：控制台删除并重建
   - 云账号 AccessKey：RAM 控制台**禁用**（先禁用而不是删除，便于查这段时间的调用记录）
2. 查调用记录：确认有没有异常调用、异常地域、异常费用。
3. 清理历史：仅删除最新提交里的 Key 是不够的，`git log -p` 仍能看到。
   用 `git filter-repo` 或 GitHub 的 secret scanning 提示重写历史，然后**强制推送并通知所有协作者重新克隆**。
4. 复盘：想清楚它为什么会被提交 —— 通常是「本地调试直接写死在代码里」。

> 本仓库装了 `detect-private-key` 钩子，能在 commit 前拦住大部分情况，
> 但**它只认得典型格式**（PEM 私钥、常见云厂商 Key 前缀），自定义格式的密钥仍可能漏过。

### 本项目对密钥的处理

- `config/setting.py` 只从环境变量读敏感值，代码里没有任何默认 Key
- `.env.example` 只放字段名，不放真实值
- `/api/ready` 只回答「有没有配」，不回显任何 Key 内容
- 日志中间件记录 method/path/status/耗时，**不记录请求体与响应体**（问句里可能有业务敏感信息）

## 依赖安全

```powershell
# 有漏洞时升级对应包，并跑一次全量测试
pip install -r requirements-dev.txt --upgrade
.venv\Scripts\python.exe -m pytest
```

`requirements*.txt` 使用下限约束（`>=`）便于获取安全修复；生产镜像构建时可加
`pip install --require-hashes` 做哈希锁定。

## 数据安全

- 知识库 PDF 与向量库**不入库**（`.gitignore` 已排除），需各自从合规渠道获取
- `runtime/memory.db` 存有真实对话记录，同样不提交
- 以 `APP_ENV=production` 运行时，日志输出 JSON，便于接入脱敏管道
