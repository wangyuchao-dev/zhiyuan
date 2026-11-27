# skills · 技能包（懒加载）

每个子目录一个 `SKILL.md`，带极简 front-matter：

```
---
name: 技能名（英文短横线）
summary: 一句话说明「什么时候该用它」
keywords: 逗号分隔的关键词
---
正文：步骤 / 判断表 / 输出格式 / 边界
```

## 为什么要懒加载

`registry.index()` 只读 front-matter，把 `name + summary` 拼成目录塞进提示词；
真正命中后才 `load_body()` 读正文。几十个技能也只占几行上下文。

## 与 MCP 的分工

| | MCP | SKILL |
|---|---|---|
| 解决 | 能不能碰到外部系统（连接、鉴权、工具描述） | 碰到之后怎么干（流程、判断、输出格式） |
| 加载 | 连上就列出工具 | 只列目录，命中才展开 |

## 用法

```python
from skills.registry import SkillRegistry

reg = SkillRegistry("skills")
reg.index()
print(reg.manifest())  # 拼进系统提示词
print(reg.render("E1017 是什么"))  # 命中才展开正文
```
