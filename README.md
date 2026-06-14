# Text2SQL 对话式数据分析 Copilot

面向**不会写 SQL 的业务用户**(运营 / 市场 / 产品)的对话式数据分析助手:用一句话提问,
拿到**答案 + 洞察 + 图表**,SQL 是可选的、可审计的折叠项。

> 它**不是**"自然语言转 SQL 的生成器",而是一个产品化的数据分析 Copilot:
> 检索 schema → 生成并校验 SQL → 只读执行 → 把结果讲成人话 + 配图,全程可观测、可治理、会学习。

整个项目按"先打地基、再做产品"的顺序迭代:前 6 个阶段是**引擎**(评测 / 检索 / 自纠错 /
可靠性 / 可观测性),P7 起转向**产品层**(答案+洞察、多轮对话、多数据源、语义层、数据治理、
缓存 / 反馈飞轮 / 仪表盘)。每个阶段都有"改动前后用 Spider 量一遍"的工程纪律。

---

## 为什么说它是"工程",不是"demo"

| 维度 | 做了什么 |
|------|----------|
| **评测** | Spider benchmark 上的**执行准确率**(Execution Accuracy),每次改动 before/after 量化 |
| **检索 (RAG)** | 词法 / 嵌入两套 schema 检索器,压缩提示词、提升选表准确率 |
| **自纠错** | Agent 闭环:执行报错 → 观察 → 改写 → 重试(有界) |
| **可靠性** | 只读访问、危险 SQL 拦截(sqlglot)、查询超时看门狗、模型主备降级 |
| **可观测性** | 每次 LLM 调用落 JSONL trace(token / 延迟 / 成本),可选接 Langfuse |
| **多数据源** | SQLAlchemy 跨方言(PostgreSQL / MySQL / SQLite),只读事务 + 语句超时 |
| **语义层** | 业务术语 / 指标口径 / JOIN 规则 / PII 定义注入提示词,口径统一 |
| **数据治理** | 鉴权 + 行级权限(RLS 重写)+ 列级权限 + PII 脱敏 + 审计日志 |
| **黏性** | 语义缓存(0 成本秒回)+ 反馈飞轮(👍 沉淀已验证示例)+ 收藏仪表盘 |

---

## 关键结果(量化)

| 优化 | 效果 |
|------|------|
| Spider baseline(随机 100 题) | 执行准确率 **65.0%**,执行错误率 **0%** |
| Schema 检索(词法) | 表召回率 **100%**,提示词 token **-6%** |
| Few-shot 示例检索 | 执行准确率 **65% → 77%(+12)** |
| 语义缓存命中 | 重复首轮问题 **0 成本、~0 延迟**,答案逐字一致 |
| 反馈飞轮 | 👍 后换个问法即命中已验证示例(同库),越用越准 |

> **核心洞察**:执行错误率本就是 0% → 瓶颈是"逻辑答错"而非"语法错" →
> 优化重点应放在**检索 / few-shot**,而不是单纯堆自纠错。这条结论直接来自评测数据。

---

## 架构

```
                          ┌──────────── 治理边界(P10,在代码执行层强制,不靠提示词)
用户提问                  │
  → ① 鉴权 / 解析身份(角色 → 行列权限 + 脱敏 + RLS)
  → ② Schema 检索(RAG)            [P3]   ── 只挑相关表,压缩提示词
  → ③ 语义层注入(术语/指标/JOIN)  [P9b]  ── 业务口径
  → ④ 反馈飞轮 + few-shot 示例      [P11b/P4B] ── 已验证示例优先
  → ⑤ LLM 生成 SQL(结构化输出)
  → ⑥ 校验(sqlglot)+ RLS 重写 + 只读执行(超时看门狗)
  → ⑦ 自纠错循环(报错则改写重试)  [P4]
  → ⑧ 输出列脱敏 → 分析师把结果讲成人话 + 选图   [P7]
       (每步都落 trace:token / 延迟 / 成本)        [P5]

语义缓存 [P11a]:重复首轮问题在 ① 之前直接秒回(key 含角色,防越权)
收藏仪表盘 [P11c]:把常用问题钉成卡片,一键刷新拿最新数
```

---

## 技术栈

Python 3.11+(开发于 3.14)· [uv](https://github.com/astral-sh/uv) · OpenAI 兼容 LLM 客户端
(DeepSeek / 通义 / 智谱 / Kimi / OpenAI 任意切换)· Pydantic · sqlglot · SQLAlchemy ·
FastAPI · 原生 HTML/CSS/JS 单页前端 + Chart.js · Spider 数据集 · 可选 Langfuse

> 前端是**手写的定制单页**(对话线程 + 图表 + 仪表盘),不是 Streamlit/Gradio 套壳。

---

## 快速开始

```bash
# 1. 安装依赖(自动建 .venv)
uv sync

# 2. 配置 LLM 提供商
cp .env.example .env          # 然后填入你的 key / base_url / model

# 3. 构建本地样例库(电商:customers / products / orders)
uv run python scripts/make_sample_db.py

# 4. 冒烟测试
uv run python scripts/smoke_db.py     # 数据库流水线(不需要 LLM)
uv run python scripts/smoke_llm.py    # LLM 连通性(需要 .env)

# 5. 单元测试(113 个)
uv run pytest

# 6. 起 Web 端(演示主入口)
uv run python scripts/serve.py        # 打开 http://127.0.0.1:8000
```

命令行单次提问:

```bash
uv run python scripts/ask.py "每个城市有多少客户"
uv run python scripts/ask.py --correct 2 "已完成订单的总金额"   # 开自纠错
```

---

## 用 Docker 一键起(推荐)

一条命令起 **Postgres + app**,容器启动时自动:建样例库 → 灌 Postgres 数据 → 建只读角色 →
起 Web。开箱即可演示包含真实 Postgres 治理在内的完整能力。

```bash
cp .env.example .env          # 填入你的 LLM key(回答问题需要)
docker compose up --build     # 打开 http://127.0.0.1:8000
```

- app 通过 compose 内网主机名 `postgres` 连库,**只读角色密码由环境注入**,
  不写死在任何提交进 git 的文件里(`connections.docker.yaml` 用 `${PG_PASSWORD}` 占位)。
- Postgres 带 healthcheck,app 等它就绪后再灌数据,**不会竞态**。
- 没配 LLM key 也能起,样例库结构 / 治理三角色都在,只是"答问题"那步需要 key。

---

## 数据治理演示(真实 Postgres)

四层防御不是 PPT,而是端到端可复现的。用 Docker 起一个真·PG、灌入同份电商数据、建只读角色,
再用三个角色对照提问:

> 用 `docker compose up`(上一节)时这步**已自动完成**;下面是不走 compose 的手动等价命令。

```bash
# 起 PostgreSQL 并灌数据 + 建只读角色(详见 项目设计书 P9a)
docker run -d --name t2s-postgres -p 5433:5432 \
  -e POSTGRES_PASSWORD=postgres postgres:16
uv run python scripts/seed_postgres.py        # 灌电商数据 + 建 readonly 角色
```

在 Web 端用身份下拉切换:

| 角色 | 看到的数据 |
|------|-----------|
| `analyst` | 全量,无脱敏 |
| `viewer` | 引用 `signup_date` 等敏感列 → **直接被拒** |
| `ops_bj` | **只能看北京**(行级权限),且客户姓名**脱敏**为 `A****` |

> 治理在**代码执行边界**强制施加(鉴权拒列 / sqlglot 把 RLS 谓词重写进 AST / 输出阶段脱敏),
> **绝不依赖提示词**。危险 SQL 守卫 fail-open(只读兜底),RLS fail-closed(解析不了就拒,防漏数据)。

审计日志:

```bash
uv run python scripts/audit_report.py         # 今天的访问审计
uv run python scripts/audit_report.py --all
```

---

## 评测(Spider benchmark)

```bash
# 下载 Spider dev 集 + 它用到的 20 个数据库
uv run python scripts/prepare_spider.py --dbs

# 跑评测(可复现的随机抽样,固定 seed)
uv run python eval/run_eval.py --dataset mini                 # 本地 sanity 集
uv run python eval/run_eval.py --dataset spider --limit 100   # Spider 抽样
uv run python eval/run_eval.py --dataset spider               # 全量 dev(1034 题)
```

三个开关正交可叠加,方便做消融实验:

```bash
uv run python scripts/prepare_spider.py --train               # 先备 few-shot 池(一次)
uv run python eval/run_eval.py --dataset spider --limit 100 \
  --schema lexical --correct 2 --fewshot 5
```

每次运行打印执行准确率 / 错误率 / 延迟 / token / 估算成本,并把带时间戳的报告存进 `eval/results/`。

可观测性报表:

```bash
uv run python scripts/trace_report.py         # 按模型汇总 调用/成功率/token/成本/p50p95
```

---

## 代码地图

```
src/text2sql/
  config.py        .env 强类型配置
  models.py        Pydantic 数据契约
  sources.py       DataSource 抽象 + 注册表(自动发现 SQLite + connections.yaml)
  engine.py        SQLAlchemy 跨方言:内省 + 只读执行 + 类型规整
  db.py / schema.py / schema_index.py   按数据源分派(SQLite 老路径零改动)
  retriever.py / embeddings.py / examples.py   schema 检索(词法/嵌入)+ few-shot
  llm.py           provider 无关的 LLM 客户端 + 主备降级 + tracing
  agent.py         generate(严格版)+ converse(多轮对话版)
  analyst.py       把查询结果讲成人话(禁编造)+ 选图
  guard.py         sqlglot 危险 SQL 拦截
  tracing.py / pricing.py   每次调用落 JSONL + 成本估算
  semantics.py     语义层:加载 semantics/*.yaml + 渲染 + 检索扩展
  governance.py    鉴权 + 行列权限(RLS 重写)+ PII 脱敏 + 审计
  cache.py         语义缓存:归一化 key(含角色)+ TTL + 持久化 + 👎 失效
  feedback.py      反馈飞轮:👍 沉淀同库已验证示例 / 👎 作废缓存 + 移除
  pins.py          收藏仪表盘:pin 问题(非答案)、绑定 db+role、upsert 去重
  service.py       编排层
  api.py           FastAPI
  web/index.html   定制单页前端(对话线程 + 图表 + 仪表盘)
eval/              评测 harness(指标 / 数据集 / run_eval)
scripts/           样例库 / 冒烟 / Spider 准备 / 起服务 / PG 灌数 / 审计报表
semantics/<id>.yaml  业务语义层(=数据分级,提交进 git)
policies.yaml        治理策略(=访问决策,提交进 git)
tests/             113 个单元测试
data/              数据库 & 数据集(gitignore)
```

---

## 设计取舍(节选)

- **缓存用"归一化精确匹配"而非嵌入相似度**:`top5 客户` 和 `top10 客户` 嵌入几乎一样但答案不同,
  模糊命中会返回错数据 —— 正确性优先于省钱。
- **few-shot 池在评测里强制排除同库**(防泄漏、量泛化),但**飞轮的已验证示例故意包含同库**
  (生产里那是 institutional knowledge,不是作弊)—— 同一个"同库"问题,评测和生产的取舍正好相反。
- **👍 和 👎 不对称**:👍 把(问题→SQL)沉淀为正样本;👎 只说"错了"不给正解 → 永不入正样本,
  只做排查 + 作废该问题缓存。
- **仪表盘收藏的是"问题"不是"答案"**:全是过期数字的看板比没有更糟,卡片随时重跑取最新值。

> 完整的阶段记录、踩坑与面试问答见 [`项目设计书.md`](项目设计书.md);一页速览见
> [`进度快照.md`](进度快照.md)。

---

## License

MIT
