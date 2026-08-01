# 危险货物智能问答系统

本项目是一个面向危险货物法规查询的 GraphRAG 问答系统。系统以 GB 12268-2025《危险货物品名表》和 GB 6944-2025《危险货物分类和品名编号》为主要知识来源，结合 Neo4j 知识图谱、法规文本向量检索、BM25、Reranker 和大语言模型，支持用户用自然语言查询危险货物的类别、包装类别、特殊规定、例外数量、法规定义和判定依据等信息。

项目当前已经完成课程设计所需的核心闭环：数据处理、知识图谱导入、文本分块与索引、混合检索、问答接口、前端页面、三级缓存和回归测试。

## 主要功能

- 支持 UN 编号查询，例如 `UN1203 的包装类别是什么？`
- 支持危险货物名称查询，例如 `汽油属于哪一类危险货物？`
- 支持法规概念查询，例如 `什么是易燃液体？`
- 支持综合解释类问题，例如 `为什么 UN1203 属于第3类？`
- 支持 GB 12268 附录 A 特殊规定精确查询，例如 `特殊规定243是什么意思？`
- 支持证据展示，回答会返回知识图谱证据和法规文本证据。
- 支持 L1、L2、L3 三级缓存，减少重复计算和重复查询。
- 支持 Streamlit 前端和 FastAPI 后端分离运行。

## 系统架构

系统整体流程如下：

```text
用户问题
  ↓
问题理解与路由
  ↓
判断查询类型：KG 查询 / 文本直接查询 / 综合查询
  ↓
知识图谱检索 + 法规文本检索
  ↓
证据合并、去重、排序
  ↓
构造 Prompt
  ↓
LLM 生成最终回答
  ↓
返回答案、证据、缓存状态
```

### 查询路由

系统会先分析用户问题，识别：

- 查询主体：如 `UN1203`、`汽油`、`乙醇溶液`
- 查询目标：如危险类别、包装类别、特殊规定、定义、判定依据
- 查询路线：知识图谱查询、文本查询或综合查询

典型路由示例：

```text
UN1203 的包装类别是什么？
=> 综合查询：先查图谱中的结构化字段，再补充 GB 6944 条文依据。

什么是易燃液体？
=> 直接文本查询：从 GB 6944 中检索定义条款。

特殊规定243是什么意思？
=> 附录 A 精确查询：直接从 GB 12268 附录 A 中查找对应编号。
```

## 核心模块说明

```text
src/run.py
  FastAPI 后端入口，负责接口、缓存、问答主流程。

src/app.py
  Streamlit 前端页面。

src/graphrag/engine.py
  检索引擎入口，协调图谱查询、文本查询和综合查询。

src/graphrag/query_understanding.py
  用户问题理解、路由判断、L2 缓存问题规范化。

src/graphrag/graph_store.py
  Neo4j 图数据库查询与 L3 实体缓存。

src/graphrag/text_search.py
  文本向量检索、BM25、RRF 融合和 Reranker 排序。

src/graphrag/chunker.py
  法规文本分块和表格元数据挂接。

src/graphrag/evidence.py
  证据合并、去重、排序和 sources 输出构造。

src/graphrag/explanation_retrieval.py
  综合解释类证据检索。

src/graphrag/index.py
  GB 12268 UN 条目、附录 A 特殊规定、来源子索引等索引缓存管理。

src/graphrag/cache.py
  L1、L2、L3 缓存实现。

src/graphrag/definitions.py
  领域常量、关键词、目标标签和特例配置。

src/llm/
  LLM 调用、Prompt 构造和最终回答生成。

src/tools/
  数据处理、图谱导入、文本索引重建和测试脚本。
```

## 数据目录

```text
data/source/
  原始标准 PDF 文件。

data/tables/
  从标准中抽取的表格数据。

data/text/
  法规文本、FAISS 索引和 metadata。

data/cache/
  L2 语义缓存、别名记忆缓存等 SQLite 文件。

models/
  本地嵌入模型和重排模型。

reports/
  查询回归测试报告。
```

当前主要数据来源包括：

- `data/source/GB+12268-2025.pdf`
- `data/source/GB+6944-2025.pdf`
- `data/text/GB+12268附录A.txt`
- `data/text/GB+6944.txt`
- `data/tables/GB12268/GB+12268_tab.csv`

## 环境要求

建议环境：

- Python 3.12
- Neo4j
- 本地模型：
  - `models/bge-m3`
  - `models/bge-reranker-base`

主要 Python 依赖包括：

- fastapi
- uvicorn
- streamlit
- neo4j
- pandas
- numpy
- faiss
- sentence-transformers
- rank-bm25
- jieba
- openai
- httpx
- pydantic

如果已有 `venv`，可以直接使用项目中的虚拟环境；如果重新部署，需要按上述依赖安装对应包。

推荐在新环境中重新创建虚拟环境并安装依赖：

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell 中激活虚拟环境可使用：

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 配置说明

运行配置集中放在 `.env` 中。LLM 相关配置仍由 `src/llm` 包读取，其他运行配置由 `src/graphrag/settings.py` 管理。

示例配置：

```env
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false
CORS_ALLOW_ORIGINS=*

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=你的Neo4j密码

EMBEDDING_MODEL=./models/bge-m3
RERANK_MODEL=./models/bge-reranker-base
HF_ENDPOINT=https://hf-mirror.com

LLM_API_KEY=你的API Key
LLM_BASE_URL=你的兼容OpenAI接口地址
LLM_MODEL=deepseek-v4-flash
LLM_TEMPERATURE=0.2
LLM_TIMEOUT=90

QUERY_ANALYSIS_LLM_ENABLED=true
QUERY_ANALYSIS_LLM_THRESHOLD=0.85

L1_CACHE_SIZE=20
L1_CACHE_TTL=1200
L2_CACHE_DB=data/cache/semantic_cache.db
L2_CACHE_MIN_THRESHOLD=0.85
L2_CACHE_DIRECT_THRESHOLD=1.0
L2_CACHE_SIZE=5000
L2_CACHE_ENABLED=true
L3_CACHE_SIZE=200
L3_CACHE_TTL=86400

DATA_VERSION=gb-2025-v1
PROMPT_VERSION=v1
```

其中 L2 语义缓存当前策略是：

- 相似度小于 `0.85`：视为未命中。
- 相似度大于等于 `0.85`：进入候选阶段。
- 候选必须经过 LLM 语义复核，通过后才算 L2 命中。

## 数据准备

### 1. 下载本地模型

系统默认使用两个本地模型：

```text
models/bge-m3
models/bge-reranker-base
```

如果压缩包中已经包含 `models/` 目录，可以跳过本步骤。否则在项目根目录执行：

```bash
PYTHONPATH=src ./venv/bin/python src/tools/model_download.py
```

该脚本会下载：

```text
BAAI/bge-m3
BAAI/bge-reranker-base
```

并分别保存到：

```text
./models/bge-m3
./models/bge-reranker-base
```

如果网络不稳定，可以重复执行该脚本，脚本支持断点续传。若使用 Hugging Face 镜像，可在 `.env` 中配置：

```env
HF_ENDPOINT=https://hf-mirror.com
```

也可以手动从 Hugging Face 下载上述两个模型，并保持目录结构与 `.env` 中配置一致：

```env
EMBEDDING_MODEL=./models/bge-m3
RERANK_MODEL=./models/bge-reranker-base
```

### 2. 导入 Neo4j 知识图谱

先启动 Neo4j，并确认 `.env` 中数据库连接信息正确。

然后在项目根目录执行：

```bash
PYTHONPATH=src ./venv/bin/python src/tools/import_to_neo4j.py
```

该脚本会读取：

```text
data/tables/GB12268/GB+12268_tab.csv
```

并导入危险货物节点、类别节点、包装类别节点、运输要求等结构化信息。

### 3. 重建文本检索索引

如果修改了 `data/text/*.txt` 或表格元数据，需要重建文本索引：

```bash
PYTHONPATH=src ./venv/bin/python src/tools/rebuild_text_index.py
```

只检查分块结果、不生成索引：

```bash
PYTHONPATH=src ./venv/bin/python src/tools/rebuild_text_index.py --dry-run
```

重建后会生成：

```text
data/text/faiss_index.bin
data/text/metadata.json
```

## 启动项目

### 1. 启动后端

在项目根目录执行：

```bash
PYTHONPATH=src ./venv/bin/python src/run.py
```

默认后端地址：

```text
http://localhost:8000
```

常用接口：

```text
GET  /health
POST /ask
POST /prompt
```

### 2. 启动前端

另开一个终端执行：

```bash
PYTHONPATH=src ./venv/bin/streamlit run src/app.py
```

然后在浏览器中打开 Streamlit 输出的地址，通常是：

```text
http://localhost:8501
```

## 测试

项目提供了查询回归测试脚本：

```bash
PYTHONPATH=src ./venv/bin/python src/tools/test_queries.py --clear-l2
```

测试通过后会在 `reports/` 下生成 Markdown 和 JSON 报告。

报告示例：

```text
reports/query_test_report_20260722_153852.md
```

## 示例问题

可以在前端或 `/ask` 接口中测试以下问题：

```text
UN1203 的包装类别是什么？
UN1203 的危险类别是什么？
为什么 UN1203 属于第3类？
UN1203 有哪些特殊规定？
特殊规定243是什么意思？
特殊规定9999是什么意思？
什么是易燃液体？
易燃液体是什么？
GB6944中包装类别是什么意思？
汽油属于哪一类危险货物？
乙醇溶液的包装类别是什么？
UN9999的危险类别是什么？
```

## 缓存设计

系统目前包含三级缓存：

### L1 精确问答缓存

基于规范化后的原始问题、`top_k`、模型和 Prompt 版本生成缓存键，适合短时间内完全相同的问题。

### L2 语义答案缓存

用于处理同一问题的不同问法。系统不会直接使用原始问题，而是先生成规范化问题，例如：

```text
什么是易燃液体
易燃液体是什么
GB6944中易燃液体的定义是什么
=> 易燃液体的定义
```

再进行向量相似度匹配。

为了避免误命中，L2 还会约束：

```text
route
intent
entity
top_k
model
data_version
prompt_version
```

并且相似度达到阈值后仍需要 LLM 复核。

### L3 实体缓存

用于缓存 Neo4j 中危险货物实体的查询结果，减少重复图数据库访问。

## 当前已处理的关键问题

项目中已经针对若干容易影响问答准确性的点做过优化：

- 将原来过大的检索逻辑拆分到多个模块，降低 `engine.py` 的复杂度。
- 将领域关键词和定义类关键词集中到 `definitions.py`。
- 将 GB 12268 UN 条目、附录 A 特殊规定、精确查询和来源子索引集中到 `index.py`。
- 修复 source filter 先召回后过滤导致的召回损失风险。
- 修复定义类问题误召回附录 A 的问题。
- 修复直接附录 A 查询时证据提示容易矛盾的问题。
- 修复文本证据合并后未重新排序的问题。
- 修复多扩展查询 RRF 排名失真的问题。
- 修复 L2 缓存直接高相似度命中的风险，改为 LLM 复核。
- 修复 `为什么是` 被误判成 `什么是` 的缓存规范化问题。

## 注意事项

- 首次启动会加载 BGE-M3 和 Reranker 模型，耗时较长是正常现象。
- 如果 Neo4j 没有启动，涉及 UN 编号或品名的图谱查询会失败或缺少结构化证据。
- 如果 LLM 未配置，系统仍可返回检索证据，但最终自然语言回答和 L2 语义复核能力会受影响。
- 如果修改了法规文本或表格数据，需要重新导入图谱或重建文本索引。
- `.env` 中可能包含数据库密码和 API Key，提交或展示代码时应避免泄露真实密钥。
