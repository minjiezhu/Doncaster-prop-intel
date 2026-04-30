# doncaster-property-intel

基于 RAG（检索增强生成）的房产研究助手，聚焦澳大利亚维多利亚州 Manningham 地区。

主要覆盖郊区：Doncaster（3108）
扩展覆盖：Doncaster East（3109）、Templestowe 及 Templestowe Lower（3106）、Bulleen（3107）

> [English README](README.md)

---

## 技术栈

- 后端：Python + FastAPI
- LLM 编排：LangChain
- 本地大模型：Ollama（默认 qwen3:14b）
- 向量数据库：Weaviate（Docker）
- 前端：React（Phase 1 仅脚手架）

---

## Phase 1 交付内容

- Weaviate + FastAPI 后端的 Docker Compose 编排
- PDF、CSV、TXT 文档摄入管线
- 两种分块策略对比：
  - 固定窗口分块（默认 512 tokens，重叠 50）
  - 语义分块（句向量断点切分）
- 分块质量结构化日志（用于对比两种策略的差异）
- 查询接口：`GET /query`，执行向量检索并通过 LangChain + Ollama 生成答案

---

## Phase 2 交付内容

- **混合检索**：Weaviate 原生 BM25 + 向量融合，alpha 权重可配置
- **本地重排器**：sentence-transformers cross-encoder（CPU 友好），支持切换到 Cohere API
- **三种检索模式**（通过 `?mode=` 参数切换）：`vector`、`hybrid`、`hybrid_rerank`
- **失败兜底**：Weaviate 不可达、Ollama 超时、空召回均有优雅降级响应
- **检索对比日志**（retrieval_comparison.jsonl）：每次查询记录三种模式的指标便于横向比较

---

## 项目结构

```
doncaster-property-intel/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI 路由
│   │   ├── ingestion/    # 文档摄入管线
│   │   ├── retrieval/    # 向量检索与问答链
│   │   ├── agent/        # Phase 3 Agent 占位
│   │   └── config.py     # 环境配置（Pydantic Settings）
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/             # React 应用（后期阶段实现）
├── data/                 # 原始数据（已 gitignore）
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## 架构决策说明

### 为什么要对比两种分块策略？

- **固定窗口分块**：窗口大小确定，检索行为稳定，适合调参与基线建立。
- **语义分块**：以句子语义边界作为切分点，减少上下文断裂，理论上提升检索相关性。
- 摄入管线会将两种策略的质量指标（chunk 数量、平均长度、覆盖率、重复率）写入 JSONL 日志，使差异可量化而非主观评估。

### 为什么用混合检索代替纯向量检索？

- 房产查询通常包含精确关键词（郊区名称、邮编、价格区间），BM25 对精确词匹配有优势。
- 纯向量检索可能在嵌入空间中无法有效区分这些词，导致漏召回。
- 混合检索融合两种信号，`alpha` 参数可调，从而在不同查询类型下灵活平衡。

### 重排器的放置位置

- 检索链路：召回（向量或混合）→ 重排 → 上下文组装 → 答案生成。
- 召回阶段宽泛取候选（`top_k × HYBRID_CANDIDATE_MULTIPLIER`），重排后收窄到 `RERANKER_TOP_N`。
- 重排后于召回、先于 Prompt 组装，在保留候选多样性的同时保证生成质量。
- 本地 cross-encoder（`ms-marco-MiniLM-L-6-v2`）无需外部 API，CPU 即可运行；设置 `RERANKER_PROVIDER=cohere` 和 `COHERE_API_KEY` 可切换到 Cohere 免费版。

---

## 本地启动

**1. 复制环境变量文件**

```bash
cp .env.example .env
```

**2. 确保 Ollama 已在本地运行，并拉取所需模型**

```bash
ollama pull qwen3:14b
ollama pull nomic-embed-text
```

**3. 启动服务**

```bash
docker compose up --build
```

**4. 访问 API 文档**

- http://localhost:8000/docs

---

## 摄入 CLI

**摄入单个文件（同时使用两种分块策略）：**

```bash
python -m backend.app.ingestion.cli --path data/sample.txt --strategy both --suburb doncaster
```

**摄入整个目录：**

```bash
python -m backend.app.ingestion.cli --path data/ --strategy fixed
```

质量日志写入 `METRICS_LOG_PATH` 指定路径（默认：`backend/logs/chunk_quality.jsonl`）。

---

## 查询接口

`mode` 参数控制检索管线：

| mode | 行为 |
|------|------|
| `vector` | Phase 1 基线 — 纯余弦向量召回 |
| `hybrid` | BM25 + 向量融合（Weaviate 原生 hybrid）|
| `hybrid_rerank` | 混合召回 + cross-encoder 重排（默认）|

**示例请求：**

```bash
curl "http://localhost:8000/query?question=3108%20%E5%AD%A6%E5%8C%BA%E6%98%AF%E4%BB%80%E4%B9%88&top_k=5&mode=hybrid_rerank"
```

**响应结构：**

```json
{
  "answer": "...",
  "sources": [
    {
      "id": "...",
      "source": "school-zones.pdf",
      "suburb": "doncaster",
      "strategy": "semantic",
      "rerank_score": 8.43,
      "chunk_index": 3,
      "mode": "hybrid_rerank"
    }
  ],
  "retrieval_debug": {
    "mode": "hybrid_rerank",
    "top_k": 5,
    "candidates_before_rerank": 15,
    "final_hits": 5,
    "retrieval_time_ms": 120,
    "rerank_time_ms": 38,
    "total_time_ms": 310,
    "fallback_triggered": false
  }
}
```

---

## Phase 1 完成标准

- Weaviate 与后端通过 Docker Compose 成功启动
- 至少一个 PDF 或 CSV 文件成功摄入 Weaviate
- `chunk_quality.jsonl` 中包含 fixed 与 semantic 两种策略的对比统计条目
- `GET /query` 返回 answer 与 sources 字段

---

## Phase 2 完成标准

- `?mode=hybrid` 返回 BM25 + 向量融合结果
- `?mode=hybrid_rerank` 返回经 cross-encoder 重排后的结果
- `retrieval_comparison.jsonl` 覆盖三种模式的对比条目
- Weaviate 不可达和 Ollama 超时均返回可解释的优雅降级响应
- Phase 2 全部测试通过

---

## 路线图

| 阶段 | 目标 | 状态 |
|------|------|------|
| Phase 1 | 基础 RAG 管线（摄入 + 向量检索 + 问答） | ✅ 已完成 |
| Phase 2 | 检索质量提升（hybrid 检索、重排、失败兜底） | ✅ 已完成 |
| Phase 3 | Agentic 工作流（多工具 Agent、Domain API、结构化日志） | 待开发 |
