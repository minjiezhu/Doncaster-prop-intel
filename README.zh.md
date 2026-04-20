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

### 为什么 Phase 1 先用纯向量检索？

- 纯向量检索是最小可验证的端到端基线。
- 在引入 BM25 关键词权重（hybrid 检索）之前，先验证摄入和分块质量是否符合预期。
- Phase 2 将在此基线上增加混合检索与重排层，进一步提升精度。

### 重排器的放置位置（后续阶段）

- 检索链路将演进为：召回（向量或混合）→ 重排 → 上下文组装 → 答案生成。
- 重排放在召回之后、Prompt 组装之前，既保留足够候选，又在生成前进行精细排序。

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

**示例请求：**

```bash
curl "http://localhost:8000/query?question=3108%20%E7%9A%84%E5%AD%A6%E5%8C%BA%E6%98%AF%E4%BB%80%E4%B9%88%EF%BC%9F&top_k=5"
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
      "distance": 0.12,
      "chunk_index": 3
    }
  ],
  "retrieval_debug": {
    "top_k": 5,
    "retrieval_time_ms": 143,
    "hits": 5
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

## 路线图

| 阶段 | 目标 | 状态 |
|------|------|------|
| Phase 1 | 基础 RAG 管线（摄入 + 向量检索 + 问答） | ✅ 已完成 |
| Phase 2 | 检索质量提升（hybrid 检索、重排、失败兜底） | 待开发 |
| Phase 3 | Agentic 工作流（多工具 Agent、Domain API、结构化日志） | 待开发 |
