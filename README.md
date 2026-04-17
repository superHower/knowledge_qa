# 企业知识库问答智能体平台

企业级 RAG + Agent 智能体系统，支持多知识库管理、文档解析、向量检索、工具调用、任务规划、自主决策。

## 功能特性

### Agent 核心能力

| 模块                 | 文件                  | 功能说明                                                                                                                                     |
| -------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **工具系统**   | `agent/tool.py`     | ToolRegistry 工具注册表，支持动态注册/注销；内置工具：KnowledgeBaseTool、CalculatorTool、DateTimeTool、WebSearchTool、DatabaseTool           |
| **记忆系统**   | `agent/memory.py`   | ShortTermMemory（短期会话）、LongTermMemory（长期用户偏好）、WorkingMemory（整合）、EpisodicMemory（情景记忆）、ReflectionMemory（反思学习） |
| **任务规划**   | `agent/planner.py`  | BasePlanner 基类；SimplePlanner（单步）、MultiStepPlanner（多步 LLM 规划）、HierarchicalPlanner（层级规划）、DynamicPlanner（动态规划）      |
| **ReAct 执行** | `agent/executor.py` | ReActExecutor（推理循环）、StreamingReActExecutor（流式）；实现 Thought → Action → Observation → 最终回答                                 |
| **自主决策**   | `agent/decision.py` | ConfidenceEvaluator（置信度评估）、ClarificationGenerator（追问生成）、DecisionEngine（决策引擎）、ErrorHandler（错误处理）                  |

### 检索增强 (RAG)

| 模块                | 文件                     | 功能说明                                                                                                                                                        |
| ------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Embedding** | `rag/embedding.py`     | EmbeddingModel 协议；OpenAIEmbedding（云端）、LocalEmbedding（本地 sentence-transformers）、EmbeddingFactory                                                    |
| **向量存储**  | `rag/vector_store.py`  | VectorStore 基类；QdrantStore（Qdrant 向量库）、InMemoryVectorStore（内存向量库，用于开发测试）                                                                 |
| **高级检索**  | `rag/retriever.py`     | AdvancedRAGRetriever，整合查询改写、多路检索、重排序、质量评估                                                                                                  |
| **查询改写**  | `rag/query_rewrite.py` | MultiQueryRewriter（多查询）、HyDERewriter（假设文档）、SubQueryRewriter（子查询拆分）、QueryExpansionRewriter（同义词扩展）、EnsembleQueryRewriter（组合改写） |
| **重排序**    | `rag/reranker.py`      | BaseReranker 基类；CrossEncoderReranker（Cross-Encoder 精排）、TfidfReranker（BM25）、ReciprocalRankReranker（RRF 融合）、ScoreWeightedReranker（分数加权）     |
| **质量评估**  | `rag/evaluator.py`     | RetrievalEvaluator（Precision/Recall/MRR/NDCG）、QualityMonitor（持续监控、阈值告警）                                                                           |

## Agent 架构详解

### ReAct 执行流程

```
用户问题
    │
    ▼
┌─────────────────────────────────────────┐
│ TaskPlanner                             │
│ 分析意图 → 拆解任务 → 确定工具顺序       │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ ReActExecutor                           │
│                                         │
│  Step 1: Thought "需要先搜索知识库"      │
│          → Action: knowledge_base_search│
│          → Observation: 找到3条相关内容  │
│                                         │
│  Step 2: Thought "信息不足，追问计算"   │
│          → Action: calculator            │
│          → Observation: 1000元          │
│                                         │
│  Step 3: Thought "信息充分，可以回答"    │
│          → Final Answer                 │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ DecisionEngine                          │
│                                         │
│ confidence ≥ 0.8 → 直接回答             │
│ confidence < 0.5 → 追问澄清             │
│ confidence < 0.3 → 转人工               │
└─────────────────────────────────────────┘
    │
    ▼
最终回答 + 引用来源 + 推理过程
```

### 工具系统

```python
from knowledge_qa.agent import ToolRegistry, KnowledgeBaseTool, CalculatorTool

# 创建工具注册表
registry = ToolRegistry()

# 注册工具
registry.register(KnowledgeBaseTool(retriever, kb_id))
registry.register(CalculatorTool())

# 获取所有工具定义（用于 LLM Function Calling）
tools = registry.get_definitions()

# 执行工具
result = await registry.execute("knowledge_base_search", query="续保流程")
```

### 记忆系统

```python
from knowledge_qa.agent import ShortTermMemory, LongTermMemory, WorkingMemory

# 短期记忆
short_term = ShortTermMemory(max_messages=20)
short_term.add_user_message("我想查保单")
short_term.add_tool_result("knowledge_base_search", "找到保单信息...")

# 长期记忆
long_term = LongTermMemory()
await long_term.update_interaction(user_id, topic="保单查询")

# 工作记忆整合
working = WorkingMemory(short_term, long_term)
context = await working.get_context_for_llm(user_id)
```

### 置信度评估

```python
from knowledge_qa.agent import ConfidenceEvaluator, Decision

evaluator = ConfidenceEvaluator(
    high_threshold=0.8,
    medium_threshold=0.5,
    low_threshold=0.3,
)

result = evaluator.evaluate(
    chunks=[{"content": "...", "score": 0.85}],
    query="保单余额多少？",
    answer="您的保单余额是1000元",
    tool_results=[{"tool": "kb", "success": True}],
)

# result.confidence: 0.78
# result.decision: Decision.ANSWER
# result.reasons: ["检索相关度高 (平均: 0.85)"]
```

---

## RAG 检索流程

### 查询改写

```python
from knowledge_qa.agent import OpenAILLM
from knowledge_qa.rag import EnsembleQueryRewriter

llm = OpenAILLM(api_key="...")
rewriter = EnsembleQueryRewriter(llm)

# 输入：用户原始查询
queries = await rewriter.rewrite("那个续保怎么弄")

# 输出：改写后的多个查询
# ["那个续保怎么弄", "如何办理续保", "续保流程步骤", "保险续期需要什么材料"]
```

### 混合检索 + 重排序

```python
from knowledge_qa.rag import (
    AdvancedRAGRetriever,
    CrossEncoderReranker,
    ScoreWeightedReranker,
)

retriever = AdvancedRAGRetriever(
    embedding_model=embedding,
    vector_store=qdrant,
    reranker=CrossEncoderReranker(),
)

result = await retriever.retrieve(
    query="续保流程",
    knowledge_base_id=1,
    top_k=5,
)
```

---

## 数据库模型

```sql
-- 知识库
knowledge_bases
  id, name, description, embedding_model
  top_k, similarity_threshold, is_active
  created_at, updated_at

-- 文档
documents
  id, knowledge_base_id, file_name, file_path
  file_type, title, status, chunk_count
  created_at, updated_at

-- 文档切片
document_chunks
  id, document_id, content, content_hash
  chunk_index, vector_id, metadata
  created_at

-- 聊天会话
chat_sessions
  id, knowledge_base_id, session_name, user_id
  llm_model, temperature, message_count
  created_at, updated_at

-- 聊天消息
chat_messages
  id, session_id, role, content
  input_tokens, output_tokens
  retrieved_chunks, citations, created_at
```

---

## 配置说明

### Agent 配置

```python
from knowledge_qa.agent import AgentConfig

config = AgentConfig(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=2000,
    max_iterations=10,           # ReAct 最大迭代次数
    max_execution_steps=5,      # 工具调用最大次数
    confidence_threshold=0.7,   # 置信度阈值
    enable_tools=True,           # 启用工具调用
)
```

### RAG 配置

```python
from knowledge_qa.rag import (
    AdvancedRAGRetriever,
    CrossEncoderReranker,
    EnsembleQueryRewriter,
)

retriever = AdvancedRAGRetriever(
    embedding_model=embedding,
    vector_store=vector_store,
    reranker=CrossEncoderReranker(),
    enable_query_rewrite=True,   # 启用查询改写
    enable_rerank=True,          # 启用重排序
    enable_evaluation=True,      # 启用质量评估
    initial_top_k=20,           # 初始召回数量
    final_top_k=5,              # 最终返回数量
)
```

---
### 文档处理

| 模块               | 文件                      | 功能说明                                                                                                       |
| ------------------ | ------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **解析器**   | `document/parsers.py`   | TextParser（TXT/Markdown）、PDFParser、DocxParser、HTMLParser、CSVParser；ParserFactory 工厂模式               |
| **切片器**   | `document/splitter.py`  | TextSplitter，递归语义切片，支持重叠；`document/structured_splitter.py` 结构感知切片，感知标题层级、章节边界 |
| **文档处理** | `document/processor.py` | DocumentProcessor（文档解析+切片流程）、FileStorage（文件存储管理）                                            |

### 数据库

| 模块                 | 文件               | 功能说明                                                                                              |
| -------------------- | ------------------ | ----------------------------------------------------------------------------------------------------- |
| **数据模型**   | `db/models.py`   | KnowledgeBase、Document、DocumentChunk、ChatSession、ChatMessage、PromptTemplate；SQLAlchemy 异步 ORM |
| **数据库管理** | `db/database.py` | 异步会话管理（AsyncSessionLocal）、init_db()、drop_db()                                               |

### API 接口

| 模块                 | 文件                      | 功能说明                                    |
| -------------------- | ------------------------- | ------------------------------------------- |
| **知识库接口** | `api/knowledge_base.py` | 创建/查询/更新/删除知识库，获取统计信息     |
| **文档接口**   | `api/document.py`       | 上传文档、流式处理、列表查询、重新处理      |
| **对话接口**   | `api/chat.py`           | 问答对话、流式 SSE 输出、会话管理、历史记录 |

### 其他

| 模块              | 文件                    | 功能说明                                         |
| ----------------- | ----------------------- | ------------------------------------------------ |
| **配置**    | `core/config.py`      | Pydantic Settings 配置管理，环境变量加载         |
| **Schemas** | `schemas/__init__.py` | Pydantic 请求/响应模型验证                       |
| **服务层**  | `services/`           | KnowledgeBaseService、DocumentService 业务逻辑层 |

---

## 项目结构

```
knowledge_qa/
├── api/                          # API 路由
│   ├── __init__.py
│   ├── knowledge_base.py         # 知识库 CRUD
│   ├── document.py               # 文档上传/处理
│   └── chat.py                   # 对话/流式/SSE
├── agent/                        # Agent 核心 ⭐
│   ├── __init__.py
│   ├── base.py                   # AgentConfig, AgentStatus, AgentResponse, BaseAgent
│   ├── llm.py                    # OpenAILLM, ClaudeLLM, LLMFactory
│   ├── tool.py                   # ToolRegistry, BaseTool, 各种 Tool 实现
│   ├── memory.py                 # 短/长期/情景/反思记忆
│   ├── planner.py                # 任务规划器
│   ├── executor.py               # ReAct 执行循环
│   ├── decision.py               # 置信度评估/追问/决策
│   ├── agent.py                  # KnowledgeQAAgent 主类, AgentFactory
│   └── prompts.py                # PromptBuilder, RefinedPromptBuilder
├── rag/                          # RAG 模块
│   ├── __init__.py
│   ├── embedding.py              # Embedding 服务
│   ├── vector_store.py           # Qdrant / 内存向量库
│   ├── retriever.py              # AdvancedRAGRetriever 高级检索
│   ├── reranker.py               # 重排序器
│   ├── query_rewrite.py          # 查询改写
│   └── evaluator.py              # 质量评估
├── document/                     # 文档处理
│   ├── __init__.py
│   ├── base.py                   # BaseDocumentParser, DocumentContent
│   ├── parsers.py                # 各种文档解析器
│   ├── splitter.py               # TextSplitter
│   ├── structured_splitter.py    # StructureAwareSplitter, MultiGranularityIndexer
│   └── processor.py              # DocumentProcessor, FileStorage
├── db/                           # 数据库
│   ├── __init__.py
│   ├── models.py                 # SQLAlchemy 模型
│   └── database.py               # 异步会话管理
├── services/                     # 业务服务层
│   ├── __init__.py
│   ├── knowledge_base.py         # 知识库服务
│   └── document.py               # 文档服务
├── schemas/                      # Pydantic Schemas
│   └── __init__.py
├── core/                         # 核心配置
│   ├── __init__.py
│   └── config.py                # Settings 配置类
├── utils/                        # 工具函数
│   └── __init__.py
├── config.py                     # 全局配置
├── main.py                       # FastAPI 应用入口
├── pyproject.toml                # 项目配置
├── .env.example                  # 环境变量示例
└── .gitignore
```

---

## 技术栈

| 类别 | 当前实现 |
| --- | --- |
| Web 框架 | FastAPI + Uvicorn |
| 配置 | Pydantic Settings |
| ORM | SQLAlchemy 2.x |
| 数据库 | MySQL |
| 向量库 | Qdrant |
| LLM | OpenAI SDK 兼容接口 |
| Embedding | OpenAI Embedding |
| 文档解析 | pypdf、python-docx、html2text、pandas |
| 异步驱动 | aiomysql |
| 容器编排 | Docker Compose |

## 从 0 开始启动项目

### 1. 准备 Python 环境

```bash
cd code
mamba create -n knowledge_qa python=3.11
mamba activate knowledge_qa
```

### 2. 安装依赖

```bash
cd knowledge_qa
pip install -e .
```

### 3. 准备 `.env`

如果没有 `.env`，先复制模板：

```bash
cp .env.example .env
```

至少检查这些配置：

```env
APP_NAME=Knowledge QA Agent
APP_VERSION=0.1.0
DEBUG=false

DATABASE_URL=mysql+aiomysql://root:123456@localhost:3306/knowledge_qa?charset=utf8mb4

OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

QDRANT_HOST=localhost
QDRANT_PORT=6400
QDRANT_COLLECTION_NAME=knowledge_base
```

### 4. 准备依赖服务

本地至少需要：

- MySQL
- Qdrant

如果希望快速启动，建议直接用 Docker Compose：

```bash
docker compose up -d mysql qdrant
```

### 5. 启动应用

从 `code` 目录启动：

```bash
uvicorn knowledge_qa.main:app --reload --port 8000
```

或从 `knowledge_qa` 目录启动：

```bash
uvicorn main:app --reload --port 8000
```

### 6. 验证

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- 健康检查: [http://localhost:8000/health](http://localhost:8000/health)

## 当前暴露的接口

### 系统接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 应用健康检查 |

### 知识库接口

代码位置：[knowledge_base.py](file:///d:/MyWork/project-agent/code/knowledge_qa/api/knowledge_base.py)

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/v1/knowledge-bases` | 创建知识库 |
| `GET` | `/api/v1/knowledge-bases` | 列出知识库 |
| `GET` | `/api/v1/knowledge-bases/{kb_id}` | 获取知识库详情 |
| `PATCH` | `/api/v1/knowledge-bases/{kb_id}` | 更新知识库 |
| `DELETE` | `/api/v1/knowledge-bases/{kb_id}` | 删除知识库 |
| `GET` | `/api/v1/knowledge-bases/{kb_id}/stats` | 获取知识库统计 |

### 文档接口

代码位置：[document.py](file:///d:/MyWork/project-agent/code/knowledge_qa/api/document.py)

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/v1/documents?knowledge_base_id={id}` | 上传并处理文档 |
| `GET` | `/api/v1/documents?knowledge_base_id={id}` | 列出文档 |
| `GET` | `/api/v1/documents/{doc_id}` | 获取文档详情 |
| `DELETE` | `/api/v1/documents/{doc_id}` | 删除文档 |
| `POST` | `/api/v1/documents/{doc_id}/reprocess` | 重新处理文档 |

当前支持的上传文件类型，定义于 [document.py](file:///d:/MyWork/project-agent/code/knowledge_qa/api/document.py#L48-L57)：

- `.txt`
- `.md`
- `.pdf`
- `.docx`
- `.doc`
- `.html`
- `.csv`

### 对话接口

代码位置：[chat.py](file:///d:/MyWork/project-agent/code/knowledge_qa/api/chat.py)

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/v1/chat?knowledge_base_id={id}` | 普通问答 |
| `POST` | `/api/v1/chat/stream?knowledge_base_id={id}` | SSE 流式问答 |
| `GET` | `/api/v1/chat/sessions?knowledge_base_id={id}` | 列出会话 |
| `GET` | `/api/v1/chat/sessions/{session_id}` | 获取会话历史 |
| `DELETE` | `/api/v1/chat/sessions/{session_id}` | 删除会话 |

## 快速调用示例

### 创建知识库

```bash
curl -X POST "http://localhost:8000/api/v1/knowledge-bases" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"产品知识库\",\"description\":\"用于测试\"}"
```

### 上传文档

```bash
curl -X POST "http://localhost:8000/api/v1/documents?knowledge_base_id=1" \
  -F "file=@./demo.pdf"
```

### 普通问答

```bash
curl -X POST "http://localhost:8000/api/v1/chat?knowledge_base_id=1" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"这个知识库里都有哪些内容？\"}"
```

### 流式问答

```bash
curl -N -X POST "http://localhost:8000/api/v1/chat/stream?knowledge_base_id=1" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"请总结这份文档的重点\"}"
```

## 主要数据模型

基于 [models.py](file:///d:/MyWork/project-agent/code/knowledge_qa/db/models.py)，当前核心实体包括：

- `KnowledgeBase`
- `Document`
- `DocumentChunk`
- `ChatSession`
- `ChatMessage`
- `PromptTemplate`

### 模型关系

- 一个 `KnowledgeBase` 对应多个 `Document`
- 一个 `Document` 对应多个 `DocumentChunk`
- 一个 `KnowledgeBase` 对应多个 `ChatSession`
- 一个 `ChatSession` 对应多个 `ChatMessage`

## 配置入口

当前配置由 [config.py](file:///d:/MyWork/project-agent/code/knowledge_qa/core/config.py) 统一加载，来源是项目根目录 `.env`。

主要配置项包括：

- 应用配置：`APP_NAME`、`APP_VERSION`、`DEBUG`
- 数据库配置：`DATABASE_URL`
- LLM 配置：`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`
- 向量库配置：`QDRANT_HOST`、`QDRANT_PORT`、`QDRANT_COLLECTION_NAME`
- RAG 配置：`TOP_K`、`SIMILARITY_THRESHOLD`
