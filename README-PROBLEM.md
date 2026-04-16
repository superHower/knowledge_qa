# 企业知识库智能体平台 - 技术问题回答（面试题库）

本文档收录项目中各项技术问题的面试回答，覆盖 RAG 检索质量、Agent 架构、企业级工程化等核心考点。

---

## 面试题目录

| 序号 | 主题 | 核心考点 | 难度 |
|------|------|----------|------|
| **面试真题一** | RAG 检索质量 | 评估机制、切片策略、查询改写、HyDE | ⭐⭐⭐ |
| **面试真题二** | Agent 架构 | 工具调用、规划能力、记忆系统、ReAct | ⭐⭐⭐⭐ |
| **真题三** | Embedding 模型 | 本地模型、长文档处理 | ⭐⭐ |
| **真题四** | 向量数据库 | 多数据库切换、抽象接口 | ⭐⭐ |
| **真题五** | 流式输出 | SSE、引用跳变、用户体验 | ⭐⭐ |
| **真题六** | 文档解析 | OCR、复杂表格处理 | ⭐⭐ |
| **真题七** | 可观测性 | 结构化日志、链路追踪、Prometheus | ⭐⭐⭐ |
| **真题八** | 可靠性设计 | 重试熔断、降级、限流配额 | ⭐⭐⭐ |
| **真题九** | 持久化扩展 | 记忆持久化、多 worker、集群 | ⭐⭐⭐ |
| **真题十** | 多租户权限 | 租户隔离、RBAC、API Key | ⭐⭐⭐ |
| **真题十一** | 安全合规 | 输入过滤、Guardrails、PII 脱敏 | ⭐⭐⭐ |

---

## 面试技巧

### 如何回答架构问题

1. **先说现状**：我们项目用了什么
2. **再说问题**：面试官质疑的点在哪里
3. **然后方案**：我们怎么解决
4. **最后升华**：业界最佳实践、我的思考

### 如何展示深度

- 不仅说"用了什么"，还要说"为什么选这个"
- 不仅说"实现了"，还要说"遇到过什么问题，怎么解决的"
- 不仅说"能跑通"，还要说"如何保障 SLA"

### 如何应对追问

面试官可能会追问：
- "为什么不用 XXX？"
- "如果这个挂了怎么办？"
- "能支持多少并发？"
- "这个的延迟是多少？"

准备好用数据回答（延迟 P99、QPS、存储量级等）。

---

## 面试真题一：RAG 检索质量（硬伤一）

> **问题来源**：面试官质疑"你的 RAG 太天真，企业里根本跑不通"
>
> **核心考点**：检索质量评估、切片策略、查询改写

### 1.1 问题：检索质量没有评估机制

**面试官的质疑**：
- 你用了向量+关键词混合检索，但没有重排（Rerank），没有检索质量评估
- 向量召回的前 5 个片段可能 3 个是噪音
- 无法回答"这个问题的检索准确率是多少？"

**回答要点**：

```python
# rag/evaluator.py 中已实现
from dataclasses import dataclass

@dataclass
class RetrievalMetrics:
    """检索质量指标"""
    precision_at_k: float   # Precision@K
    recall_at_k: float      # Recall@K
    mrr: float             # Mean Reciprocal Rank
    ndcg_at_k: float       # NDCG@K
    noise_ratio: float     # 噪音比例

class QualityMonitor:
    """持续监控检索质量，自动触发优化"""
    thresholds = {
        "precision": 0.7,   # 低于阈值自动告警
        "recall": 0.8,
    }
```

**关键指标说明**：

| 指标 | 含义 | 企业 SLA |
|------|------|----------|
| Precision@K | 检索结果中相关的比例 | ≥ 0.7 |
| Recall@K | 召回的相关文档比例 | ≥ 0.8 |
| NDCG | 排序质量（考虑位置衰减） | ≥ 0.6 |
| MRR | 第一个相关结果的位置 | ≥ 0.5 |

**面试加分话术**：
> "我们不仅计算指标，还设置了阈值告警。比如 Precision < 0.7 时会自动发告警，提示需要优化切片策略或调整检索参数。"

---

### 1.2 问题：切片策略太简单

**面试官的质疑**：
- 切片长度、重叠策略、多粒度索引没有配置
- 企业 PDF 有表格、代码块、列表，统一切片必翻车
- 缺少文档结构感知，导致切片破坏语义完整性

**回答要点**：

```python
# document/structured_splitter.py 中已实现
class StructureAwareSplitter:
    """文档结构感知切片器"""

    # 自动识别标题层级
    HEADING_PATTERNS = [
        r'^#{1,6}\s+(.+)$',           # Markdown: # Title
        r'<h([1-6])[^>]*>(.+)</h\1>', # HTML: <h1>Title</h1>
        r'^(\d+[\.、\s]+[^\n]+)$',    # 数字标题: 1. Title
        r'^(第[一二三四五六七八九十百千]+[章节篇部分])',  # 中文: 第一章
    ]

    def split_with_structure(self, text: str):
        # 1. 解析文档结构（识别标题层级）
        # 2. 按语义单元切分（不破坏段落完整性）
        # 3. 智能重叠（50 字 overlap）
```

**多粒度索引**：

```python
class MultiGranularityIndexer:
    """同一文档创建不同层级的索引"""
    # 章节级：适用于长查询、概览类问题
    # 段落级：适用于中等长度查询
    # 句子级：适用于短查询、精确匹配
```

**面试加分话术**：
> "切片不是简单的硬截断，而是感知文档结构的。比如 Markdown 会识别 ## 标题层级，确保切片边界不破坏语义完整性。对于超长段落，会按句子边界递归切分。"

---

### 1.3 问题：没有 HyDE 或 Query Rewriting

**面试官的质疑**：
- 用户问"那个续保怎么弄"，直接拿原问句去搜，大概率搜不到"续保流程"
- 缺少查询改写、多查询融合等高级检索技术

**回答要点**：

```python
# rag/query_rewrite.py 中已实现
class EnsembleQueryRewriter:
    """组合查询改写器"""

    # 1. Multi-Query：生成多角度查询
    # 用户："那个续保怎么弄"
    # 改写：
    #   - "如何办理续保"
    #   - "续保流程步骤"
    #   - "保险续期需要什么材料"

    # 2. HyDE：生成假设性答案
    # "根据公司规定，续保需要先准备以下材料..."

    # 3. 并行检索所有变体，合并候选集
```

**面试加分话术**：
> "查询改写解决的是'表达多样性'问题。用户的问题往往口语化、模糊，但知识库是结构化的。我们用 Multi-Query 生成多个角度的查询，用 HyDE 生成假设性答案来捕捉语义意图。"

---

### 1.4 最终方案：完整检索流程

```
用户问题
    │
    ▼
┌─────────────────────────────────────┐
│ 1. 查询改写                          │
│    Multi-Query + HyDE → 5 个查询变体 │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. 多路检索                          │
│    向量 + 关键词 → Top 20 候选        │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. Cross-Encoder 重排序              │
│    精排 → Top 5 + 置信度             │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. 质量评估                          │
│    Precision/Recall/NDCG → SLA 监控  │
└─────────────────────────────────────┘
```

---

## 面试真题二：Agent 架构（硬伤二）

> **问题来源**：面试官质疑"根本不算 Agent，只是个 Retriever + LLM"
>
> **核心考点**：工具调用、规划能力、记忆系统、自主决策

### 2.1 问题：没有真正的工具调用

**面试官的质疑**：
- 项目名叫"知识库问答智能体平台"，但只有一个 agent.py 封装了 LLM 调用
- 真正的 Agent 要有工具调用（查数据库、发邮件、调 API）

**回答要点**：

```python
# agent/tool.py 中已实现完整工具系统
class ToolRegistry:
    """工具注册表 - 动态注册/注销工具"""
    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        tool = self.get(tool_name)
        return await tool.execute(**kwargs)

# 内置工具
class KnowledgeBaseTool(BaseTool):
    """知识库检索工具"""
    name = "knowledge_base_search"
    description = "在企业知识库中搜索相关信息"

class CalculatorTool(BaseTool):
    """计算器工具 - 涉及费用计算时使用"""

class DateTimeTool(BaseTool):
    """日期时间工具"""

class DatabaseTool(BaseTool):
    """数据库查询工具 - 查保单、理赔记录等"""
```

**面试加分话术**：
> "工具系统是 Agent 的核心能力。我们实现了 ToolRegistry，支持动态注册工具。比如知识库搜索是默认工具，但根据业务需求，可以扩展'查保单'、'发邮件'、'调飞书 API'等工具。LLM 根据上下文自主决定调用哪个工具。"

---

### 2.2 问题：没有任务规划能力

**面试官的质疑**：
- 多步推理能力缺失，比如"先查产品条款 → 再查用户保单 → 计算保费"
- 需要 Planner 来拆解复杂任务

**回答要点**：

```python
# agent/planner.py 中已实现多步骤规划
class MultiStepPlanner(BasePlanner):
    """使用 LLM 分析查询并拆解为多个步骤"""

    SYSTEM_PROMPT = """你是一个任务规划专家。
    你的任务是将用户的复杂问题拆解为可执行的步骤。"""

    async def create_plan(self, query: str, available_tools: list[str]) -> Plan:
        # LLM 分析意图
        # 生成步骤计划：
        # Step 1: 查询产品条款 (knowledge_base_search)
        # Step 2: 查询用户保单 (database_query)
        # Step 3: 计算保费 (calculator)
        # 返回 Plan 对象，包含依赖关系和执行顺序
```

**面试加分话术**：
> "复杂问题需要拆解。比如用户问'我的保单今年能赔多少'，Agent 会自动规划：先查产品条款中的赔付规则，再查用户的保单信息，最后计算具体金额。每个步骤之间有依赖关系，后续步骤可以用前一步的结果。"

---

### 2.3 问题：记忆系统不完整

**面试官的质疑**：
- 短期会话记忆有，但长期用户偏好记忆没有
- 无法学习用户习惯，实现个性化服务

**回答要点**：

```python
# agent/memory.py 中已实现完整记忆系统
class ShortTermMemory:
    """短期记忆 - 会话上下文"""
    def add_message(self, role: str, content: str):
        # 对话历史
    def get_context_for_llm(self) -> list[dict]:
        # 构建 LLM 上下文

class LongTermMemory:
    """长期记忆 - 用户偏好"""
    async def update_interaction(self, user_id: str, topic: str):
        # 更新用户交互记录

    async def extract_patterns(self, user_id: str) -> dict:
        # 提取用户模式：
        # - 高频话题
        # - 回答风格偏好
        # - 交互频率

@dataclass
class UserPreference:
    frequent_topics: list[str]  # ["保单查询", "理赔"]
    preferred_style: str         # "detailed" 或 "concise"
    interaction_count: int      # 交互次数
```

**面试加分话术**：
> "我们实现了完整的记忆分层：短期记忆管理当前会话，长期记忆记录用户偏好。比如用户经常问保单问题，Agent 会记住这个偏好，后续回答会更侧重保单相关内容。"

---

### 2.4 问题：没有自主决策能力

**面试官的质疑**：
- 缺少置信度评估和智能追问
- 比如"如果置信度低于 0.7，则反问用户澄清"

**回答要点**：

```python
# agent/decision.py 中已实现自主决策
class ConfidenceEvaluator:
    """置信度评估 - 多维度评分"""

    def evaluate(self, chunks, query, answer, tool_results) -> ConfidenceResult:
        # 1. 检索结果评估
        # 2. 工具调用结果评估
        # 3. 回答内容质量评估
        # 返回置信度 0-1

class DecisionEngine:
    """决策引擎 - 根据置信度做决策"""

    # 置信度 ≥ 0.8 → 直接回答
    # 置信度 0.5-0.8 → 回答 + 置信度说明
    # 置信度 < 0.5 → 追问用户澄清
    # 置信度 < 0.3 → 转人工/拒答
```

**面试加分话术**：
> "Agent 不是有问必答，而是有'自知之明'。置信度低于阈值会主动追问用户。比如用户问'我想办那个事'，Agent 会反问'您是指续保还是理赔呢？'，而不是瞎猜。"

---

### 2.5 最终方案：ReAct 执行循环

```python
# agent/executor.py 中实现 ReAct
class ReActExecutor:
    """ReAct = Reasoning + Acting"""

    async def execute(self, query: str) -> ExecutionResult:
        # Thought: "用户问保单余额，需要先搜索知识库"
        # Action: knowledge_base_search(query="保单余额查询")
        # Observation: "找到 3 条相关信息，最高相关度 0.85"
        #
        # Thought: "信息足够，可以回答"
        # Final Answer: "您的保单余额是 10000 元..."

        # 返回完整推理过程，可视化展示
```

**完整 Agent 架构**：

```
用户问题
    │
    ▼
┌─────────────────────────────────────────┐
│ Planner                                  │
│ 意图分析 → 任务拆解 → 执行顺序           │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ ReActExecutor                            │
│ ┌─────────────────────────────────────┐ │
│ │ Thought → Action → Observation       │ │
│ │     ↑_________________________|       │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ ToolRegistry                             │
│ 知识库 │ 计算器 │ 日期 │ 数据库          │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ DecisionEngine                           │
│ 置信度评估 → 智能追问 / 直接回答          │
└─────────────────────────────────────────┘
```

**面试加分话术**：
> "这是业界标准的 ReAct 模式。Agent 不是一次性生成答案，而是'边想边做'。每一步都有 Thought 记录思考过程，Action 执行工具，Observation 观察结果，直到置信度足够才生成最终答案。面试官可以随时打断问'你为什么调用这个工具'。"

---

## 工程化问题（硬伤三、四）

以下问题涉及企业级工程化能力，包括可观测性、可靠性、水平扩展、安全合规等。

---

## 3. Embedding 模型选择

### 问题

- text-embedding-3-small 是云端模型，企业私有化部署（金融、医疗）要求数据不出域
- 切片超过 8192 token 怎么办？

### 回答

**本地 Embedding 支持**

```python
# rag/embedding.py 中已实现
class LocalEmbedding:
    """本地 Embedding 模型"""
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device="cuda")
```

项目已支持切换到本地模型：
- **BGE（BAAI）**：中文最强，支持 512 tokens
- **GTE（GTE-base-zh）**：阿里中文模型
- **InstructorEmbedding**：支持自定义指令的 Embedding

```python
# 使用本地模型
embedding = EmbeddingFactory.create(
    provider="local",
    model_name="BAAI/bge-large-zh-v1.5",
)
```

**长文档截断处理**

```python
# structured_splitter.py 中已实现多粒度切���
class MultiGranularityIndexer:
    def create_multi_granularity_chunks(self, text: str):
        # 按语义单元切分，而非硬截断
        # 章节 < 512 tokens → 整章
        # 章节 > 512 tokens → 按段落递归切分
```

**Token 限制处理策略**：
1. **语义优先**：按段落/句子边界切分，不在句子中间截断
2. **重叠保留**：相邻切片有 50 字重叠，保持上下文连续性
3. **层级索引**：章节级/段落级/句子级多粒度，检索时按需召回

---

## 4. 向量数据库选型

### 问题

- 企业常用 Milvus、Pinecone、Elasticsearch
- 需要抽象接口支持切换

### 回答

**向量存储抽象**

```python
# rag/vector_store.py 中已实现基类
class VectorStore(ABC):
    """向量数据库抽象接口"""
    
    @abstractmethod
    async def create_collection(self, collection_name: str, vector_size: int):
        pass
    
    @abstractmethod
    async def search(self, collection_name: str, query_vector: list[float], top_k: int):
        pass
```

项目已实现：
- `QdrantStore`：Qdrant 向量库（已实现）
- `InMemoryVectorStore`：内存向量库（用于开发测试）

**扩展 Milvus 支持**（下一步）

```python
class MilvusStore(VectorStore):
    """Milvus 向量数据库"""
    def __init__(self, connection_string: str):
        from pymilvus import connections, Collection
        self.connection = connections.connect(host=connection_string)
```

**多数据库切换配置**：

```python
# .env
VECTOR_STORE_PROVIDER=qdrant  # 或 milvus, elasticsearch, pinecone
QDRANT_HOST=localhost
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

---

## 5. 流式输出实现

### 问题

- 引用来源往往在回答结束后才能完整召回
- 流式返回时引用会"跳变"

### 回答

**流式方案设计**

```
SSE 流式事件类型：

1. retrieval 事件（回答前）
   - 返回初步检索到的片段（高置信度）
   - 用户可以看到"正在检索..."

2. content 事件（回答中）
   - 流式返回回答内容
   - 引用信息以 [来源1] 形式内联在文本中

3. done 事件（回答后）
   - 返回完整的引用列表和置信度
   - 用户可以点击查看原始片段
```

**引用"跳变"解决方案**：

1. **乐观渲染**：先显示已检索到的引用，回答生成后整体更新
2. **锚点机制**：用 `[REF-1]` 标记引用，回答结束后补充内容
3. **延迟确认**：前端先显示"参考来源 3 个"，生成结束后高亮来源列表

```python
# agent/executor.py 中已实现流式
async for event in streaming_executor.execute_stream(query):
    if event["type"] == "retrieval":
        # 返回检索结果预览
        yield {"event": "retrieval", "chunks": [...]}
    elif event["type"] == "content":
        # 流式输出文本
        yield {"event": "content", "delta": token}
    elif event["type"] == "done":
        # 返回完整引用
        yield {"event": "done", "sources": [...], "citations": [...]}
```

---

## 6. 文档解析鲁棒性

### 问题

- pypdf 遇到扫描件/图片 PDF 直接跪
- 复杂表格（嵌套、合并单元格）没有处理

### 回答

**PDF 解析策略**

```python
# document/parsers.py 中实现分层解析
class PDFParser(BaseDocumentParser):
    async def parse(self, file_path: str):
        # 层级一：尝试文本提取
        text = self._extract_text_fast(file_path)
        
        # 层级二：如果文本稀疏，触发 OCR
        if self._is_text_sparse(text):
            text = await self._ocr_with_paddle(file_path)
        
        # 层级三：返回结果（附带元数据）
        return DocumentContent(
            content=text,
            metadata={"ocr_used": True, "table_count": len(tables)}
        )
```

**OCR 集成**（下一步）

```python
async def _ocr_with_paddle(self, file_path: str):
    """使用 PaddleOCR 识别扫描件"""
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')
    result = ocr.ocr(file_path)
    # 合并识别结果
```

**复杂表格处理**

```python
class TableExtractor:
    """表格提取器"""
    
    def extract_tables(self, file_path: str) -> list[Table]:
        # 1. 检测表格区域（YOLOv5 / 传统方法）
        # 2. 识别表头
        # 3. 解析单元格（处理合并单元格）
        # 4. 转为结构化数据
        # 5. 转回文本（用于检索）
```

---

## 7. 可观测性

### 问题

- 没有结构化日志、链路追踪、指标
- 无法回答：检索延迟、重排序 P99、工具调用慢的原因

### 回答

**结构化日志**

```python
# utils/logger.py（下一步实现）
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)

log = structlog.get_logger()
log.info("retrieval_completed", 
    query="续保流程",
    chunks_found=5,
    duration_ms=120,
    kb_id=1,
    trace_id="abc123",
)
```

**链路追踪**

```python
# 中间件自动注入 Trace ID
@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-ID", uuid.uuid4().hex)
    request.state.trace_id = trace_id
    
    # OpenTelemetry 自动追踪
    with tracer.start_as_current_span("knowledge_qa.chat") as span:
        span.set_attribute("trace_id", trace_id)
        response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id
        return response
```

**Prometheus 指标**

```python
# metrics.py（下一步实现）
from prometheus_client import Counter, Histogram, Gauge

# 请求计数
chat_requests = Counter(
    "chat_requests_total",
    "Chat requests",
    ["kb_id", "status"]
)

# 检索延迟
retrieval_latency = Histogram(
    "retrieval_latency_seconds",
    "Retrieval latency",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0]
)

# 工具调用计数
tool_calls = Counter(
    "tool_calls_total",
    "Tool calls",
    ["tool_name", "success"]
)
```

**关键指标**：

| 指标 | 说明 | 告警阈值 |
|------|------|----------|
| `retrieval_latency_p99` | 检索 P99 延迟 | > 500ms |
| `rerank_latency_p99` | 重排序 P99 | > 200ms |
| `tool_call_slow_rate` | 慢工具调用占比 | > 5% |
| `retrieval_quality` | 各知识库检索质量 | < 0.6 |
| `openai_quota_remaining` | API 配额剩余 | < 10% |
| `qdrant_health` | Qdrant 连接状态 | down |

**健康检查增强**

```python
# main.py 中实现
@app.get("/health")
async def health_check():
    checks = {
        "api": True,
        "qdrant": await check_qdrant(),
        "openai_quota": await check_openai_quota(),
        "database": await check_database(),
    }
    
    all_healthy = all(checks.values())
    return {
        "status": "ok" if all_healthy else "degraded",
        "checks": checks,
    }
```

---

## 8. 可靠性设计

### 问题

- OpenAI 调用失败直接抛异常，不会重试
- Qdrant 挂了服务不可用
- 文档重复上传没有幂等性
- 没有限流与配额

### 回答

**重试与熔断**

```python
# agent/llm.py 中实现重试
from tenacity import retry, stop_after_attempt, wait_exponential

class OpenAILLM(BaseLLM):
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((RateLimitError, TimeoutError))
    )
    async def generate(self, prompt: str, **kwargs):
        # 自动重试，指数退避
        pass

# 熔断器
class CircuitBreaker:
    """熔断器：连续失败 N 次后熔断"""
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = "closed"  # closed, open, half_open
```

**降级策略**

```python
# Qdrant 不可用时的降级
async def retrieve_with_fallback(self, query: str, kb_id: int):
    try:
        # 优先使用向量检索
        return await self.vector_search(query, kb_id)
    except VectorStoreError:
        # 降级：使用全文搜索
        return await self.fulltext_search(query, kb_id)
    except Exception as e:
        # 最终降级：返回"系统繁忙"
        return {"error": "service_degraded"}
```

**幂等性设计**

```python
# 文档重复上传检测
class DocumentProcessor:
    async def process_document(self, file_content: bytes, file_name: str):
        # 计算内容哈希
        content_hash = hashlib.md5(file_content).hexdigest()
        
        # 检查是否已存在
        existing = await self.db.find_doc_by_hash(content_hash)
        if existing:
            # 返回已有文档，不重复处理
            return existing, {"duplicated": True}
        
        # 正常处理流程...
```

**限流与配额**

```python
# api/middleware.py（下一步实现）
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/chat")
@limiter.limit("10/minute")  # 每人每分钟 10 次
async def chat(request: Request):
    pass

# 基于租户的配额
@app.middleware
async def quota_check(request: Request, call_next):
    tenant_id = request.headers.get("X-Tenant-ID")
    quota = await get_quota(tenant_id)
    
    if quota.requests_per_minute < 10:
        return JSONResponse({"error": "quota_exceeded"}, status_code=429)
    
    response = await call_next(request)
    return response
```

---

## 9. 持久化与水平扩展

### 问题

- LongTermMemory 没有持久化
- ToolRegistry 多 worker 无法共享
- Qdrant 单点

### 回答

**记忆持久化**

```python
# agent/memory.py 中实现持久化
class LongTermMemory:
    def __init__(self, db_session):
        self.db = db_session
    
    async def save_preference(self, preference: UserPreference):
        # 保存到数据库
        await self.db.merge(preference)
        await self.db.commit()
    
    async def get_preference(self, user_id: str) -> Optional[UserPreference]:
        stmt = select(UserPreference).where(UserPreference.user_id == user_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
```

**ToolRegistry 分布式**

```python
# 方案一：Redis 共享状态
class DistributedToolRegistry:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def register(self, tool: BaseTool):
        self.redis.hset("tools", tool.name, json.dumps(tool.get_definition()))
    
    def get_all(self):
        return [json.loads(v) for v in self.redis.hgetall("tools").values()]

# 方案二：配置中心（Apollo / etcd）
# 工具注册信息存储在配置中心，所有 worker 共享
```

**Qdrant 集群**

```bash
# docker-compose.yml
qdrant:
  image: qdrant/qdrant:latest
  environment:
    QDRANT__CLUSTER__ENABLED: "true"
    QDRANT__SERVICE__GRPC_PORT: 6334
```

**异步任务队列**

```python
# 处理大文档解析，使用消息队列
class DocumentProcessingService:
    def __init__(self, message_queue):
        self.queue = message_queue
    
    async def submit_document(self, doc_id: int):
        # 提交到队列，不阻塞 API
        await self.queue.publish(
            "document.process",
            {"doc_id": doc_id, "priority": "normal"}
        )
        return {"status": "queued", "doc_id": doc_id}
    
    async def get_status(self, doc_id: int):
        # 查询处理状态
        return await self.redis.get(f"doc_status:{doc_id}")
```

---

## 10. 多租户与权限模型

### 问题

- 没有租户隔离
- 没有 RBAC（角色）
- 没有 API Key + Scope
- 没有操作审计

### 回答

**租户隔离**

```python
# db/models.py 中添加租户字段
class KnowledgeBase(Base):
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    
class Tenant(Base):
    """租户"""
    __tablename__ = "tenants"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    quota: Mapped[dict] = mapped_column(JSON)  # 存储配额配置
```

**RBAC 权限模型**

```python
# db/models.py
class Role(Base):
    """角色"""
    __tablename__ = "roles"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))  # admin, editor, viewer
    permissions: Mapped[list] = mapped_column(JSON)


# 权限枚举
class Permission(str, Enum):
    KB_CREATE = "kb:create"
    KB_DELETE = "kb:delete"
    DOC_UPLOAD = "doc:upload"
    DOC_DELETE = "doc:delete"
    CHAT = "chat:read"
    CHAT_WRITE = "chat:write"
```

**API Key + Scope**

```python
# db/models.py
class APIKey(Base):
    """API Key"""
    __tablename__ = "api_keys"
    
    key_hash: Mapped[str] = mapped_column(String(64))  # SHA256 哈希
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    scopes: Mapped[list] = mapped_column(JSON)  # ["kb:read", "chat:write"]
    rate_limit: Mapped[int] = mapped_column(Integer, default=100)  # RPM
    created_at: Mapped[datetime]
    expires_at: Mapped[Optional[datetime]]
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
```

**操作审计日志**

```python
# db/models.py
class AuditLog(Base):
    """审计日志"""
    __tablename__ = "audit_logs"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int]
    user_id: Mapped[str]
    action: Mapped[str]  # create, update, delete
    resource_type: Mapped[str]  # knowledge_base, document, chat_session
    resource_id: Mapped[int]
    details: Mapped[dict] = mapped_column(JSON)  # 请求参数、变更前后
    ip_address: Mapped[str]
    created_at: Mapped[datetime]


# 中间件自动记录
@app.middleware
async def audit_log(request: Request, call_next):
    response = await call_next(request)
    
    if request.method in ("POST", "PUT", "DELETE"):
        await db.add(AuditLog(
            tenant_id=request.state.tenant_id,
            user_id=request.state.user_id,
            action=request.method,
            resource_type=get_resource_type(request.path),
            resource_id=get_resource_id(request.path),
            details={"body": await request.body()},
            ip_address=request.client.host,
        ))
        await db.commit()
    
    return response
```

---

## 11. 安全与合规

### 问题

- SQL 注入、Prompt Injection
- LLM 幻觉、有害内容、PII 泄露
- 敏感数据明文存储
- 审计日志

### 回答

**输入过滤**

```python
# utils/security.py（下一步实现）
class InputSanitizer:
    """输入净化器"""
    
    def sanitize(self, text: str) -> str:
        # 1. SQL 注入检测
        sql_patterns = ["'", '"', ";", "--", "UNION", "DROP TABLE"]
        for pattern in sql_patterns:
            if pattern.lower() in text.lower():
                raise ValueError("Invalid input detected")
        
        # 2. Prompt Injection 检测
        injection_patterns = ["ignore previous", "disregard instructions", "#!/"]
        for pattern in injection_patterns:
            if pattern.lower() in text.lower():
                text = text.replace(pattern, "[FILTERED]")
        
        return text
```

**LLM Guardrails**

```python
# agent/llm.py 中实现输出过滤
class LLMOutputGuardrail:
    """LLM 输出护栏"""
    
    def __init__(self):
        self.pii_detector = PIIDetector()  # 使用 Presidio
        self.content_filter = ContentFilter()
    
    async def filter_output(self, text: str) -> str:
        # 1. PII 检测与脱敏
        text = self.pii_detector.mask(text)
        
        # 2. 有害内容检测
        is_safe = await self.content_filter.check(text)
        if not is_safe:
            return "抱歉，我无法回答这个问题。"
        
        return text
```

**敏感数据脱敏**

```python
# PII 脱敏示例
from presidio_analyzer import AnalyzerEngine

class PIIMasker:
    def mask(self, text: str) -> str:
        analyzer = AnalyzerEngine()
        results = analyzer.analyze(text, language="zh")
        
        for result in results:
            entity_type = result.entity_type
            start, end = result.start, result.end
            
            if entity_type == "PHONE_NUMBER":
                text = text[:start] + "138****" + text[end:]
            elif entity_type == "EMAIL_ADDRESS":
                text = text[:start] + "***@***.com" + text[end:]
            elif entity_type == "CHINESE_ID":
                text = text[:start] + "**********" + text[end-4:]
        
        return text
```

**数据库敏感字段加密**

```python
# db/models.py
class ChatMessage(Base):
    """聊天消息（加密存储）"""
    
    content: Mapped[str] = mapped_column(
        Column(String(2000), nullable=False)
        # 使用数据库透明加密（TDE）或应用层加密
    )
    
    # 可选：加密字段
    encrypted_metadata: Mapped[Optional[bytes]] = mapped_column(
        LargeBinary, nullable=True
    )
```

**完整审计日志**

```python
# 审计日志字段
class AuditLog(Base):
    """完整审计日志"""
    
    # 谁
    tenant_id: Mapped[int]
    user_id: Mapped[str]
    api_key_id: Mapped[Optional[int]]
    
    # 什么
    action: Mapped[str]
    resource_type: Mapped[str]
    resource_id: Mapped[int]
    request_body: Mapped[Optional[dict]]
    
    # 结果
    status_code: Mapped[int]
    response_body: Mapped[Optional[dict]]
    error_message: Mapped[Optional[str]]
    
    # 上下文
    ip_address: Mapped[str]
    user_agent: Mapped[str]
    trace_id: Mapped[str]
    
    # 成本
    tokens_used: Mapped[Optional[int]]
    cost_usd: Mapped[Optional[float]]
    
    # 时间
    created_at: Mapped[datetime]
    duration_ms: Mapped[int]
```

---

## 总结：后续开发优先级

| 优先级 | 功能 | 说明 |
|--------|------|------|
| **P0** | 重试熔断、限流配额 | 保障服务稳定性 |
| **P0** | 输入过滤、Guardrails | 安全保障 |
| **P1** | 结构化日志、链路追踪 | 可观测性 |
| **P1** | 多租户、RBAC | 企业级需求 |
| **P1** | OCR 集成 | 文档解析鲁棒性 |
| **P2** | 本地 Embedding 完善 | 私有化部署 |
| **P2** | 异步任务队列 | 大文档处理 |
| **P2** | PII 检测、脱敏 | 合规 |
| **P3** | Milvus/ES 向量支持 | 数据库切换 |
| **P3** | 审计日志完善 | 合规 |

---

*本文档持续更新，记录技术选型考量和实现计划*
