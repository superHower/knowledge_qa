
使用 LangGraph 重构知识库问答 Agent 技术方案
📋 现状分析
当前架构（自定义 ReAct 实现）
knowledge_qa 当前技术栈：
├── agent/
│   ├── base.py          # BaseAgent 抽象基类
│   ├── agent.py         # KnowledgeQAAgent（主 Agent）
│   ├── executor.py      # ReActExecutor（手写 ReAct 循环）
│   ├── planner.py       # MultiStepPlanner（任务规划）
│   ├── tool.py          # ToolRegistry（工具系统）
│   ├── memory.py        # 记忆系统（短期/长期）
│   ├── decision.py      # DecisionEngine（置信度评估）
│   └── prompts.py       # Prompt 模板
├── rag/
│   ├── embedding.py     # OpenAI Embedding
│   ├── vector_store.py  # Qdrant 客户端
│   ├── retriever.py     # AdvancedRAGRetriever（多路召回+重排序）
│   ├── reranker.py      # 重排序器
│   ├── query_rewrite.py # 查询改写
│   └── evaluator.py     # 质量评估
└── api/chat.py          # FastAPI 接口（REST + SSE）
核心问题：

❌ ReAct 循环自己实现（executor.py 110-195 行）——容易出错，难调试
❌ 工具调用逻辑自己管理（tool.py）——状态传递不清晰
❌ 状态机分散（agent.py、executor.py、memory.py）
❌ 难以扩展新流程（比如加个反思步骤）
❌ 工具调用后状态回传逻辑复杂
🎯 重构目标
用 LangGraph 替代手写 ReAct 循环

预期收益
维度	当前	LangGraph 重构
代码量	~500 行（executor + agent）	~200 行（StateGraph 定义）
状态管理	分散在多个类	统一的 AgentState
流程控制	while 循环 + if-else	声明式图（节点 + 边）
扩展性	改代码	改图结构（加节点/边）
容错	手动 try-catch	内置异常处理、重试
流式	自定义 SSE	LangGraph 原生支持
🏗️ 新架构设计
整体架构图
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI 应用（knowledge_qa）             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              LangGraph StateGraph（核心）            │  │
│  │                                                      │  │
│  │   ┌─────────┐    ┌─────────┐    ┌─────────┐        │  │
│  │   │ retrieve │───▶│ generate│───▶│  decide │        │  │
│  │   └─────────┘    └─────────┘    └─────────┘        │  │
│  │        │              ▲              │              │  │
│  │        │              │              ▼              │  │
│  │        │         ┌─────────┐   ┌─────────┐          │  │
│  │        └────────▶│  tool   │──▶│ generate│          │  │
│  │                  └─────────┘   └─────────┘          │  │
│  │                         │                           │  │
│  │                         ▼                           │  │
│  │                      ┌─────┐                        │  │
│  │                      │ end │                        │  │
│  │                      └─────┘                        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
核心组件映射
当前代码	LangGraph 重构后
ReActExecutor（292 行）	StateGraph 节点 + 边（~50 行）
KnowledgeQAAgent.run()（164 行）	graph.invoke() 调用
agent.decision.py	decide 节点（条件路由）
agent.memory.py	AgentState 中的 messages 字段
agent/tool.py	ToolNode（LangGraph 内置）
自定义日志	FastAPI 标准日志
📐 LangGraph 状态定义
AgentState（统一状态）

# knowledge_qa/graph/state.py

from typing import Annotated, List, Dict, Any, Optional
from dataclasses import dataclass, field
from langgraph.graph import add_messages

@dataclass
class AgentState:
    """Agent 全局状态"""

    # 对话消息（LangGraph 自动追加）
    messages: Annotated[List[Dict[str, Any]], add_messages] = field(default_factory=list)

    # 用户查询
    query: str = ""

    # 检索结果
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)

    # 工具调用历史
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    # 当前思考
    current_thought: Optional[str] = None

    # 置信度（0-1）
    confidence: float = 0.0

    # 是否需要澄清
    needs_clarification: bool = False

    # 澄清问题
    clarification_question: Optional[str] = None

    # 最终答案
    final_answer: Optional[str] = None

    # 会话 ID
    session_id: Optional[int] = None

    # 知识库 ID
    knowledge_base_id: int = 1

    # 错误信息
    error: Optional[str] = None
🔄 图（Graph）设计
节点实现（Nodes）

# knowledge_qa/graph/nodes.py

from knowledge_qa.graph.state import AgentState

async def retrieve_node(state: AgentState) -> AgentState:
    """
    节点 1：检索（Retrieve）
    - 查询改写（可选）
    - 向量检索（调用 AdvancedRAGRetriever）
    - 重排序
    """
    from knowledge_qa.rag import get_vector_store, AdvancedRAGRetriever
    from knowledge_qa.rag.embedding import OpenAIEmbedding

    embedding = OpenAIEmbedding()
    vector_store = get_vector_store()
    retriever = AdvancedRAGRetriever(embedding, vector_store)

    result = await retriever.retrieve(
        query=state.query,
        knowledge_base_id=state.knowledge_base_id,
        top_k=5,
    )

    state.retrieved_chunks = [
        {
            "content": c.content,
            "source": c.document_name,
            "relevance": c.score,
        }
        for c in result.chunks
    ]
    return state

async def generate_node(state: AgentState) -> AgentState:
    """
    节点 2：生成（Generate）
    - 构建上下文
    - 调用 LLM
    - 判断是否最终回答
    """
    from knowledge_qa.agent.llm import OpenAILLM

    llm = OpenAILLM()

    # 构建上下文
    context = _build_context(state.retrieved_chunks)

    # 构建消息
    messages = [
        {"role": "system", "content": _get_system_prompt()},
        {"role": "user", "content": f"{context}\n\n问题：{state.query}"},
    ]

    # 调用 LLM
    response = await llm.generate(prompt=messages, temperature=0.7)

    state.current_thought = response.content
    return state

async def decide_node(state: AgentState) -> AgentState:
    """
    节点 3：决策（Decide）
    - 置信度评估
    - 判断是否需要澄清
    """
    # 简单实现：基于检索结果质量判断
    if not state.retrieved_chunks:
        state.confidence = 0.3
        state.needs_clarification = True
        state.clarification_question = "抱歉，知识库中没有找到相关信息，能否换个问法？"
    elif state.retrieved_chunks[0].get("relevance", 0) < 0.5:
        state.confidence = 0.5
        state.needs_clarification = True
        state.clarification_question = "我对这个问题的答案不太确定，您能补充一些信息吗？"
    else:
        state.confidence = 0.85
        state.final_answer = state.current_thought

    return state

def _build_context(chunks: list[dict]) -> str:
    """构建 RAG 上下文"""
    if not chunks:
        return "没有找到相关上下文信息。"

    parts = ["【参考上下文】\n"]
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] {chunk['source']}:\n{chunk['content']}\n")
    return "\n".join(parts)

def _get_system_prompt() -> str:
    """系统提示词"""
    return """你是一个专业的企业知识库问答助手。
要求：

1. 基于提供的上下文信息回答问题
2. 如果上下文中没有相关信息，诚实地告知用户
3. 回答要准确、简洁、有条理
   """
   条件路由（Edges）

# knowledge_qa/graph/edges.py

from knowledge_qa.graph.state import AgentState

def should_continue(state: AgentState) -> str:
    """
    条件路由：决定下一步走哪个节点

    - 如果需要澄清 → end
    - 如果有工具调用意图 → tool
    - 否则 → decide
    """
    if state.needs_clarification:
        return "end"
    elif state.tool_calls:
        return "tool"
    else:
        return "decide"

def after_tool(state: AgentState) -> str:
    """工具执行后，回到生成节点继续推理"""
    return "generate"

def after_decide(state: AgentState) -> str:
    """决策完成，结束"""
    return "end"
图编译（Graph）

# knowledge_qa/graph/graph.py

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from knowledge_qa.graph.state import AgentState
from knowledge_qa.graph.nodes import retrieve_node, generate_node, decide_node
from knowledge_qa.agent.langchain_tools import tools

def create_graph() -> StateGraph:
    """创建并编译图"""

    # 1. 创建图
    workflow = StateGraph(AgentState)

    # 2. 添加节点
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("decide", decide_node)
    workflow.add_node("tool", ToolNode(tools))

    # 3. 设置入口点
    workflow.set_entry_point("retrieve")

    # 4. 添加边
    workflow.add_edge("retrieve", "generate")

    # 5. 条件边：生成后判断
    workflow.add_conditional_edges(
        "generate",
        lambda state: "tool" if state.tool_calls else "decide",
        {
            "tool": "tool",
            "decide": "decide",
        }
    )

    workflow.add_edge("tool", "generate")
    workflow.add_edge("decide", END)

    # 6. 编译（添加检查点）
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    return graph

# 单例

graph = create_graph()
🛠️ 工具集成
LangChain Tools 格式

# knowledge_qa/agent/langchain_tools.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field

class KnowledgeBaseInput(BaseModel):
    query: str = Field(description="用户问题")
    top_k: int = Field(default=5, description="检索数量")

@tool("knowledge_base_search", args_schema=KnowledgeBaseInput)
async def knowledge_base_search(query: str, top_k: int = 5) -> dict:
    """
    搜索知识库文档。
    适用于：产品信息、政策、技术文档、FAQ 等问题。
    """
    from knowledge_qa.rag import get_vector_store, AdvancedRAGRetriever
    from knowledge_qa.rag.embedding import OpenAIEmbedding

    embedding = OpenAIEmbedding()
    vector_store = get_vector_store()
    retriever = AdvancedRAGRetriever(embedding, vector_store)

    result = await retriever.retrieve(query=query, top_k=top_k, knowledge_base_id=1)

    return {
        "chunks": [
            {
                "content": c.content,
                "source": c.document_name,
                "relevance": c.score,
            }
            for c in result.chunks
        ],
        "total": result.total_chunks,
    }

@tool("calculator")
async def calculator(expression: str) -> dict:
    """计算数学表达式"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": str(e)}

# 工具列表（导出给 ToolNode 使用）

tools = [knowledge_base_search, calculator]
📡 API 接口
替换 /chat/stream

# knowledge_qa/api/chat.py（直接替换现有实现）

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from knowledge_qa.graph import graph
from knowledge_qa.graph.state import AgentState

router = APIRouter(prefix="/chat", tags=["对话"])

@router.post("/stream")
async def chat_stream(
    query: str,
    knowledge_base_id: int = 1,
    session_id: int = None,
):
    """
    流式对话接口（基于 LangGraph）
    替换原有 KnowledgeQAAgent + chat_stream 实现
    """
    import json
    import logging
    logger = logging.getLogger(__name__)

    async def event_generator():
        config = {"configurable": {"thread_id": str(session_id) if session_id else "default"}}

    try:
            # 阶段 1：检索
            yield f"event: status\ndata: {json.dumps({'status': 'retrieving', 'message': '正在检索...'})}\n\n"

    async for event in graph.astream(
                {
                    "query": query,
                    "knowledge_base_id": knowledge_base_id,
                    "session_id": session_id,
                    "messages": [],
                },
                config=config,
                stream_mode="values",
            ):
                # 检索完成
                if event.get("retrieved_chunks") and not event.get("final_answer"):
                    chunk_count = len(event["retrieved_chunks"])
                    yield f"event: status\ndata: {json.dumps({'status': 'retrieved', 'count': chunk_count})}\n\n"

    # LLM 生成中
                if event.get("current_thought"):
                    yield f"event: content\ndata: {event['current_thought']}\n\n"

    # 最终答案
                if event.get("final_answer"):
                    yield f"event: done\ndata: {json.dumps({\n                        'answer': event['final_answer'],\n                        'confidence': event.get('confidence', 0),\n                        'citations': event.get('retrieved_chunks', []),\n                    })}\n\n"

    # 需要澄清
                if event.get("needs_clarification") and event.get("clarification_question"):
                    yield f"event: clarify\ndata: {json.dumps({'question': event['clarification_question']})}\n\n"

    # 错误处理
                if event.get("error"):
                    yield f"event: error\ndata: {json.dumps({'error': event['error']})}\n\n"

    except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
SSE 事件格式
事件	payload	说明
status	{"status": "retrieving", "message": "..."}	状态更新
retrieved	{"status": "retrieved", "count": 5}	检索完成
content	纯文本	LLM 输出片段
done	{"answer": "...", "confidence": 0.85, "citations": []}	最终结果
clarify	{"question": "..."}	需要澄清
error	{"error": "..."}	错误信息
其他接口保持不变

# 以下接口不受影响

@router.get("/sessions")           # 会话列表
@router.get("/sessions/{id}")     # 会话历史
@router.delete("/sessions/{id}")   # 删除会话
🔄 对比：重构前后
代码量
文件	重构前	重构后
agent/agent.py	260 行	删除
agent/executor.py	292 行	删除
agent/decision.py	~150 行	删除
api/chat.py	260 行	~80 行（简化）
graph/state.py	-	50 行
graph/nodes.py	-	100 行
graph/edges.py	-	30 行
graph/graph.py	-	50 行
总计	~960 行	~310 行
代码减少：68%

🚀 迁移策略
Phase 1：新建 graph/ 模块（并行开发）
knowledge_qa/
├── agent/              # 旧代码（保留）
├── graph/              # 新代码（LangGraph）
│   ├── state.py
│   ├── nodes.py
│   ├── edges.py
│   ├── graph.py
│   └── langchain_tools.py
└── api/
    └── chat.py         # 直接修改（替换 /stream）
Phase 2：单元测试

# tests/test_graph.py

import pytest
from knowledge_qa.graph import graph

@pytest.mark.asyncio
async def test_retrieve_and_generate():
    """测试完整流程"""
    result = await graph.ainvoke({
        "query": "产品保修期",
        "knowledge_base_id": 1,
        "messages": [],
    })
    assert result.get("final_answer") or result.get("clarification_question")
Phase 3：替换上线

# 1. 确认测试通过

pytest tests/test_graph.py -v

# 2. 删除旧代码

rm knowledge_qa/agent/executor.py
rm knowledge_qa/agent/decision.py

# 3. 部署

docker-compose up -d --build
📋 依赖变更
requirements.txt

# 新增

langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.1.0

# 保留现有依赖

fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
qdrant-client==1.7.0
openai==1.12.0
🗺️ 重构路线图
周次	里程碑	交付物
Week 1	环境搭建 + 核心图	✅ graph/state.py
✅ graph/nodes.py（retrieve + generate）
✅ graph/graph.py（编译）
Week 2	决策 + 工具 + API	✅ decide 节点
✅ ToolNode 集成
✅ API /v2 接口
✅ SSE 流式
Week 3	测试 + 灰度	✅ 单元测试（覆盖率 50%）
✅ A/B 测试对比
✅ 性能测试
Week 4	上线	✅ 切换流量
✅ 删除旧代码
✅ 文档更新
资源需求：1 人 × 4 周

💡 为什么选 LangGraph？
方案	优点	缺点	适用性
LangGraph	✅ 官方 LangChain 生态
✅ 声明式流程定义
✅ 内置 ToolNode
✅ 支持检查点	⚠️ 学习成本	⭐⭐⭐⭐⭐ 完美匹配
DSPy	✅ 编译优化	⚠️ 生态小
⚠️ 不适合 RAG	⭐⭐ 不推荐
自研状态机	✅ 完全可控	❌ 重复造轮子
❌ 维护成本高	❌ 已用烂了
⚠️ 潜在问题
问题	解决方案
LangGraph API 变更	锁定版本 langgraph==0.2.0
性能开销	压测对比（预期 <5% 开销）
团队学习成本	培训 1 天 + 官方示例
🎯 总结
维度	当前	LangGraph 重构
代码量	~700 行	~230 行（减少 67%）
状态管理	分散	统一 AgentState
流程定义	代码控制	声明式图
调试	日志	标准日志 + IDE 断点
扩展	改代码	改图结构
推荐执行： 投入 1 人 × 4 周，收益明显。

文档版本： v1.0
创建日期： 2026-04-14
预计工期： 4 周（1 人）
