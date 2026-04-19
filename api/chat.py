"""
API 路由 - 对话

使用依赖注入模式，复用 LLM 和 Retriever 实例。
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import json

from knowledge_qa.db import get_db
from knowledge_qa.db.models import ChatSession, ChatMessage
from knowledge_qa.schemas import (
    ChatMessageCreate,
    ChatMessageResponse,
    ChatSessionResponse,
    ChatHistoryResponse,
)
from knowledge_qa.agent import KnowledgeQAAgent
from knowledge_qa.agent.llm import OpenAILLM
from knowledge_qa.rag import OpenAIEmbedding, AdvancedRAGRetriever
from knowledge_qa.agent.tool import ToolRegistry, KnowledgeBaseTool, CalculatorTool, DateTimeTool
from knowledge_qa.core.config import settings
from knowledge_qa.graph import GraphDependencies, set_dependencies, create_graph

router = APIRouter(prefix="/chat", tags=["对话"])


# ============================================================
# 依赖注入：单例 LLM 和 Retriever
# ============================================================

_llm_instance: Optional[OpenAILLM] = None
_embedding_instance: Optional[OpenAIEmbedding] = None
_retriever_instance: Optional[AdvancedRAGRetriever] = None
_initialized: bool = False


def _get_llm() -> OpenAILLM:
    """获取或创建 LLM 单例"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = OpenAILLM(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
        )
    return _llm_instance


def _get_embedding() -> OpenAIEmbedding:
    """获取或创建 Embedding 单例"""
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = OpenAIEmbedding(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model=settings.OPENAI_EMBEDDING_MODEL,
        )
    return _embedding_instance


def _get_retriever(knowledge_base_id: int) -> AdvancedRAGRetriever:
    """获取或创建 Retriever 单例（按 knowledge_base_id）"""
    global _retriever_instance
    if _retriever_instance is None:
        from knowledge_qa.rag import get_vector_store
        vector_store = get_vector_store()
        _retriever_instance = AdvancedRAGRetriever(
            embedding=_get_embedding(),
            vector_store=vector_store,
        )
    return _retriever_instance


def init_graph_dependencies(knowledge_base_id: int) -> GraphDependencies:
    """
    初始化 Graph 依赖
    
    在应用启动或切换知识库时调用。
    """
    global _initialized
    
    deps = GraphDependencies(
        llm=_get_llm(),
        embedding=_get_embedding(),
        retriever=_get_retriever(knowledge_base_id),
        default_top_k=5,
        default_temperature=0.7,
        max_tokens=2000,
    )
    
    # 设置全局依赖
    set_dependencies(deps)
    _initialized = True
    
    return deps


def get_agent(knowledge_base_id: int) -> KnowledgeQAAgent:
    """
    获取 Agent 实例（复用 LLM 和工具）
    
    使用单例模式，避免每次请求都创建新实例。
    """
    # 确保依赖已初始化
    if not _initialized:
        init_graph_dependencies(knowledge_base_id)
    
    llm = _get_llm()
    
    # 创建工具注册表
    registry = ToolRegistry()
    registry.register(KnowledgeBaseTool(_get_retriever(knowledge_base_id), knowledge_base_id))
    registry.register(CalculatorTool())
    registry.register(DateTimeTool())
    
    return KnowledgeQAAgent(llm, registry)


def get_graph(knowledge_base_id: int):
    """
    获取编译好的 Graph 实例
    
    使用依赖注入，共享 LLM 和 Retriever。
    """
    if not _initialized:
        init_graph_dependencies(knowledge_base_id)
    
    from knowledge_qa.graph import graph as _graph
    return _graph


# ============================================================
# API 路由
# ============================================================

def format_sse(event_name: str, data) -> str:
    """格式化 SSE 消息"""
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_name}\ndata: {payload}\n\n"


@router.post("", response_model=ChatMessageResponse)
async def chat(
    data: ChatMessageCreate,
    knowledge_base_id: int,
    db: AsyncSession = Depends(get_db),
):
    """问答对话"""
    agent = get_agent(knowledge_base_id)
    
    # 获取对话历史
    history = None
    if data.session_id:
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == data.session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(10)
        )
        result = await db.execute(stmt)
        messages = list(result.scalars().all())
        if messages:
            history = [
                {"role": m.role, "content": m.content}
                for m in reversed(messages)
            ]
    
    # 调用 Agent
    response = await agent.chat(
        query=data.query,
        knowledge_base_id=knowledge_base_id,
        session_id=data.session_id,
        conversation_history=history,
        top_k=data.top_k,
        temperature=data.temperature,
        stream=False,
        db=db,
    )
    
    # 保存对话
    session, message = await agent.save_conversation(
        query=data.query,
        answer=response.answer,
        session_id=data.session_id,
        knowledge_base_id=knowledge_base_id,
        retrieved_chunks=response.citations,
        db=db,
    )
    
    return ChatMessageResponse(
        session_id=session.id,
        message_id=message.id,
        answer=response.answer,
        sources=response.sources,
        citations=response.citations,
        usage=response.usage,
    )


@router.post("/stream")
async def chat_stream(
    data: ChatMessageCreate,
    knowledge_base_id: int,
    db: AsyncSession = Depends(get_db),
):
    """流式问答对话 (SSE)"""
    agent = get_agent(knowledge_base_id)
    
    # 获取对话历史
    history = None
    if data.session_id:
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == data.session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(10)
        )
        result = await db.execute(stmt)
        messages = list(result.scalars().all())
        if messages:
            history = [
                {"role": m.role, "content": m.content}
                for m in reversed(messages)
            ]
    
    async def event_generator():
        try:
            async for event in agent.chat_stream(
                query=data.query,
                knowledge_base_id=knowledge_base_id,
                session_id=data.session_id,
                conversation_history=history,
                top_k=data.top_k,
                temperature=data.temperature,
                db=db,
            ):
                event_type = event["type"]

                if event_type in {
                    "run.start",
                    "status",
                    "node.start",
                    "node.done",
                    "retrieval",
                    "answer.start",
                }:
                    yield format_sse(event_type, event)
                elif event_type == "answer.delta":
                    yield format_sse("answer.delta", event)
                    # 兼容旧前端：继续输出原始文本增量事件
                    yield format_sse("content", event["delta"])
                elif event_type == "clarify":
                    yield format_sse("clarify", event)
                elif event_type == "error":
                    yield format_sse("error", event)
                elif event_type == "answer.done":
                    # 保存对话
                    session, message = await agent.save_conversation(
                        query=data.query,
                        answer=event["answer"],
                        session_id=data.session_id,
                        knowledge_base_id=knowledge_base_id,
                        retrieved_chunks=event.get("retrieved_chunks", []),
                        db=db,
                    )
                    saved_data = {
                        "session_id": session.id,
                        "message_id": message.id,
                        "answer": event["answer"],
                    }
                    yield format_sse("message.saved", saved_data)

                    done_data = {
                        "session_id": session.id,
                        "message_id": message.id,
                        "answer": event["answer"],
                        "sources": event.get("sources", []),
                        "citations": event.get("citations", []),
                        "confidence": event.get("confidence", 0.85),
                    }
                    yield format_sse("answer.done", done_data)
                    # 兼容旧前端：保留 done 事件
                    yield format_sse("done", done_data)
        except Exception as e:
            error_data = {
                "error": str(e),
                "type": type(e).__name__,
            }
            yield format_sse("error", error_data)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions", response_model=list[ChatSessionResponse])
async def list_sessions(
    knowledge_base_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """列出会话"""
    stmt = (
        select(ChatSession)
        .where(ChatSession.knowledge_base_id == knowledge_base_id)
        .order_by(ChatSession.updated_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/sessions/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: int,
    db: AsyncSession = Depends(get_db),
):
    """获取会话历史"""
    # 获取会话
    session_stmt = select(ChatSession).where(ChatSession.id == session_id)
    session_result = await db.execute(session_stmt)
    session = session_result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 获取消息
    msg_stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
    )
    msg_result = await db.execute(msg_stmt)
    messages = msg_result.scalars().all()
    
    return ChatHistoryResponse(
        session_id=session_id,
        messages=[
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ],
    )


@router.delete("/sessions/{session_id}", response_model=dict)
async def delete_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
):
    """删除会话"""
    stmt = select(ChatSession).where(ChatSession.id == session_id)
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    await db.delete(session)
    await db.commit()
    
    return {"success": True, "message": "删除成功"}
