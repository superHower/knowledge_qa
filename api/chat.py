"""
API 路由 - 对话
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import json
import sse_starlette.sse as sse

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

router = APIRouter(prefix="/chat", tags=["对话"])


def get_agent(knowledge_base_id: int) -> KnowledgeQAAgent:
    """获取 Agent 实例"""
    llm = OpenAILLM(
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_MODEL,
    )
    embedding = OpenAIEmbedding(api_key=settings.OPENAI_API_KEY)
    # 使用单例向量存储管理器
    from knowledge_qa.rag import get_vector_store
    vector_store = get_vector_store()
    retriever = AdvancedRAGRetriever(embedding, vector_store)
    
    # 创建工具注册表
    registry = ToolRegistry()
    registry.register(KnowledgeBaseTool(retriever, knowledge_base_id))
    registry.register(CalculatorTool())
    registry.register(DateTimeTool())
    
    return KnowledgeQAAgent(llm, registry)


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
                if event["type"] == "status":
                    yield f"event: status\ndata: {json.dumps(event)}\n\n"
                elif event["type"] == "retrieval":
                    yield f"event: retrieval\ndata: {json.dumps(event)}\n\n"
                elif event["type"] == "content":
                    yield f"event: content\ndata: {event['delta']}\n\n"
                elif event["type"] == "done":
                    # 保存对话
                    session, message = await agent.save_conversation(
                        query=data.query,
                        answer=event["answer"],
                        session_id=data.session_id,
                        knowledge_base_id=knowledge_base_id,
                        retrieved_chunks=[],
                        db=db,
                    )
                    done_data = json.dumps({
                        "session_id": session.id,
                        "message_id": message.id,
                        "answer": event["answer"],
                        "sources": event.get("sources", []),
                    })
                    yield f"event: done\ndata: {done_data}\n\n"
        except Exception as e:
            error_data = json.dumps({
                "error": str(e),
                "type": type(e).__name__,
            })
            yield f"event: error\ndata: {error_data}\n\n"
    
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
