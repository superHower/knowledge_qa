"""
API 路由 - 知识库
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_qa.db import get_db
from knowledge_qa.schemas import (
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeBaseResponse,
    SuccessResponse,
)
from knowledge_qa.services import KnowledgeBaseService

router = APIRouter(prefix="/knowledge-bases", tags=["知识库"])


@router.post("", response_model=KnowledgeBaseResponse)
async def create_knowledge_base(
    data: KnowledgeBaseCreate,
    db: AsyncSession = Depends(get_db),
):
    """创建知识库"""
    service = KnowledgeBaseService()
    kb = await service.create(
        db=db,
        name=data.name,
        description=data.description,
        embedding_model=data.embedding_model,
        top_k=data.top_k,
        similarity_threshold=data.similarity_threshold,
    )
    return kb


@router.get("", response_model=list[KnowledgeBaseResponse])
async def list_knowledge_bases(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    is_active: Optional[bool] = None,
    db: AsyncSession = Depends(get_db),
):
    """列出知识库"""
    service = KnowledgeBaseService()
    items, total = await service.list_(
        db=db,
        page=page,
        page_size=page_size,
        is_active=is_active,
    )
    return items


@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    kb_id: int,
    db: AsyncSession = Depends(get_db),
):
    """获取知识库详情"""
    service = KnowledgeBaseService()
    kb = await service.get(db, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return kb


@router.patch("/{kb_id}", response_model=KnowledgeBaseResponse)
async def update_knowledge_base(
    kb_id: int,
    data: KnowledgeBaseUpdate,
    db: AsyncSession = Depends(get_db),
):
    """更新知识库"""
    service = KnowledgeBaseService()
    kb = await service.update(
        db=db,
        kb_id=kb_id,
        name=data.name,
        description=data.description,
        top_k=data.top_k,
        similarity_threshold=data.similarity_threshold,
    )
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return kb


@router.delete("/{kb_id}", response_model=SuccessResponse)
async def delete_knowledge_base(
    kb_id: int,
    db: AsyncSession = Depends(get_db),
):
    """删除知识库"""
    service = KnowledgeBaseService()
    success = await service.delete(db, kb_id)
    if not success:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return SuccessResponse(message="删除成功")


@router.get("/{kb_id}/stats")
async def get_knowledge_base_stats(
    kb_id: int,
    db: AsyncSession = Depends(get_db),
):
    """获取知识库统计"""
    service = KnowledgeBaseService()
    stats = await service.get_stats(db, kb_id)
    if not stats:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return stats
