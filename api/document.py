"""
API 路由 - 文档
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_qa.db import get_db
from knowledge_qa.schemas import (
    DocumentResponse,
    DocumentListResponse,
    SuccessResponse,
)
from knowledge_qa.services import DocumentService
from knowledge_qa.document import DocumentProcessor, FileStorage
from knowledge_qa.rag import ChunkIndexer, OpenAIEmbedding
from knowledge_qa.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["文档"])


def get_document_service() -> DocumentService:
    """获取文档服务实例"""
    embedding = OpenAIEmbedding(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        model=settings.OPENAI_EMBEDDING_MODEL,
    )
    # 使用单例向量存储
    from knowledge_qa.rag import get_vector_store
    vector_store = get_vector_store()
    chunk_indexer = ChunkIndexer(embedding, vector_store)
    
    processor = DocumentProcessor()
    storage = FileStorage()
    
    return DocumentService(processor, storage, chunk_indexer)


@router.post("", response_model=DocumentResponse)
async def upload_document(
    knowledge_base_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """上传并处理文档"""
    # 验证文件类型
    allowed_types = [".txt", ".md", ".pdf", ".docx", ".doc", ".html", ".csv"]
    file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型，仅支持: {', '.join(allowed_types)}"
        )
    
    # 读取文件内容
    content = await file.read()
    
    # 处理文档
    service = get_document_service()
    
    try:
        logger.info(f"开始处理文档: {file.filename}")
        doc = await service.upload_and_process(
            db=db,
            file_content=content,
            file_name=file.filename,
            knowledge_base_id=knowledge_base_id,
        )
        logger.info(f"文档处理完成: {file.filename}, chunks: {doc.chunk_count}")
        return DocumentResponse.model_validate(doc)
    except Exception as e:
        logger.error(f"文档处理失败: {file.filename}, error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    knowledge_base_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """列出文档"""
    service = get_document_service()
    docs, total = await service.list_(
        db=db,
        knowledge_base_id=knowledge_base_id,
        page=page,
        page_size=page_size,
        status=status,
    )
    items = [DocumentResponse.model_validate(doc) for doc in docs]
    return DocumentListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: int,
    db: AsyncSession = Depends(get_db),
):
    """获取文档详情"""
    service = get_document_service()
    doc = await service.get(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    return DocumentResponse.model_validate(doc)


@router.delete("/{doc_id}", response_model=SuccessResponse)
async def delete_document(
    doc_id: int,
    db: AsyncSession = Depends(get_db),
):
    """删除文档"""
    service = get_document_service()
    success = await service.delete(db, doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="文档不存在")
    return SuccessResponse(message="删除成功")


@router.post("/{doc_id}/reprocess", response_model=DocumentResponse)
async def reprocess_document(
    doc_id: int,
    db: AsyncSession = Depends(get_db),
):
    """重新处理文档"""
    logger.info(f"开始重新处理文档: {doc_id}")
    service = get_document_service()
    try:
        doc = await service.reprocess(db, doc_id)
        logger.info(f"文档重新处理完成: {doc_id}")
        return DocumentResponse.model_validate(doc)
    except ValueError as e:
        logger.error(f"文档不存在: {doc_id}, error: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"文档重新处理失败: {doc_id}, error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")
