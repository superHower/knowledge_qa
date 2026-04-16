"""
服务层 - 知识库服务
"""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from knowledge_qa.db.models import KnowledgeBase, Document, DocumentChunk
from knowledge_qa.rag import ChunkIndexer


class KnowledgeBaseService:
    """知识库服务"""
    
    def __init__(self, chunk_indexer: Optional[ChunkIndexer] = None):
        self.chunk_indexer = chunk_indexer
    
    async def create(
        self,
        db: AsyncSession,
        name: str,
        description: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> KnowledgeBase:
        """创建知识库"""
        kb = KnowledgeBase(
            name=name,
            description=description,
            embedding_model=embedding_model,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        db.add(kb)
        await db.commit()
        await db.refresh(kb)
        return kb
    
    async def get(
        self,
        db: AsyncSession,
        kb_id: int,
    ) -> Optional[KnowledgeBase]:
        """获取知识库"""
        stmt = select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def list_(
        self,
        db: AsyncSession,
        page: int = 1,
        page_size: int = 20,
        is_active: Optional[bool] = None,
    ) -> tuple[list[KnowledgeBase], int]:
        """列出知识库"""
        stmt = select(KnowledgeBase)
        
        if is_active is not None:
            stmt = stmt.where(KnowledgeBase.is_active == is_active)
        
        # 统计总数
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_result = await db.execute(count_stmt)
        total = total_result.scalar()
        
        # 分页
        stmt = stmt.order_by(KnowledgeBase.updated_at.desc())
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        
        result = await db.execute(stmt)
        items = result.scalars().all()
        
        return list(items), total
    
    async def update(
        self,
        db: AsyncSession,
        kb_id: int,
        **kwargs,
    ) -> Optional[KnowledgeBase]:
        """更新知识库"""
        kb = await self.get(db, kb_id)
        if not kb:
            return None
        
        for key, value in kwargs.items():
            if value is not None and hasattr(kb, key):
                setattr(kb, key, value)
        
        await db.commit()
        await db.refresh(kb)
        return kb
    
    async def delete(
        self,
        db: AsyncSession,
        kb_id: int,
    ) -> bool:
        """删除知识库"""
        kb = await self.get(db, kb_id)
        if not kb:
            return False
        
        # 删除向量索引
        if self.chunk_indexer:
            await self.chunk_indexer.delete_collection(kb_id)
        
        await db.delete(kb)
        await db.commit()
        return True
    
    async def get_stats(
        self,
        db: AsyncSession,
        kb_id: int,
    ) -> dict:
        """获取知识库统计信息"""
        kb = await self.get(db, kb_id)
        if not kb:
            return {}
        
        # 文档数量
        doc_stmt = select(func.count()).select_from(Document).where(
            Document.knowledge_base_id == kb_id
        )
        doc_result = await db.execute(doc_stmt)
        doc_count = doc_result.scalar()
        
        # 切片数量
        chunk_stmt = select(func.count()).select_from(DocumentChunk).join(Document).where(
            Document.knowledge_base_id == kb_id
        )
        chunk_result = await db.execute(chunk_stmt)
        chunk_count = chunk_result.scalar()
        
        return {
            "id": kb.id,
            "name": kb.name,
            "document_count": doc_count,
            "chunk_count": chunk_count,
        }
