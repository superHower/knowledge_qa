"""
服务层 - 文档服务
"""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from knowledge_qa.db.models import Document, DocumentChunk, KnowledgeBase
from knowledge_qa.document import DocumentProcessor, FileStorage
from knowledge_qa.rag import ChunkIndexer


class DocumentService:
    """文档服务"""
    
    def __init__(
        self,
        document_processor: DocumentProcessor,
        file_storage: FileStorage,
        chunk_indexer: ChunkIndexer,
    ):
        self.processor = document_processor
        self.file_storage = file_storage
        self.chunk_indexer = chunk_indexer
    
    async def upload_and_process(
        self,
        db: AsyncSession,
        file_content: bytes,
        file_name: str,
        knowledge_base_id: int,
    ) -> Document:
        """上传并处理文档"""
        # 1. 保存文件
        file_path = await self.file_storage.save_file(
            file_content=file_content,
            file_name=file_name,
            knowledge_base_id=knowledge_base_id,
        )
        
        # 2. 处理文档（解析 + 切片）
        document = await self.processor.process_document(
            file_path=file_path,
            file_name=file_name,
            knowledge_base_id=knowledge_base_id,
            db=db,
        )
        
        # 3. 索引切片到向量数据库
        stmt = select(DocumentChunk).where(DocumentChunk.document_id == document.id)
        result = await db.execute(stmt)
        chunks = result.scalars().all()
        
        if chunks and self.chunk_indexer:
            await self.chunk_indexer.index_chunks_batch(
                chunks=[
                    (
                        chunk.id,
                        chunk.content,
                        chunk.document_id,
                        document.file_name,
                        chunk.metadata_,
                    )
                    for chunk in chunks
                ],
                knowledge_base_id=knowledge_base_id,
            )
        
        return document
    
    async def get(
        self,
        db: AsyncSession,
        doc_id: int,
    ) -> Optional[Document]:
        """获取文档"""
        stmt = select(Document).where(Document.id == doc_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def list_(
        self,
        db: AsyncSession,
        knowledge_base_id: int,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
    ) -> tuple[list[Document], int]:
        """列出文档"""
        stmt = select(Document).where(Document.knowledge_base_id == knowledge_base_id)
        
        if status:
            stmt = stmt.where(Document.status == status)
        
        # 统计总数
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_result = await db.execute(count_stmt)
        total = total_result.scalar()
        
        # 分页
        stmt = stmt.order_by(Document.updated_at.desc())
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        
        result = await db.execute(stmt)
        items = result.scalars().all()
        
        return list(items), total
    
    async def delete(
        self,
        db: AsyncSession,
        doc_id: int,
    ) -> bool:
        """删除文档"""
        doc = await self.get(db, doc_id)
        if not doc:
            return False
        
        # 获取切片ID列表
        chunk_stmt = select(DocumentChunk.id).where(DocumentChunk.document_id == doc_id)
        chunk_result = await db.execute(chunk_stmt)
        chunk_ids = [row[0] for row in chunk_result.fetchall()]
        
        # 删除向量索引
        if self.chunk_indexer and chunk_ids:
            await self.chunk_indexer.delete_document_vectors(
                document_id=doc_id,
                knowledge_base_id=doc.knowledge_base_id,
                chunk_ids=chunk_ids,
            )
        
        # 删除文件
        await self.file_storage.delete_file(doc.file_path)
        
        # 删除数据库记录（级联删除 chunks）
        await db.delete(doc)
        await db.commit()
        
        return True
    
    async def reprocess(
        self,
        db: AsyncSession,
        doc_id: int,
    ) -> Document:
        """重新处理文档"""
        from sqlalchemy import delete
        
        doc = await self.get(db, doc_id)
        if not doc:
            raise ValueError(f"文档不存在: {doc_id}")
        
        # 记录旧文件路径
        old_file_path = doc.file_path
        kb_id = doc.knowledge_base_id
        
        # 删除旧切片向量
        chunk_stmt = select(DocumentChunk.id).where(DocumentChunk.document_id == doc_id)
        chunk_result = await db.execute(chunk_stmt)
        chunk_ids = [row[0] for row in chunk_result.fetchall()]
        
        if self.chunk_indexer and chunk_ids:
            await self.chunk_indexer.delete_document_vectors(
                document_id=doc_id,
                knowledge_base_id=kb_id,
                chunk_ids=chunk_ids,
            )
        
        # 删除旧切片记录
        delete_stmt = delete(DocumentChunk).where(DocumentChunk.document_id == doc_id)
        await db.execute(delete_stmt)
        
        # 重置状态
        doc.status = "processing"
        doc.error_message = None
        await db.commit()
        
        # 读取旧文件
        try:
            file_content = open(old_file_path, "rb").read()
        except FileNotFoundError:
            raise ValueError(f"原始文件不存在: {old_file_path}")
        
        # 重新处理
        return await self.upload_and_process(
            db=db,
            file_content=file_content,
            file_name=doc.file_name,
            knowledge_base_id=kb_id,
        )
