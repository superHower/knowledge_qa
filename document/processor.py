"""
文档处理服务
"""

import hashlib
import shutil
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from knowledge_qa.core.config import settings
from knowledge_qa.db.models import Document, DocumentChunk, KnowledgeBase
from .parsers import ParserFactory
from .splitter import TextSplitter, TextChunk


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.parser_factory = ParserFactory()
        self.text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    
    async def process_document(
        self,
        file_path: str,
        file_name: str,
        knowledge_base_id: int,
        db: AsyncSession,
        user_id: str | None = None,
    ) -> Document:
        """处理单个文档"""
        # 1. 创建文档记录
        file_size = Path(file_path).stat().st_size
        file_type = Path(file_path).suffix.lower()
        
        document = Document(
            knowledge_base_id=knowledge_base_id,
            file_name=file_name,
            file_path=file_path,
            file_type=file_type,
            file_size=file_size,
            status="processing",
        )
        db.add(document)
        await db.flush()
        
        try:
            # 2. 解析文档
            parser = self.parser_factory.get_parser(file_path)
            if not parser:
                raise ValueError(f"不支持的文件类型: {file_type}")
            
            doc_content = await parser.parse(file_path)
            
            # 更新文档元信息
            document.title = doc_content.title
            document.author = doc_content.metadata.get("author") if doc_content.metadata else None
            document.page_count = doc_content.metadata.get("page_count") if doc_content.metadata else None
            
            # 3. 切片
            chunks = self.text_splitter.split_text_recursive(doc_content.content)
            
            # 4. 保存切片
            for chunk_data in chunks:
                chunk = DocumentChunk(
                    document_id=document.id,
                    content=chunk_data.content,
                    content_hash=chunk_data.content_hash,
                    chunk_index=chunk_data.chunk_index,
                    start_char=chunk_data.start_char,
                    end_char=chunk_data.end_char,
                    metadata_={
                        "source": file_path,
                        "title": doc_content.title,
                    },
                )
                db.add(chunk)
            
            document.chunk_count = len(chunks)
            document.status = "completed"
            
        except Exception as e:
            document.status = "failed"
            document.error_message = str(e)
            raise
        
        await db.commit()
        await db.refresh(document)
        
        return document
    
    async def process_document_stream(
        self,
        file_path: str,
        file_name: str,
        knowledge_base_id: int,
        db: AsyncSession,
    ) -> AsyncGenerator[dict, None]:
        """流式处理文档（用于大文件）"""
        yield {"status": "start", "message": "开始处理文档"}
        
        # 解析
        yield {"status": "parsing", "message": "解析文档"}
        parser = self.parser_factory.get_parser(file_path)
        if not parser:
            raise ValueError(f"不支持的文件类型")
        
        doc_content = await parser.parse(file_path)
        yield {"status": "parsed", "title": doc_content.title}
        
        # 切片
        yield {"status": "chunking", "message": "切分文档"}
        chunks = self.text_splitter.split_text_recursive(doc_content.content)
        yield {"status": "chunked", "chunk_count": len(chunks)}
        
        # 保存
        yield {"status": "saving", "message": "保存切片"}
        for chunk_data in chunks:
            yield {"status": "progress", "chunk": chunk_data.chunk_index, "total": len(chunks)}
        
        yield {"status": "completed", "message": "处理完成"}


class FileStorage:
    """文件存储管理"""
    
    def __init__(self, upload_dir: str = None):
        self.upload_dir = Path(upload_dir or settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_file(
        self,
        file_content: bytes,
        file_name: str,
        knowledge_base_id: int,
    ) -> str:
        """保存文件到存储目录"""
        # 按知识库ID组织目录
        kb_dir = self.upload_dir / str(knowledge_base_id)
        kb_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成唯一文件名
        file_hash = hashlib.md5(file_content).hexdigest()[:8]
        file_ext = Path(file_name).suffix
        unique_name = f"{Path(file_name).stem}_{file_hash}{file_ext}"
        
        file_path = kb_dir / unique_name
        file_path.write_bytes(file_content)
        
        return str(file_path)
    
    async def delete_file(self, file_path: str) -> bool:
        """删除文件"""
        try:
            Path(file_path).unlink(missing_ok=True)
            return True
        except Exception:
            return False
    
    async def cleanup_knowledge_base_files(self, knowledge_base_id: int) -> int:
        """清理知识库下所有文件"""
        kb_dir = self.upload_dir / str(knowledge_base_id)
        
        if not kb_dir.exists():
            return 0
        
        deleted_count = 0
        for file_path in kb_dir.iterdir():
            file_path.unlink()
            deleted_count += 1
        
        kb_dir.rmdir()
        
        return deleted_count
