"""
文档处理器基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator


@dataclass
class DocumentContent:
    """文档内容"""
    title: str | None = None
    content: str = ""
    metadata: dict | None = None


class BaseDocumentParser(ABC):
    """文档解析器基类"""
    
    @abstractmethod
    async def parse(self, file_path: str) -> DocumentContent:
        """解析文档"""
        pass
    
    @abstractmethod
    async def extract_text(self, file_path: str) -> str:
        """提取纯文本"""
        pass
    
    async def extract_text_stream(self, file_path: str) -> AsyncGenerator[str, None]:
        """流式提取文本（用于大文件）"""
        text = await self.extract_text(file_path)
        yield text
