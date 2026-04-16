"""
文档处理模块
"""

from .base import BaseDocumentParser, DocumentContent
from .parsers import ParserFactory, TextParser, PDFParser, DocxParser, HTMLParser, CSVParser
from .splitter import TextSplitter, TextChunk
from .processor import DocumentProcessor, FileStorage
from .structured_splitter import (
    StructureAwareSplitter,
    DocumentStructure,
    HierarchicalChunk,
    ChunkLevel,
    MultiGranularityIndexer,
)

__all__ = [
    "BaseDocumentParser",
    "DocumentContent",
    "ParserFactory",
    "TextParser",
    "PDFParser",
    "DocxParser",
    "HTMLParser",
    "CSVParser",
    "TextSplitter",
    "TextChunk",
    "DocumentProcessor",
    "FileStorage",
    "StructureAwareSplitter",
    "DocumentStructure",
    "HierarchicalChunk",
    "ChunkLevel",
    "MultiGranularityIndexer",
]
