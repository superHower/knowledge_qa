"""
智能文档结构感知切片器
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class ChunkLevel(str, Enum):
    """切片层级"""
    CHAPTER = "chapter"      # 章节级别
    SECTION = "section"     # 小节级别
    PARAGRAPH = "paragraph"  # 段落级别
    SENTENCE = "sentence"    # 句子级别


@dataclass
class DocumentStructure:
    """文档结构"""
    title: Optional[str] = None
    chapters: list[dict] = None  # [{"title": ..., "level": ..., "content": ..., "start": ..., "end": ...}]
    
    def __post_init__(self):
        if self.chapters is None:
            self.chapters = []


@dataclass
class HierarchicalChunk:
    """层级切片"""
    content: str
    level: ChunkLevel
    chunk_index: int
    parent_index: Optional[int]  # 父切片索引
    path: str  # 路径，如 "1.2.3"
    metadata: dict


class StructureAwareSplitter:
    """文档结构感知切片器
    
    感知文档的标题层级、章节边界，
    在保持语义完整性的前提下进行智能切片
    """

    # Markdown/HTML 标题模式
    HEADING_PATTERNS = [
        r'^#{1,6}\s+(.+)$',           # Markdown: # Title
        r'<h([1-6])[^>]*>(.+)</h\1>', # HTML: <h1>Title</h1>
        r'^(\d+[\.、\s]+[^\n]+)$',    # 数字标题: 1. Title
        r'^([A-Z][\.\)]\s+[^\n]+)$',  # 大写字母: A. Title
        r'^(第[一二三四五六七八九十百千]+[章节篇部分]?\s*[：:\s].+)$',  # 中文: 第一章
    ]

    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 800,
        overlap: int = 50,
        preserve_structure: bool = True,
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.preserve_structure = preserve_structure
    
    def parse_structure(self, text: str) -> DocumentStructure:
        """解析文档结构"""
        structure = DocumentStructure()
        
        lines = text.split('\n')
        current_chapter = None
        chapter_start = 0
        
        for i, line in enumerate(lines):
            heading_info = self._match_heading(line)
            
            if heading_info:
                # 保存之前的章节
                if current_chapter:
                    current_chapter['end'] = i
                    structure.chapters.append(current_chapter)
                
                # 开始新章节
                level, title = heading_info
                current_chapter = {
                    'level': level,
                    'title': title,
                    'content': '',
                    'start': i,
                    'end': len(lines),
                    'path': '',
                }
                
                if level == 1:
                    structure.title = title
            elif current_chapter:
                current_chapter['content'] += line + '\n'
        
        # 保存最后一个章节
        if current_chapter:
            current_chapter['end'] = len(lines)
            structure.chapters.append(current_chapter)
        
        # 生成路径
        self._generate_paths(structure.chapters)
        
        return structure
    
    def _match_heading(self, line: str) -> Optional[tuple[int, str]]:
        """匹配标题"""
        line = line.strip()
        if not line:
            return None
        
        for pattern in self.HEADING_PATTERNS:
            match = re.match(pattern, line, re.MULTILINE)
            if match:
                if pattern.startswith(r'^#{1,6}'):
                    level = len(match.group(0).split()[0])
                    return level, match.group(1).strip()
                elif '<h' in pattern:
                    level = int(match.group(1))
                    return level, match.group(2).strip()
                else:
                    return 3, match.group(1).strip()  # 默认3级
        
        return None
    
    def _generate_paths(self, chapters: list[dict]):
        """生成章节路径"""
        path_stack = []
        
        for chapter in chapters:
            level = chapter['level']
            
            # 调整路径栈
            while len(path_stack) >= level:
                path_stack.pop()
            
            if not path_stack:
                path_stack.append(1)
            else:
                path_stack[-1] += 1
            
            chapter['path'] = '.'.join(str(p) for p in path_stack)
    
    def split_with_structure(
        self,
        text: str,
        structure: Optional[DocumentStructure] = None,
    ) -> list[HierarchicalChunk]:
        """基于文档结构进行切片"""
        if structure is None:
            structure = self.parse_structure(text)
        
        chunks = []
        chunk_index = 0
        parent_index = None
        
        for chapter in structure.chapters:
            level = chapter['level']
            content = chapter['content'].strip()
            
            if not content:
                continue
            
            # 根据层级和内容大小决定切片方式
            if len(content) <= self.max_chunk_size:
                # 内容适中，直接作为一片
                chunks.append(HierarchicalChunk(
                    content=content,
                    level=ChunkLevel(level) if level <= 3 else ChunkLevel.SECTION,
                    chunk_index=chunk_index,
                    parent_index=parent_index,
                    path=chapter['path'],
                    metadata={
                        'title': chapter['title'],
                        'level': level,
                    },
                ))
                chunk_index += 1
                parent_index = chunk_index - 1
            else:
                # 内容太长，需要进一步切分
                sub_chunks = self._split_long_content(
                    content,
                    level,
                    chapter['path'],
                    parent_index,
                    chunk_index,
                )
                chunks.extend(sub_chunks)
                chunk_index = len(chunks)
                parent_index = chunk_index - 1 if chunks else None
        
        # 计算内容哈希
        for i, chunk in enumerate(chunks):
            chunk.content_hash = hashlib.md5(chunk.content.encode()).hexdigest()
        
        return chunks
    
    def _split_long_content(
        self,
        content: str,
        level: int,
        path: str,
        parent_index: Optional[int],
        start_index: int,
    ) -> list[HierarchicalChunk]:
        """切分长内容"""
        chunks = []
        
        # 尝试按段落分割
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_chunk = ""
        current_index = start_index
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 如果当前段落加上现有内容超过上限
            if len(current_chunk) + len(para) + 2 > self.max_chunk_size:
                if current_chunk:
                    chunks.append(HierarchicalChunk(
                        content=current_chunk.strip(),
                        level=ChunkLevel.PARAGRAPH,
                        chunk_index=current_index,
                        parent_index=parent_index,
                        path=f"{path}.{current_index - start_index + 1}",
                        metadata={},
                    ))
                    current_index += 1
                
                # 如果单个段落就超过上限，按句子切
                if len(para) > self.max_chunk_size:
                    sentence_chunks = self._split_by_sentence(
                        para, level, path, parent_index, current_index
                    )
                    chunks.extend(sentence_chunks)
                    current_index = len(chunks) + start_index
                    current_chunk = ""
                else:
                    current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # 保存最后一个切片
        if current_chunk.strip():
            chunks.append(HierarchicalChunk(
                content=current_chunk.strip(),
                level=ChunkLevel.PARAGRAPH,
                chunk_index=current_index,
                parent_index=parent_index,
                path=f"{path}.{current_index - start_index + 1}",
                metadata={},
            ))
        
        return chunks
    
    def _split_by_sentence(
        self,
        content: str,
        level: int,
        path: str,
        parent_index: Optional[int],
        start_index: int,
    ) -> list[HierarchicalChunk]:
        """按句子切分"""
        chunks = []
        
        # 中英文句子分隔符
        sentences = re.split(r'([。！？；\n]|(?<=[a-zA-Z])\.(?=[A-Z]))', content)
        
        # 合并句子和其后的标点
        merged_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            sent = sentences[i].strip()
            punct = sentences[i + 1] if i + 1 < len(sentences) else ''
            if sent:
                merged_sentences.append(sent + punct)
        
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            merged_sentences.append(sentences[-1].strip())
        
        current_chunk = ""
        current_index = start_index
        
        for sent in merged_sentences:
            if len(current_chunk) + len(sent) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(HierarchicalChunk(
                        content=current_chunk.strip(),
                        level=ChunkLevel.SENTENCE,
                        chunk_index=current_index,
                        parent_index=parent_index,
                        path=f"{path}.s{current_index - start_index + 1}",
                        metadata={},
                    ))
                    current_index += 1
                    current_chunk = ""
            else:
                current_chunk += sent
        
        if current_chunk.strip():
            chunks.append(HierarchicalChunk(
                content=current_chunk.strip(),
                level=ChunkLevel.SENTENCE,
                chunk_index=current_index,
                parent_index=parent_index,
                path=f"{path}.s{current_index - start_index + 1}",
                metadata={},
            ))
        
        return chunks


class MultiGranularityIndexer:
    """多粒度索引器
    
    对同一文档创建不同层级的索引，
    检索时根据查询长度选择合适的粒度
    """

    def __init__(self, splitter: StructureAwareSplitter):
        self.splitter = splitter
    
    def create_multi_granularity_chunks(
        self,
        text: str,
    ) -> dict[ChunkLevel, list[HierarchicalChunk]]:
        """创建多粒度切片
        
        Returns:
            {level: [chunks], ...}
        """
        structure = self.splitter.parse_structure(text)
        
        # 章节级切片
        chapter_chunks = [
            HierarchicalChunk(
                content=ch['content'] or "",
                level=ChunkLevel.SECTION,
                chunk_index=i,
                parent_index=None,
                path=ch['path'],
                metadata={'title': ch['title'], 'level': ch['level']},
            )
            for i, ch in enumerate(structure.chapters)
            if ch['content']
        ]
        
        # 段落级切片
        paragraph_chunks = self.splitter.split_with_structure(text, structure)
        
        # 句子级切片（用于精确匹配）
        sentence_chunks = []
        for ch in paragraph_chunks:
            if len(ch.content) > 500:
                sentences = re.split(r'([。！？；\n])', ch.content)
                merged = []
                for i in range(0, len(sentences) - 1, 2):
                    sent = sentences[i].strip()
                    punct = sentences[i + 1] if i + 1 < len(sentences) else ''
                    if sent:
                        merged.append(sent + punct)
                
                for j, s in enumerate(merged):
                    sentence_chunks.append(HierarchicalChunk(
                        content=s,
                        level=ChunkLevel.SENTENCE,
                        chunk_index=j,
                        parent_index=ch.chunk_index,
                        path=f"{ch.path}.{j + 1}",
                        metadata={},
                    ))
        
        return {
            ChunkLevel.CHAPTER: chapter_chunks,
            ChunkLevel.SECTION: paragraph_chunks,
            ChunkLevel.PARAGRAPH: paragraph_chunks,
            ChunkLevel.SENTENCE: sentence_chunks,
        }
    
    def select_granularity(
        self,
        query: str,
        chunks: dict[ChunkLevel, list[HierarchicalChunk]],
    ) -> ChunkLevel:
        """根据查询长度选择合适的粒度"""
        query_len = len(query)
        
        if query_len < 20:
            # 短查询，用句子级
            return ChunkLevel.SENTENCE
        elif query_len < 50:
            # 中等查询，用段落级
            return ChunkLevel.PARAGRAPH
        else:
            # 长查询，用章节级
            return ChunkLevel.CHAPTER
