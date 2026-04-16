"""
文本切片器
"""

import hashlib
from dataclasses import dataclass
from typing import Iterator


@dataclass
class TextChunk:
    """文本切片"""
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    content_hash: str
    
    @classmethod
    def create(
        cls,
        content: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
    ) -> "TextChunk":
        """创建切片并计算hash"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return cls(
            content=content,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            content_hash=content_hash,
        )


class TextSplitter:
    """文本切片器"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = "\n",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def split_text(self, text: str) -> list[TextChunk]:
        """拆分文本为切片"""
        chunks = []
        
        # 按段落分割
        paragraphs = text.split(self.separator)
        
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # 如果当前段落加上现有内容超过 chunk_size
            if len(current_chunk) + len(para) + 1 > self.chunk_size:
                # 保存当前切片
                if current_chunk:
                    chunks.append(TextChunk.create(
                        content=current_chunk.strip(),
                        chunk_index=chunk_index,
                        start_char=current_start,
                        end_char=current_start + len(current_chunk),
                    ))
                    chunk_index += 1
                
                # 开始新的切片，保留 overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # 取最后 overlap 个字符作为新切片的开头
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_start = current_start + len(current_chunk) - len(overlap_text)
                    current_chunk = overlap_text + "\n" + para
                else:
                    current_start = current_start + len(current_chunk) + 1
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n" + para
                else:
                    current_chunk = para
        
        # 保存最后一个切片
        if current_chunk.strip():
            chunks.append(TextChunk.create(
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=current_start + len(current_chunk),
            ))
        
        return chunks
    
    def split_text_recursive(self, text: str) -> list[TextChunk]:
        """递归拆分文本（更智能的切分方式）"""
        chunks = []
        
        # 递归分割器
        def split_recursive(
            text: str,
            start: int,
            depth: int = 0,
        ) -> Iterator[TextChunk]:
            """递归分割"""
            if len(text) <= self.chunk_size:
                yield TextChunk.create(
                    content=text.strip(),
                    chunk_index=len(chunks),
                    start_char=start,
                    end_char=start + len(text),
                )
                return
            
            # 尝试在不同层级分割
            separators = [
                "\n\n",  # 段落级别
                "\n",    # 行级别
                "。",    # 句子级别（中文）
                ". ",    # 句子级别（英文）
                " ",     # 词级别
            ]
            
            sep = separators[min(depth, len(separators) - 1)]
            
            parts = text.split(sep)
            current_part = ""
            
            for part in parts:
                if len(current_part) + len(part) + len(sep) <= self.chunk_size:
                    current_part += part + sep
                else:
                    if current_part.strip():
                        yield TextChunk.create(
                            content=current_part.strip(),
                            chunk_index=len(chunks),
                            start_char=start,
                            end_char=start + len(current_part),
                        )
                        start += len(current_part)
                        # 保留 overlap
                        if self.chunk_overlap > 0 and len(current_part) > self.chunk_overlap:
                            overlap = current_part[-self.chunk_overlap:]
                            start -= len(overlap)
                            current_part = overlap + sep + part + sep
                        else:
                            current_part = part + sep
                    else:
                        # 单个部分太长，递归处理
                        yield from split_recursive(
                            part, start, depth + 1
                        )
                        start += len(part)
                        current_part = ""
            
            if current_part.strip():
                yield TextChunk.create(
                    content=current_part.strip(),
                    chunk_index=len(chunks),
                    start_char=start,
                    end_char=start + len(current_part),
                )
        
        for chunk in split_recursive(text, 0):
            chunks.append(chunk)
        
        return chunks
