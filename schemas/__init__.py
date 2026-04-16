"""
Pydantic Schemas
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


# ===================
# 知识库 Schemas
# ===================

class KnowledgeBaseCreate(BaseModel):
    """创建知识库"""
    name: str = Field(..., min_length=1, max_length=255, description="知识库名称")
    description: Optional[str] = Field(None, description="知识库描述")
    embedding_model: str = Field("text-embedding-3-small", description="Embedding 模型")
    top_k: int = Field(5, ge=1, le=20, description="检索召回数量")
    similarity_threshold: float = Field(0.5, ge=0, le=1, description="相似度阈值")


class KnowledgeBaseUpdate(BaseModel):
    """更新知识库"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    top_k: Optional[int] = Field(None, ge=1, le=20)
    similarity_threshold: Optional[float] = Field(None, ge=0, le=1)


class KnowledgeBaseResponse(BaseModel):
    """知识库响应"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: Optional[str]
    embedding_model: str
    top_k: int
    similarity_threshold: float
    is_active: bool
    document_count: int = 0
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime


# ===================
# 文档 Schemas
# ===================

class DocumentResponse(BaseModel):
    """文档响应"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    knowledge_base_id: int
    file_name: str
    file_type: str
    file_size: int
    title: Optional[str]
    status: str
    chunk_count: int
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    items: list[DocumentResponse]
    total: int
    page: int
    page_size: int


# ===================
# 对话 Schemas
# ===================

class ChatMessageCreate(BaseModel):
    """创建聊天消息"""
    query: str = Field(..., min_length=1, max_length=5000, description="用户问题")
    session_id: Optional[int] = Field(None, description="会话ID，不传则创建新会话")
    top_k: Optional[int] = Field(None, ge=1, le=20)
    temperature: Optional[float] = Field(None, ge=0, le=2)
    stream: bool = Field(False, description="是否流式输出")


class ChatMessageResponse(BaseModel):
    """聊天消息响应"""
    session_id: int
    message_id: int
    answer: str
    sources: list[dict]
    citations: list[dict]
    usage: Optional[dict]


class ChatSessionResponse(BaseModel):
    """会话响应"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    knowledge_base_id: int
    session_name: str
    user_id: Optional[str]
    message_count: int
    created_at: datetime
    updated_at: datetime


class ChatHistoryResponse(BaseModel):
    """聊天历史响应"""
    session_id: int
    messages: list[dict]


# ===================
# 通用 Schemas
# ===================

class SuccessResponse(BaseModel):
    """成功响应"""
    success: bool = True
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """错误响应"""
    success: bool = False
    error: str
    detail: Optional[str] = None


class PaginatedResponse(BaseModel):
    """分页响应基类"""
    items: list
    total: int
    page: int
    page_size: int
    total_pages: int
