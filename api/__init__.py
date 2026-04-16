"""
API 路由
"""

from fastapi import APIRouter

from .knowledge_base import router as kb_router
from .document import router as doc_router
from .chat import router as chat_router

api_router = APIRouter()

api_router.include_router(kb_router)
api_router.include_router(doc_router)
api_router.include_router(chat_router)

__all__ = ["api_router"]
