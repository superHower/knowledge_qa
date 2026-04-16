"""
向量存储管理器 - 单例模式，确保向量数据在应用生命周期内持久化
"""

from typing import Optional

from knowledge_qa.core.config import settings
from knowledge_qa.rag.vector_store import InMemoryVectorStore, QdrantStore, VectorStore


class VectorStoreManager:
    """向量存储管理器（单例）"""
    
    _instance = None
    _vector_store: VectorStore = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            
            # 根据配置选择向量存储
            if settings.QDRANT_HOST:
                cls._vector_store = QdrantStore(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                )
            else:
                cls._vector_store = InMemoryVectorStore(vector_size=1536)
                
        return cls._instance
    
    @property
    def vector_store(self) -> VectorStore:
        """获取向量存储实例"""
        return self._vector_store
    
    def get_collection_stats(self) -> dict:
        """获取所有集合的统计信息"""
        if isinstance(self._vector_store, InMemoryVectorStore):
            stats = {}
            for collection_name, points in self._vector_store.collections.items():
                stats[collection_name] = len(points)
            return stats
        return {}
    
    async def get_stats(self) -> dict:
        """获取向量存储统计信息"""
        store_type = type(self._vector_store).__name__
        return {
            "type": store_type,
            "host": settings.QDRANT_HOST if hasattr(self, '_vector_store') else None,
        }


# 全局实例
vector_store_manager = VectorStoreManager()


def get_vector_store() -> VectorStore:
    """获取向量存储实例"""
    return vector_store_manager.vector_store
