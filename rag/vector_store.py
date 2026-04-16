"""
向量数据库服务
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import uuid


@dataclass
class VectorPoint:
    """向量点"""
    id: str
    vector: list[float]
    payload: dict


@dataclass
class SearchResult:
    """检索结果"""
    id: str
    score: float
    payload: dict


class VectorStore(ABC):
    """向量数据库抽象"""
    
    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int = 1536,
        distance: str = "Cosine",
    ) -> bool:
        """创建集合"""
        pass
    
    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        pass
    
    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        pass
    
    @abstractmethod
    async def upsert(
        self,
        collection_name: str,
        points: list[VectorPoint],
    ) -> bool:
        """插入/更新向量"""
        pass
    
    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[dict] = None,
    ) -> list[SearchResult]:
        """向量检索"""
        pass
    
    @abstractmethod
    async def delete(
        self,
        collection_name: str,
        ids: list[str],
    ) -> bool:
        """删除向量"""
        pass
    
    @abstractmethod
    async def count(self, collection_name: str) -> int:
        """统计向量数量"""
        pass


class QdrantStore(VectorStore):
    """Qdrant 向量数据库"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6400,
        grpc_port: int = 6334,
        api_key: Optional[str] = None,
    ):
        from qdrant_client import AsyncQdrantClient
        
        self.client = AsyncQdrantClient(
            host=host,
            port=port,
            grpc_port=grpc_port,
            api_key=api_key,
            timeout=30,
        )
    
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int = 1536,
        distance: str = "Cosine",
    ) -> bool:
        """创建集合"""
        from qdrant_client.http.models import Distance, VectorParams
        
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclidean": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        
        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance_map.get(distance, Distance.COSINE),
            ),
        )
        return True
    
    async def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        await self.client.delete_collection(collection_name)
        return True
    
    async def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        collections = await self.client.get_collections()
        return collection_name in [c.name for c in collections.collections]
    
    async def upsert(
        self,
        collection_name: str,
        points: list[VectorPoint],
    ) -> bool:
        """插入/更新向量"""
        from qdrant_client.http.models import PointStruct
        
        points_struct = [
            PointStruct(
                id=p.id,
                vector=p.vector,
                payload=p.payload,
            )
            for p in points
        ]
        
        await self.client.upsert(
            collection_name=collection_name,
            points=points_struct,
        )
        return True
    
    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[dict] = None,
    ) -> list[SearchResult]:
        """向量检索"""
        from qdrant_client.http.models import Filter, FieldCondition, MatchAny, Range
        
        search_params = {"limit": top_k}
        
        if score_threshold is not None:
            search_params["score_threshold"] = score_threshold
        
        filter_condition = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchAny(any=value),
                        )
                    )
                elif isinstance(value, dict):
                    range_filter = {}
                    if "gte" in value:
                        range_filter["gte"] = value["gte"]
                    if "lte" in value:
                        range_filter["lte"] = value["lte"]
                    if range_filter:
                        conditions.append(
                            FieldCondition(
                                key=key,
                                range=Range(**range_filter),
                            )
                        )
                else:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchAny(any=[value]),
                        )
                    )
            
            if conditions:
                filter_condition = Filter(must=conditions)
        
        results = await self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=filter_condition,
            limit=top_k,
            score_threshold=score_threshold,
        )
        
        return [
            SearchResult(
                id=str(r.id),
                score=r.score,
                payload=r.payload or {},
            )
            for r in results.points
        ]
    
    async def delete(
        self,
        collection_name: str,
        ids: list[str],
    ) -> bool:
        """删除向量"""
        from qdrant_client.http.models import PointsSelector, PointIdsList
        
        await self.client.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=ids),
        )
        return True
    
    async def count(self, collection_name: str) -> int:
        """统计向量数量"""
        result = await self.client.get_collection(collection_name)
        # 兼容新旧版本
        return getattr(result, 'points_count', getattr(result, 'vectors_count', 0))


class InMemoryVectorStore(VectorStore):
    """内存向量存储（用于开发/测试）"""
    
    def __init__(self, vector_size: int = 1536):
        self.vector_size = vector_size
        self.collections: dict[str, dict[str, VectorPoint]] = {}
    
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int = 1536,
        distance: str = "Cosine",
    ) -> bool:
        self.collections[collection_name] = {}
        return True
    
    async def delete_collection(self, collection_name: str) -> bool:
        self.collections.pop(collection_name, None)
        return True
    
    async def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections
    
    async def upsert(
        self,
        collection_name: str,
        points: list[VectorPoint],
    ) -> bool:
        if collection_name not in self.collections:
            await self.create_collection(collection_name, self.vector_size)
        
        for point in points:
            self.collections[collection_name][point.id] = point
        
        return True
    
    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[dict] = None,
    ) -> list[SearchResult]:
        from numpy.linalg import norm
        from numpy import dot
        
        if collection_name not in self.collections:
            return []
        
        def cosine_sim(a: list[float], b: list[float]) -> float:
            return dot(a, b) / (norm(a) * norm(b) + 1e-8)
        
        results = []
        for point in self.collections[collection_name].values():
            # 简单的过滤
            if filters:
                match = True
                for key, value in filters.items():
                    if point.payload.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            score = cosine_sim(query_vector, point.vector)
            
            if score_threshold is not None and score < score_threshold:
                continue
            
            results.append(SearchResult(
                id=point.id,
                score=float(score),
                payload=point.payload,
            ))
        
        # 排序并返回 top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    async def delete(
        self,
        collection_name: str,
        ids: list[str],
    ) -> bool:
        if collection_name in self.collections:
            for id_ in ids:
                self.collections[collection_name].pop(id_, None)
        return True
    
    async def count(self, collection_name: str) -> int:
        return len(self.collections.get(collection_name, {}))
