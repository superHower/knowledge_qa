"""
高级 RAG 检索器（整合查询改写 + 重排序 + 质量评估）
"""

from dataclasses import dataclass
from typing import Optional, Callable
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from knowledge_qa.db.models import KnowledgeBase, DocumentChunk
from knowledge_qa.rag.embedding import EmbeddingModel
from knowledge_qa.rag.vector_store import VectorStore, VectorPoint, SearchResult
from knowledge_qa.rag.reranker import (
    BaseReranker, 
    CrossEncoderReranker,
    ScoreWeightedReranker,
)
from knowledge_qa.rag.query_rewrite import EnsembleQueryRewriter
from knowledge_qa.rag.evaluator import RetrievalEvaluator, QualityMonitor


@dataclass
class RetrievedChunk:
    """检索到的切片"""
    chunk_id: int
    content: str
    document_id: int
    document_name: str
    score: float
    rerank_score: float = 0.0
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetrievalResult:
    """检索结果"""
    query: str
    rewritten_queries: list[str]  # 改写后的查询
    chunks: list[RetrievedChunk]
    total_chunks: int
    metrics: dict = None  # 检索质量指标
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class AdvancedRAGRetriever:
    """高级 RAG 检索器
    
    整合以下能力：
    1. 查询改写 (Multi-Query + HyDE)
    2. 多路召回 (向量 + 关键词)
    3. 重排序 (Cross-Encoder)
    4. 质量评估与监控
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore = None,
        reranker: Optional[BaseReranker] = None,
        query_rewriter: Optional[EnsembleQueryRewriter] = None,
        evaluator: Optional[RetrievalEvaluator] = None,
    ):
        self.embedding_model = embedding_model
        # 如果没有提供向量存储，使用单例管理器
        if vector_store is None:
            from knowledge_qa.rag.vector_store_manager import get_vector_store
            vector_store = get_vector_store()
        self.vector_store = vector_store
        self.reranker = reranker or ScoreWeightedReranker()
        self.query_rewriter = query_rewriter
        self.evaluator = evaluator or RetrievalEvaluator(k=5)
        self.quality_monitor = QualityMonitor(self.evaluator)
        
        # 检索配置
        self.enable_query_rewrite = True
        self.enable_rerank = True
        self.enable_evaluation = True
        self.initial_top_k = 20  # 初始召回数量（重排序前）
        self.final_top_k = 5    # 最终返回数量
    
    async def retrieve(
        self,
        query: str,
        knowledge_base_id: int,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        db: Optional[AsyncSession] = None,
    ) -> RetrievalResult:
        """检索相关切片（完整流程）"""
        collection_name = f"kb_{knowledge_base_id}"
        
        # 1. 检查集合是否存在
        if not await self.vector_store.collection_exists(collection_name):
            return RetrievalResult(
                query=query,
                rewritten_queries=[query],
                chunks=[],
                total_chunks=0,
            )
        
        # 2. 查询改写
        rewritten_queries = [query]
        if self.enable_query_rewrite and self.query_rewriter:
            try:
                rewritten_queries = await self.query_rewriter.rewrite(query)
            except Exception as e:
                print(f"Query rewrite failed: {e}")
                rewritten_queries = [query]
        
        # 3. 多查询并行检索
        all_candidates = await self._multi_query_search(
            queries=rewritten_queries,
            collection_name=collection_name,
            top_k=self.initial_top_k,
        )
        
        # 4. 去重 + 合并候选集
        seen_ids = set()
        unique_candidates = []
        for candidate in all_candidates:
            if candidate.id not in seen_ids:
                seen_ids.add(candidate.id)
                unique_candidates.append(candidate)
        
        # 5. 重排序
        if self.enable_rerank and len(unique_candidates) > 1:
            unique_candidates = await self._rerank(
                query=query,
                candidates=unique_candidates,
            )
        
        # 6. 取 top_k
        final_chunks = unique_candidates[:top_k]
        
        # 7. 填充详情
        chunks = await self._enrich_chunks(
            candidates=final_chunks,
            db=db,
        )
        
        # 8. 质量评估
        metrics = {}
        if self.enable_evaluation:
            metrics = self._evaluate_candidates(query, final_chunks)
        
        # 9. 统计总数
        total = await self.vector_store.count(collection_name)
        
        return RetrievalResult(
            query=query,
            rewritten_queries=rewritten_queries,
            chunks=chunks,
            total_chunks=total,
            metrics=metrics,
        )
    
    async def _multi_query_search(
        self,
        queries: list[str],
        collection_name: str,
        top_k: int,
    ) -> list[SearchResult]:
        """多查询并行检索"""
        tasks = []
        for q in queries:
            task = self._search_single(q, collection_name, top_k)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_results = []
        for result in results:
            if isinstance(result, Exception):
                continue
            all_results.extend(result)
        
        return all_results
    
    async def _search_single(
        self,
        query: str,
        collection_name: str,
        top_k: int,
    ) -> list[SearchResult]:
        """单查询检索"""
        # 向量化查询
        query_vector = await self.embedding_model.embed_text(query)
        
        # 向量检索
        results = await self.vector_store.search(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=None,  # 先不过滤
        )
        
        return results
    
    async def _rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        """重排序"""
        if not candidates:
            return []
        
        # 提取文档内容
        docs = [c.payload.get("content", "") for c in candidates]
        vector_scores = [c.score for c in candidates]
        
        # 使用 Cross-Encoder 或加权重排序
        reranked_indices = await self.reranker.rerank(
            query=query,
            documents=docs,
            top_k=len(candidates),
            vector_scores=vector_scores,
        )
        
        # 重新排序
        reranked = []
        for idx, score in reranked_indices:
            candidate = candidates[idx]
            candidate.score = score  # 更新分数
            reranked.append(candidate)
        
        return reranked
    
    async def _enrich_chunks(
        self,
        candidates: list[SearchResult],
        db: Optional[AsyncSession],
    ) -> list[RetrievedChunk]:
        """填充切片详情"""
        chunks = []
        
        for candidate in candidates:
            chunk_id = candidate.payload.get("chunk_id")
            
            if db and chunk_id:
                # 从数据库获取详情
                stmt = select(DocumentChunk).where(DocumentChunk.id == chunk_id)
                result = await db.execute(stmt)
                chunk = result.scalar_one_or_none()
                
                if chunk:
                    chunks.append(RetrievedChunk(
                        chunk_id=chunk.id,
                        content=chunk.content,
                        document_id=chunk.document_id,
                        document_name=chunk.document.file_name if chunk.document else "",
                        score=candidate.score,
                        rerank_score=candidate.score,
                        metadata=chunk.metadata_ or {},
                    ))
                else:
                    # 从 payload 获取
                    chunks.append(RetrievedChunk(
                        chunk_id=chunk_id,
                        content=candidate.payload.get("content", ""),
                        document_id=candidate.payload.get("document_id", 0),
                        document_name=candidate.payload.get("document_name", ""),
                        score=candidate.score,
                        rerank_score=candidate.score,
                        metadata=candidate.payload.get("metadata", {}),
                    ))
            else:
                chunks.append(RetrievedChunk(
                    chunk_id=chunk_id,
                    content=candidate.payload.get("content", ""),
                    document_id=candidate.payload.get("document_id", 0),
                    document_name=candidate.payload.get("document_name", ""),
                    score=candidate.score,
                    rerank_score=candidate.score,
                    metadata=candidate.payload.get("metadata", {}),
                ))
        
        return chunks
    
    def _evaluate_candidates(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> dict:
        """评估检索质量"""
        # 这里简化处理，实际需要标注数据
        # 返回基本统计信息
        return {
            "candidate_count": len(chunks),
            "avg_score": sum(c.score for c in chunks) / len(chunks) if chunks else 0,
            "max_score": max((c.score for c in chunks), default=0),
            "min_score": min((c.score for c in chunks), default=0),
        }
    
    def get_quality_report(self) -> dict:
        """获取质量报告"""
        return self.quality_monitor.get_stats()


class ChunkIndexer:
    """切片索引器"""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore = None,
    ):
        self.embedding_model = embedding_model
        # 如果没有提供向量存储，使用单例管理器
        if vector_store is None:
            from knowledge_qa.rag.vector_store_manager import get_vector_store
            vector_store = get_vector_store()
        self.vector_store = vector_store

    async def index_chunk(
        self,
        chunk_id: int,
        content: str,
        knowledge_base_id: int,
        document_id: int,
        document_name: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """索引单个切片"""
        import hashlib
        
        vector = await self.embedding_model.embed_text(content)
        vector_id = hashlib.md5(f"{knowledge_base_id}_{chunk_id}".encode()).hexdigest()
        collection_name = f"kb_{knowledge_base_id}"

        if not await self.vector_store.collection_exists(collection_name):
            await self.vector_store.create_collection(
                collection_name=collection_name,
                vector_size=len(vector),
            )

        point = VectorPoint(
            id=vector_id,
            vector=vector,
            payload={
                "chunk_id": chunk_id,
                "content": content,
                "document_id": document_id,
                "document_name": document_name,
                "metadata": metadata or {},
            },
        )

        await self.vector_store.upsert(collection_name, [point])
        return vector_id

    async def index_chunks_batch(
        self,
        chunks: list[tuple[int, str, int, str, dict]],
        knowledge_base_id: int,
    ) -> list[str]:
        """批量索引切片"""
        import hashlib

        contents = [c[1] for c in chunks]
        vectors = await self.embedding_model.embed_texts(contents)
        collection_name = f"kb_{knowledge_base_id}"

        if not await self.vector_store.collection_exists(collection_name):
            await self.vector_store.create_collection(
                collection_name=collection_name,
                vector_size=len(vectors[0]),
            )

        points = []
        for i, (chunk_id, content, document_id, document_name, metadata) in enumerate(chunks):
            vector_id = hashlib.md5(f"{knowledge_base_id}_{chunk_id}".encode()).hexdigest()
            points.append(VectorPoint(
                id=vector_id,
                vector=vectors[i],
                payload={
                    "chunk_id": chunk_id,
                    "content": content,
                    "document_id": document_id,
                    "document_name": document_name,
                    "metadata": metadata,
                },
            ))

        await self.vector_store.upsert(collection_name, points)
        return [p.id for p in points]

    async def delete_chunk(
        self,
        chunk_id: int,
        knowledge_base_id: int,
    ) -> bool:
        """删除切片"""
        import hashlib
        collection_name = f"kb_{knowledge_base_id}"
        vector_id = hashlib.md5(f"{knowledge_base_id}_{chunk_id}".encode()).hexdigest()
        return await self.vector_store.delete(collection_name, [vector_id])

    async def delete_document_vectors(
        self,
        document_id: int,
        knowledge_base_id: int,
        chunk_ids: list[int],
    ) -> bool:
        """删除文档的所有向量"""
        import hashlib
        collection_name = f"kb_{knowledge_base_id}"
        vector_ids = [
            hashlib.md5(f"{knowledge_base_id}_{cid}".encode()).hexdigest()
            for cid in chunk_ids
        ]
        return await self.vector_store.delete(collection_name, vector_ids)

    async def delete_collection(
        self,
        knowledge_base_id: int,
    ) -> bool:
        """删除整个知识库的向量"""
        collection_name = f"kb_{knowledge_base_id}"
        return await self.vector_store.delete_collection(collection_name)


# 保持向后兼容
RAGRetriever = AdvancedRAGRetriever
