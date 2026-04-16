"""
重排序器
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseReranker(ABC):
    """重排序基类"""
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """重排序
        
        Args:
            query: 查询
            documents: 文档列表
            top_k: 返回前k个
            
        Returns:
            [(doc_index, score), ...] 按相关性排序
        """
        pass


class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder 重排序
    
    使用 Cross-Encoder 模型进行更精确的相关性打分
    Cross-Encoder: [query, document] → relevance_score
    比 Bi-Encoder 的向量相似度更准确，但速度较慢
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 16,
    ):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name, device=device)
        except ImportError:
            self.model = None
            print("Warning: sentence-transformers not installed, using fallback reranker")
        
        self.batch_size = batch_size
    
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """重排序"""
        if not documents:
            return []
        
        if self.model is None:
            # Fallback: 返回原始顺序
            return [(i, 1.0 / (i + 1)) for i in range(len(documents))]
        
        # 构建 query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # 批量预测
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        
        # 排序
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        
        # 按分数降序排列
        ranked = sorted(
            [(i, float(score)) for i, score in enumerate(scores)],
            key=lambda x: x[1],
            reverse=True,
        )
        
        return ranked[:top_k]


class TfidfReranker(BaseReranker):
    """TF-IDF 重排序
    
    结合 BM25 和 TF-IDF 进行重排序
    """

    def __init__(self):
        self.vectorizer = None
    
    def _tokenize(self, text: str) -> set[str]:
        """简单分词"""
        import re
        # 简单的中英文分词
        tokens = re.findall(r'[\w]+', text.lower())
        return set(tokens)
    
    def _calculate_bm25(
        self,
        query: str,
        document: str,
        avg_doc_len: float,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        """计算 BM25 分数"""
        from collections import Counter
        
        doc_tokens = self._tokenize(document)
        query_tokens = self._tokenize(query)
        
        doc_len = len(doc_tokens)
        doc_freq = Counter(doc_tokens)
        
        score = 0.0
        for token in query_tokens:
            if token not in doc_freq:
                continue
            
            tf = doc_freq[token]
            # 简化的 IDF（实际应该用全局语料库）
            idf = 1.0
            
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
            
            score += idf * numerator / denominator
        
        return score
    
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """重排序"""
        if not documents:
            return []
        
        # 计算平均文档长度
        avg_len = np.mean([len(d) for d in documents])
        
        # 计算每个文档的 BM25 分数
        scores = []
        for i, doc in enumerate(documents):
            score = self._calculate_bm25(query, doc, avg_len)
            scores.append((i, score))
        
        # 排序
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]


class ReciprocalRankReranker(BaseReranker):
    """倒数排序融合 (RRF)
    
    将多个排序结果进行融合
    """

    def __init__(self, rerankers: list[BaseReranker], k: int = 60):
        self.rerankers = rerankers
        self.k = k  # RRF 公式中的常数
    
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """融合多个排序结果"""
        if not documents:
            return []
        
        # 收集所有排序结果
        all_rankings = []
        for reranker in self.rerankers:
            try:
                ranking = await reranker.rerank(query, documents, len(documents))
                all_rankings.append(ranking)
            except Exception:
                continue
        
        if not all_rankings:
            return [(i, 1.0) for i in range(len(documents))]
        
        # RRF 融合
        rrf_scores = [0.0] * len(documents)
        
        for ranking in all_rankings:
            for rank, (doc_idx, score) in enumerate(ranking):
                # RRF 公式: 1 / (k + rank)
                rrf_scores[doc_idx] += 1.0 / (self.k + rank + 1)
        
        # 排序
        ranked = sorted(
            [(i, score) for i, score in enumerate(rrf_scores)],
            key=lambda x: x[1],
            reverse=True,
        )
        
        return ranked[:top_k]


class ScoreWeightedReranker(BaseReranker):
    """分数加权重排序
    
    结合多种信号进行加权评分
    """

    def __init__(
        self,
        vector_weight: float = 0.4,
        bm25_weight: float = 0.3,
        exact_match_weight: float = 0.2,
        length_penalty: float = 0.1,
    ):
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.exact_match_weight = exact_match_weight
        self.length_penalty = length_penalty
    
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        vector_scores: Optional[list[float]] = None,
    ) -> list[tuple[int, float]]:
        """加权重排序"""
        if not documents:
            return []
        
        import re
        
        # 计算各维度分数
        scores = []
        for i, doc in enumerate(documents):
            doc_lower = doc.lower()
            query_lower = query.lower()
            query_tokens = set(re.findall(r'\w+', query_lower))
            doc_tokens = set(re.findall(r'\w+', doc_lower))
            
            # 向量相似度
            vector_score = vector_scores[i] if vector_scores else 0.5
            
            # BM25 分数（简化版）
            common = query_tokens & doc_tokens
            bm25_score = len(common) / len(query_tokens) if query_tokens else 0
            
            # 精确匹配分数
            exact_score = 1.0 if query_lower in doc_lower else 0.0
            
            # 长度惩罚（偏好适中文档）
            length_pen = 1.0 - abs(len(doc) - 200) / 1000  # 200字左右最佳
            length_pen = max(0.5, min(1.0, length_pen))
            
            # 综合分数
            final_score = (
                self.vector_weight * vector_score +
                self.bm25_weight * bm25_score +
                self.exact_match_weight * exact_score +
                self.length_penalty * length_pen
            )
            
            scores.append((i, final_score))
        
        # 排序
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]
