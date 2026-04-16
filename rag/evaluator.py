"""
检索质量评估器
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class RetrievalMetrics:
    """检索质量指标"""
    precision_at_k: float      # Precision@K
    recall_at_k: float         # Recall@K
    mrr: float                 # Mean Reciprocal Rank
    ndcg_at_k: float          # NDCG@K
    coverage: float            # 召回覆盖率
    noise_ratio: float        # 噪音比例


class RetrievalEvaluator:
    """检索质量评估器
    
    用于评估检索结果的质量，计算多种指标
    """
    
    def __init__(self, k: int = 5):
        self.k = k
    
    def evaluate(
        self,
        retrieved_ids: list[str],
        relevant_ids: list[str],
    ) -> RetrievalMetrics:
        """评估检索结果
        
        Args:
            retrieved_ids: 检索返回的ID列表（按相关性排序）
            relevant_ids: 实际相关的ID列表
            
        Returns:
            检索质量指标
        """
        retrieved_set = set(retrieved_ids[:self.k])
        relevant_set = set(relevant_ids)
        
        # 交集
        true_positives = retrieved_set & relevant_set
        
        # Precision@K: 检索结果中相关的比例
        precision = len(true_positives) / min(self.k, len(retrieved_ids)) if retrieved_ids else 0
        
        # Recall@K: 召回的相关文档比例
        recall = len(true_positives) / len(relevant_set) if relevant_set else 0
        
        # MRR: 第一个相关结果的位置倒数
        mrr = 0.0
        for i, rid in enumerate(retrieved_ids):
            if rid in relevant_set:
                mrr = 1.0 / (i + 1)
                break
        
        # NDCG@K: 归一化折损累计增益
        ndcg = self._calculate_ndcg(retrieved_ids, relevant_ids)
        
        # 覆盖率
        coverage = len(true_positives) / len(relevant_set) if relevant_set else 0
        
        # 噪音比例
        noise_ratio = 1 - (len(true_positives) / min(self.k, len(retrieved_ids))) if retrieved_ids else 1
        
        return RetrievalMetrics(
            precision_at_k=round(precision, 4),
            recall_at_k=round(recall, 4),
            mrr=round(mrr, 4),
            ndcg_at_k=round(ndcg, 4),
            coverage=round(coverage, 4),
            noise_ratio=round(noise_ratio, 4),
        )
    
    def _calculate_ndcg(
        self,
        retrieved: list[str],
        relevant: set[str],
    ) -> float:
        """计算 NDCG"""
        def dcg(scores: list[int]) -> float:
            return sum((2**s - 1) / np.log2(i + 2) for i, s in enumerate(scores))
        
        # 计算 DCG
        relevance_scores = [1 if rid in relevant else 0 for rid in retrieved[:self.k]]
        dcg_value = dcg(relevance_scores)
        
        # 计算 IDCG（理想状态下的 DCG）
        ideal_scores = [1] * min(len(relevant), self.k)
        idcg_value = dcg(ideal_scores)
        
        if idcg_value == 0:
            return 0.0
        
        return dcg_value / idcg_value
    
    def evaluate_batch(
        self,
        results: list[tuple[list[str], list[str]]],
    ) -> dict[str, float]:
        """批量评估
        
        Args:
            results: [(retrieved_ids, relevant_ids), ...]
        """
        metrics_list = [self.evaluate(r[0], r[1]) for r in results]
        
        return {
            "avg_precision": np.mean([m.precision_at_k for m in metrics_list]),
            "avg_recall": np.mean([m.recall_at_k for m in metrics_list]),
            "avg_mrr": np.mean([m.mrr for m in metrics_list]),
            "avg_ndcg": np.mean([m.ndcg_at_k for m in metrics_list]),
            "avg_coverage": np.mean([m.coverage for m in metrics_list]),
            "avg_noise_ratio": np.mean([m.noise_ratio for m in metrics_list]),
        }


class QualityMonitor:
    """检索质量监控
    
    持续监控检索质量，自动触发优化
    """
    
    def __init__(self, evaluator: RetrievalEvaluator):
        self.evaluator = evaluator
        self.history: list[dict] = []
        self.thresholds = {
            "precision": 0.7,
            "recall": 0.8,
            "ndcg": 0.6,
        }
    
    def record(self, query: str, retrieved: list[str], relevant: list[str]):
        """记录一次检索"""
        metrics = self.evaluator.evaluate(retrieved, relevant)
        self.history.append({
            "query": query,
            "retrieved": retrieved,
            "relevant": relevant,
            "metrics": metrics,
        })
        
        # 检查是否低于阈值
        if metrics.precision_at_k < self.thresholds["precision"]:
            return {"alert": "low_precision", "query": query}
        if metrics.recall_at_k < self.thresholds["recall"]:
            return {"alert": "low_recall", "query": query}
        
        return None
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        if not self.history:
            return {}
        
        recent = self.history[-100:]  # 最近100条
        recent_metrics = [r["metrics"] for r in recent]
        
        return {
            "total_queries": len(self.history),
            "recent_precision": np.mean([m.precision_at_k for m in recent_metrics]),
            "recent_recall": np.mean([m.recall_at_k for m in recent_metrics]),
            "recent_ndcg": np.mean([m.ndcg_at_k for m in recent_metrics]),
            "low_quality_queries": len([
                r for r in recent 
                if r["metrics"].precision_at_k < self.thresholds["precision"]
            ]),
        }
