"""
自主决策引擎

核心能力：
1. 置信度评估
2. 澄清询问
3. 错误处理
4. 策略选择
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class Decision(str, Enum):
    """决策类型"""
    ANSWER = "answer"                    # 直接回答
    CLARIFY = "clarify"                  # 追问澄清
    TRANSFER = "transfer"                # 转人工
    RETRY = "retry"                      # 重试
    REFUSE = "refuse"                    # 拒答


@dataclass
class ConfidenceResult:
    """置信度评估结果"""
    confidence: float
    decision: Decision
    reasons: list[str]
    suggestion: Optional[str] = None


@dataclass
class ClarificationResult:
    """澄清询问结果"""
    question: str
    context: str
    options: list[str] = None  # 可选的选项


class ConfidenceEvaluator:
    """置信度评估器
    
    评估回答的置信度，决定是否需要追问或拒答
    """

    def __init__(
        self,
        high_threshold: float = 0.8,
        medium_threshold: float = 0.5,
        low_threshold: float = 0.3,
    ):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.low_threshold = low_threshold

    def evaluate(
        self,
        chunks: list[dict],
        query: str,
        answer: str,
        tool_results: list[dict],
    ) -> ConfidenceResult:
        """评估置信度"""
        reasons = []
        confidence_scores = []
        
        # 1. 基于检索结果评估
        if chunks:
            avg_relevance = sum(c.get("score", 0) for c in chunks) / len(chunks)
            max_relevance = max(c.get("score", 0) for c in chunks)
            
            if avg_relevance < 0.5:
                confidence_scores.append(0.3)
                reasons.append(f"检索相关度偏低 (平均: {avg_relevance:.2f})")
            elif avg_relevance > 0.7:
                confidence_scores.append(0.9)
                reasons.append(f"检索相关度高 (平均: {avg_relevance:.2f})")
            else:
                confidence_scores.append(0.6)
                reasons.append(f"检索相关度中等 (平均: {avg_relevance:.2f})")
            
            # 检查召回数量
            if len(chunks) < 2:
                confidence_scores.append(-0.1)
                reasons.append("召回结果较少，可能遗漏信息")
        else:
            confidence_scores.append(0.2)
            reasons.append("未找到相关文档")
        
        # 2. 基于工具调用结果评估
        if tool_results:
            failed_tools = [t for t in tool_results if not t.get("success", True)]
            success_rate = (len(tool_results) - len(failed_tools)) / len(tool_results)
            
            confidence_scores.append(success_rate)
            if failed_tools:
                reasons.append(f"{len(failed_tools)} 个工具调用失败")
        else:
            confidence_scores.append(0.5)
        
        # 3. 基于查询复杂度评估
        query_complexity = self._assess_query_complexity(query)
        confidence_scores.append(query_complexity)
        
        # 4. 基于回答内容评估
        answer_confidence = self._assess_answer_quality(answer)
        confidence_scores.append(answer_confidence)
        if answer_confidence < 0.5:
            reasons.append("回答内容不完整或模糊")
        
        # 综合评分
        final_confidence = sum(confidence_scores) / len(confidence_scores)
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # 决策
        if final_confidence >= self.high_threshold:
            decision = Decision.ANSWER
        elif final_confidence >= self.medium_threshold:
            decision = Decision.ANSWER  # 可以回答，但添加置信度说明
        elif final_confidence >= self.low_threshold:
            decision = Decision.CLARIFY
            reasons.append("置信度不足，建议追问用户")
        else:
            decision = Decision.REFUSE
            reasons.append("置信度过低，不适合自动回答")
        
        return ConfidenceResult(
            confidence=round(final_confidence, 3),
            decision=decision,
            reasons=reasons,
        )
    
    def _assess_query_complexity(self, query: str) -> float:
        """评估查询复杂度"""
        # 简单启发式
        score = 0.7  # 默认中等
        
        # 包含比较词
        if any(w in query for w in ["和", "与", "比较", "区别", "哪个"]):
            score -= 0.1
        
        # 包含数字/计算
        if any(c.isdigit() for c in query):
            score -= 0.1
        
        # 问句过长
        if len(query) > 100:
            score -= 0.1
        
        # 问句过短
        if len(query) < 5:
            score -= 0.2
        
        return max(0.3, min(0.9, score))
    
    def _assess_answer_quality(self, answer: str) -> float:
        """评估回答质量"""
        score = 0.7
        
        # 回答过短
        if len(answer) < 20:
            score -= 0.2
        
        # 回答过长（可能是废话）
        if len(answer) > 2000:
            score -= 0.1
        
        # 包含拒答话术
        refuse_phrases = ["无法", "不知道", "不清楚", "没有找到"]
        if any(p in answer for p in refuse_phrases):
            score -= 0.3
        
        # 包含引用
        if "根据" in answer or "来源" in answer:
            score += 0.1
        
        return max(0.2, min(1.0, score))


class ClarificationGenerator:
    """澄清询问生成器
    
    当置信度不足时，生成追问
    """

    QUESTION_TEMPLATES = {
        "missing_entity": "您是指{entity}吗？",
        "missing_time": "您想了解的是哪个时间段？",
        "missing_aspect": "您更关注哪方面？",
        "ambiguous": "您的问题有点模糊，可以具体说明一下吗？",
        "multiple_match": "我找到多个相关内容，您想了解哪个方面？",
    }

    async def generate(
        self,
        query: str,
        low_confidence_reasons: list[str],
        retrieved_chunks: list[dict],
    ) -> ClarificationResult:
        """生成澄清询问"""
        # 根据原因生成对应问题
        
        if "未找到相关文档" in low_confidence_reasons:
            question = "抱歉，知识库中暂未收录这方面的内容。您可以换个关键词试试，或者联系相关人员获取帮助。"
            return ClarificationResult(
                question=question,
                context="知识库中没有找到相关信息",
            )
        
        if "检索相关度偏低" in low_confidence_reasons:
            # 尝试从已有结果中提取可能的实体
            entities = []
            for chunk in retrieved_chunks:
                content = chunk.get("content", "")
                # 简单提取可能的实体（实际可用 NER）
                if len(content) > 50:
                    entities.append(content[:50] + "...")
            
            if entities:
                return ClarificationResult(
                    question="您是想了解以下哪个方面？",
                    context="\n".join(entities[:3]),
                    options=entities[:3],
                )
        
        if "召回结果较少" in low_confidence_reasons:
            return ClarificationResult(
                question="您的问题涉及的内容可能需要更具体的描述",
                context="您可以尝试补充更多细节",
            )
        
        # 默认澄清
        return ClarificationResult(
            question="您能更具体地描述一下您的问题吗？",
            context="为了给您更准确的回答",
        )


class DecisionEngine:
    """决策引擎
    
    整合置信度评估和澄清生成，
    做出最终决策
    """

    def __init__(
        self,
        confidence_evaluator: ConfidenceEvaluator,
        clarification_generator: ClarificationGenerator,
    ):
        self.confidence_evaluator = confidence_evaluator
        self.clarification_generator = clarification_generator
    
    async def decide(
        self,
        query: str,
        chunks: list[dict],
        answer: str,
        tool_results: list[dict],
    ) -> tuple[Decision, ConfidenceResult, Optional[ClarificationResult]]:
        """做出决策
        
        Returns:
            (决策, 置信度结果, 澄清询问（如果有）)
        """
        # 评估置信度
        confidence_result = self.confidence_evaluator.evaluate(
            chunks=chunks,
            query=query,
            answer=answer,
            tool_results=tool_results,
        )
        
        # 根据决策采取行动
        if confidence_result.decision == Decision.ANSWER:
            return confidence_result.decision, confidence_result, None
        
        elif confidence_result.decision == Decision.CLARIFY:
            clarification = await self.clarification_generator.generate(
                query=query,
                low_confidence_reasons=confidence_result.reasons,
                retrieved_chunks=chunks,
            )
            return confidence_result.decision, confidence_result, clarification
        
        elif confidence_result.decision == Decision.REFUSE:
            clarification = ClarificationResult(
                question="抱歉，我无法回答这个问题",
                context="建议您联系人工客服获取帮助",
            )
            return confidence_result.decision, confidence_result, clarification
        
        else:
            return Decision.RETRY, confidence_result, None


class ErrorHandler:
    """错误处理器"""
    
    def __init__(self):
        self.error_history: list[dict] = []
    
    def handle(
        self,
        error: Exception,
        context: dict,
    ) -> dict:
        """处理错误
        
        Returns:
            {"action": "retry/transfer/refuse", "message": str, "recoverable": bool}
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        record = {
            "error_type": error_type,
            "error_msg": error_msg,
            "context": context,
        }
        self.error_history.append(record)
        
        # 根据错误类型决定处理方式
        if "timeout" in error_msg.lower():
            return {
                "action": "retry",
                "message": "请求超时，正在重试...",
                "recoverable": True,
            }
        
        if "rate limit" in error_msg.lower():
            return {
                "action": "retry",
                "message": "请求过于频繁，请稍后重试",
                "recoverable": True,
            }
        
        if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            return {
                "action": "refuse",
                "message": "系统配置问题，请联系管理员",
                "recoverable": False,
            }
        
        # 默认重试
        return {
            "action": "retry",
            "message": f"发生错误: {error_msg}",
            "recoverable": True,
        }
    
    def get_error_stats(self) -> dict:
        """获取错误统计"""
        if not self.error_history:
            return {}
        
        error_types = {}
        for record in self.error_history[-100:]:
            et = record["error_type"]
            error_types[et] = error_types.get(et, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(self.error_history[-10:]),
            "error_types": error_types,
        }
