"""
LangGraph AgentState 定义
"""

from typing import TypedDict, Optional, List, Dict, Any, Literal


class AgentState(TypedDict, total=False):
    """Agent 全局状态（LangGraph StateGraph 使用）"""

    # ==================== 输入参数 ====================
    query: str
    knowledge_base_id: int
    session_id: Optional[int]
    conversation_history: Optional[List[Dict[str, Any]]]
    top_k: int
    temperature: float

    # ==================== 检索结果 ====================
    retrieved_chunks: List[Dict[str, Any]]

    # ==================== LLM 生成结果 ====================
    current_thought: Optional[str]

    # ==================== 决策结果 ====================
    final_answer: Optional[str]
    confidence: float
    needs_clarification: bool
    clarification_question: Optional[str]

    # ==================== 流程控制（条件边） ====================
    # 当前步骤
    current_step: Literal["retrieve", "generate", "decide", "clarify", "rewrite"]
    # 检索轮次（用于限制最大重试次数）
    retrieval_round: int
    # 澄清类型
    clarification_type: Optional[Literal["retry", "user_input", "fallback"]]

    # ==================== 错误信息 ====================
    error: Optional[str]

    # ==================== 元数据 ====================
    # 来源文档列表（用于去重）
    retrieved_sources: List[str]


def get_default_state() -> AgentState:
    """获取默认状态"""
    return AgentState(
        top_k=5,
        temperature=0.7,
        retrieved_chunks=[],
        current_step="retrieve",
        retrieval_round=0,
        confidence=0.0,
        needs_clarification=False,
        retrieved_sources=[],
    )
