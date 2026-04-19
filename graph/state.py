"""
LangGraph AgentState 定义
"""

from typing import TypedDict, Optional, List, Dict, Any


class AgentState(TypedDict, total=False):
    """Agent 全局状态（LangGraph StateGraph 使用）"""

    # 输入
    query: str
    knowledge_base_id: int
    session_id: Optional[int]
    conversation_history: Optional[List[Dict[str, Any]]]
    top_k: int
    temperature: float

    # 检索结果
    retrieved_chunks: List[Dict[str, Any]]

    # LLM 生成结果
    current_thought: Optional[str]

    # 决策结果
    final_answer: Optional[str]
    confidence: float
    needs_clarification: bool
    clarification_question: Optional[str]

    # 错误信息
    error: Optional[str]
