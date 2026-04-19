"""
LangGraph 节点实现

节点：retrieve → generate → decide

依赖通过 GraphDependencies 注入，避免在节点内部创建实例。
"""

from typing import TypedDict, Literal, Optional, List, Dict, Any

from knowledge_qa.graph.state import AgentState
from knowledge_qa.graph.dependencies import GraphDependencies, get_dependencies


# ============================================================
# 辅助函数
# ============================================================

def _build_context(chunks: list[dict]) -> str:
    """构建 RAG 上下文"""
    if not chunks:
        return "没有找到相关的上下文信息。"
    parts = ["【参考上下文】\n"]
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] {chunk.get('source', '未知来源')}:\n{chunk.get('content', '')}\n")
    return "\n".join(parts)


def _build_messages(
    query: str,
    context: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> list[dict]:
    """构建对话消息列表"""
    messages = []
    if conversation_history:
        for msg in conversation_history:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": f"{context}\n\n【当前问题】\n{query}"})
    return messages


# ============================================================
# 节点实现
# ============================================================

async def retrieve_node(state: AgentState, deps: GraphDependencies) -> AgentState:
    """
    节点 1：知识库检索
    
    Args:
        state: Agent 状态
        deps: 注入的依赖（包含 retriever）
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        retriever = deps.retriever
        if retriever is None:
            logger.error("Retriever not configured in dependencies")
            return {**state, "retrieved_chunks": [], "error": "Retriever not configured"}
        
        result = await retriever.retrieve(
            query=state["query"],
            knowledge_base_id=state.get("knowledge_base_id", 1),
            top_k=state.get("top_k", deps.default_top_k),
        )
        
        return {
            **state,
            "retrieved_chunks": [
                {
                    "content": c.content,
                    "source": c.document_name,
                    "relevance": c.score,
                }
                for c in result.chunks
            ],
        }
    except Exception as e:
        logger.error(f"retrieve_node failed: {e}")
        return {**state, "retrieved_chunks": [], "error": str(e)}


async def generate_node(state: AgentState, deps: GraphDependencies) -> AgentState:
    """
    节点 2：LLM 生成
    
    Args:
        state: Agent 状态
        deps: 注入的依赖（包含 llm 和 system_prompt）
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        llm = deps.llm
        if llm is None:
            logger.error("LLM not configured in dependencies")
            return {**state, "error": "LLM not configured"}
        
        context = _build_context(state.get("retrieved_chunks", []))
        messages = _build_messages(
            query=state["query"],
            context=context,
            conversation_history=state.get("conversation_history"),
        )
        
        response = await llm.generate(
            prompt=messages,
            system_prompt=deps.system_prompt,
            temperature=state.get("temperature", deps.default_temperature),
            max_tokens=deps.max_tokens,
        )
        
        return {
            **state,
            "current_thought": response.content,
            "final_answer": response.content,
        }
    except Exception as e:
        logger.error(f"generate_node failed: {e}")
        return {**state, "error": str(e)}


async def decide_node(state: AgentState, deps: GraphDependencies) -> AgentState:
    """
    节点 3：置信度决策
    
    基于检索结果和置信度决定是否需要澄清。
    
    Args:
        state: Agent 状态
        deps: 注入的依赖
    """
    chunks = state.get("retrieved_chunks", [])
    
    if not chunks:
        return {
            **state,
            "confidence": 0.3,
            "needs_clarification": True,
            "clarification_question": "抱歉，知识库中没有找到相关信息，能否换个问法？",
        }
    
    top_relevance = chunks[0].get("relevance", 0)
    if top_relevance < 0.5:
        return {
            **state,
            "confidence": 0.5,
            "needs_clarification": True,
            "clarification_question": "我对这个问题的答案不太确定，您能补充一些信息吗？",
        }
    
    return {
        **state,
        "confidence": 0.85,
        "needs_clarification": False,
    }


# ============================================================
# 带条件边的节点（用于更复杂的流程控制）
# ============================================================

async def analyze_node(state: AgentState, deps: GraphDependencies) -> AgentState:
    """
    分析节点：判断是否需要检索
    
    如果 query 很明确可以直接回答，就跳过检索。
    """
    query = state.get("query", "")
    
    # 简单的启发式判断：短 query 通常需要检索
    needs_retrieval = len(query) < 50 or "什么" in query or "如何" in query or "为什么" in query
    
    return {
        **state,
        "_needs_retrieval": needs_retrieval,
    }


# ============================================================
# 包装函数（用于 LangGraph add_node）
# ============================================================

def create_retrieve_node(deps: GraphDependencies):
    """创建带依赖的 retrieve_node"""
    async def node(state: AgentState) -> AgentState:
        return await retrieve_node(state, deps)
    return node


def create_generate_node(deps: GraphDependencies):
    """创建带依赖的 generate_node"""
    async def node(state: AgentState) -> AgentState:
        return await generate_node(state, deps)
    return node


def create_decide_node(deps: GraphDependencies):
    """创建带依赖的 decide_node"""
    async def node(state: AgentState) -> AgentState:
        return await decide_node(state, deps)
    return node
