"""
LangGraph 节点实现

节点流程：
- 基础流程：retrieve → generate → decide → END
- 澄清流程：decide (置信度低) → rewrite → retrieve → generate → decide
- 人工介入：decide (无结果) → clarify → END (等待用户输入)

依赖通过 GraphDependencies 注入，避免在节点内部创建实例。
"""

from typing import TypedDict, Literal, Optional, List, Dict, Any, Callable

from knowledge_qa.graph.state import AgentState
from knowledge_qa.graph.dependencies import GraphDependencies


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
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        retriever = deps.retriever
        if retriever is None:
            logger.error("Retriever not configured in dependencies")
            return {**state, "retrieved_chunks": [], "error": "Retriever not configured"}
        
        # 获取来源去重
        previous_sources = set(state.get("retrieved_sources", []))
        
        result = await retriever.retrieve(
            query=state["query"],
            knowledge_base_id=state.get("knowledge_base_id", 1),
            top_k=state.get("top_k", deps.default_top_k),
        )
        
        # 过滤已检索过的来源
        new_chunks = []
        new_sources = []
        for c in result.chunks:
            if c.document_name not in previous_sources:
                new_chunks.append({
                    "content": c.content,
                    "source": c.document_name,
                    "relevance": c.score,
                })
                new_sources.append(c.document_name)
        
        # 合并已有和新检索的 chunks
        all_chunks = state.get("retrieved_chunks", []) + new_chunks
        
        return {
            **state,
            "retrieved_chunks": all_chunks,
            "retrieved_sources": list(previous_sources) + new_sources,
            "current_step": "generate",
        }
    except Exception as e:
        logger.error(f"retrieve_node failed: {e}")
        return {**state, "retrieved_chunks": [], "error": str(e)}


async def generate_node(state: AgentState, deps: GraphDependencies) -> AgentState:
    """
    节点 2：LLM 生成
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
            "current_step": "decide",
        }
    except Exception as e:
        logger.error(f"generate_node failed: {e}")
        return {**state, "error": str(e)}


async def decide_node(state: AgentState, deps: GraphDependencies) -> AgentState:
    """
    节点 3：置信度决策
    
    基于检索结果和置信度决定下一步：
    - 需要澄清但可重试 → rewrite (改写查询重试)
    - 无检索结果且不可重试 → clarify (等待用户输入)
    - 置信度正常 → END
    """
    chunks = state.get("retrieved_chunks", [])
    retrieval_round = state.get("retrieval_round", 0)
    max_retries = 2  # 最多重试 2 次
    
    # 情况 1：没有任何检索结果
    if not chunks:
        if retrieval_round < max_retries:
            # 还有重试机会，改写查询重试
            return {
                **state,
                "confidence": 0.3,
                "needs_clarification": True,
                "clarification_type": "retry",
                "current_step": "rewrite",
            }
        else:
            # 重试次数用完，等待用户输入
            return {
                **state,
                "confidence": 0.2,
                "needs_clarification": True,
                "clarification_question": "抱歉，知识库中没有找到相关信息，能否换个问法或提供更多关键词？",
                "clarification_type": "user_input",
                "current_step": "clarify",
            }
    
    # 情况 2：有检索结果但置信度低
    top_relevance = chunks[0].get("relevance", 0) if chunks else 0
    if top_relevance < 0.5 and retrieval_round < max_retries:
        # 置信度低但还有重试机会
        return {
            **state,
            "confidence": 0.5,
            "needs_clarification": True,
            "clarification_type": "retry",
            "current_step": "rewrite",
        }
    elif top_relevance < 0.3:
        # 置信度非常低，直接要求用户输入
        return {
            **state,
            "confidence": 0.3,
            "needs_clarification": True,
            "clarification_question": "我对这个问题的答案不太确定，您能补充一些信息吗？",
            "clarification_type": "user_input",
            "current_step": "clarify",
        }
    
    # 情况 3：置信度正常，流程结束
    return {
        **state,
        "confidence": 0.85,
        "needs_clarification": False,
        "current_step": "END",
    }


async def rewrite_node(state: AgentState, deps: GraphDependencies) -> AgentState:
    """
    节点 4：查询改写
    
    当置信度低时，使用 LLM 改写查询，尝试用不同的表达方式检索。
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        llm = deps.llm
        if llm is None:
            logger.error("LLM not configured")
            return {**state, "current_step": "clarify"}
        
        current_query = state["query"]
        retrieval_round = state.get("retrieval_round", 0)
        
        # 构建改写 prompt
        rewrite_prompt = f"""请将以下用户问题改写成不同的表达方式，保留核心意图。

原始问题：{current_query}

要求：
1. 只输出改写后的问题，不要其他解释
2. 改写 1-2 个不同角度的表达
3. 使用不同的关键词组合

改写问题："""
        
        response = await llm.generate(
            prompt=[{"role": "user", "content": rewrite_prompt}],
            system_prompt="你是一个专业的查询改写助手。",
            temperature=0.8,
            max_tokens=200,
        )
        
        # 提取改写后的查询（取第一行）
        rewritten = response.content.strip().split("\n")[0]
        
        logger.info(f"Query rewritten: '{current_query}' -> '{rewritten}'")
        
        return {
            **state,
            "query": rewritten,
            "retrieval_round": retrieval_round + 1,
            "current_step": "retrieve",
        }
    except Exception as e:
        logger.error(f"rewrite_node failed: {e}")
        # 改写失败，直接进入澄清节点
        return {**state, "current_step": "clarify"}


async def clarify_node(state: AgentState, deps: GraphDependencies) -> AgentState:
    """
    节点 5：澄清节点
    
    标记需要用户输入或直接提供友好回复。
    在流式 API 中会触发前端显示澄清问题。
    """
    clarification_type = state.get("clarification_type", "user_input")
    
    # 如果已经有澄清问题，保持不变
    if state.get("clarification_question"):
        return {**state, "current_step": "END"}
    
    # 根据类型生成默认澄清问题
    default_questions = {
        "retry": "让我尝试从另一个角度检索相关信息...",
        "user_input": "抱歉，知识库中没有找到相关信息，能否换个问法？",
        "fallback": "这个问题我暂时无法回答，请联系管理员补充知识库内容。",
    }
    
    return {
        **state,
        "clarification_question": default_questions.get(clarification_type, default_questions["user_input"]),
        "current_step": "END",
    }


# ============================================================
# 条件边路由函数
# ============================================================

def route_after_decide(state: AgentState) -> Literal["rewrite", "clarify", "__end__"]:
    """
    decide 节点之后的条件路由
    
    Returns:
        "rewrite": 需要改写查询重试
        "clarify": 需要等待用户输入
        "__end__": 流程结束
    """
    current_step = state.get("current_step", "")
    
    if current_step == "rewrite":
        return "rewrite"
    elif current_step == "clarify":
        return "clarify"
    else:
        return "__end__"


def route_after_rewrite(state: AgentState) -> Literal["retrieve", "clarify"]:
    """
    rewrite 节点之后的条件路由
    
    Returns:
        "retrieve": 继续检索
        "clarify": 进入澄清（改写失败）
    """
    if state.get("error"):
        return "clarify"
    return "retrieve"


# ============================================================
# 包装函数（用于 LangGraph add_node）
# ============================================================

def create_retrieve_node(deps: GraphDependencies):
    async def node(state: AgentState) -> AgentState:
        return await retrieve_node(state, deps)
    return node


def create_generate_node(deps: GraphDependencies):
    async def node(state: AgentState) -> AgentState:
        return await generate_node(state, deps)
    return node


def create_decide_node(deps: GraphDependencies):
    async def node(state: AgentState) -> AgentState:
        return await decide_node(state, deps)
    return node


def create_rewrite_node(deps: GraphDependencies):
    async def node(state: AgentState) -> AgentState:
        return await rewrite_node(state, deps)
    return node


def create_clarify_node(deps: GraphDependencies):
    async def node(state: AgentState) -> AgentState:
        return await clarify_node(state, deps)
    return node
