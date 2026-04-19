"""
LangGraph 节点实现

节点：retrieve → generate → decide
"""

from knowledge_qa.graph.state import AgentState


SYSTEM_PROMPT = """你是一个专业的企业知识库问答助手。

要求：
1. 基于提供的上下文信息回答问题
2. 如果上下文中没有相关信息，诚实地告知用户
3. 回答要准确、简洁、有条理
4. 如果涉及具体数据或政策，引用来源

回答格式：
- 首先给出直接回答
- 然后列出参考来源（如有）"""


async def retrieve_node(state: AgentState) -> AgentState:
    """节点 1：知识库检索"""
    try:
        from knowledge_qa.rag import get_vector_store, AdvancedRAGRetriever
        from knowledge_qa.rag.embedding import OpenAIEmbedding
        from knowledge_qa.core.config import settings

        embedding = OpenAIEmbedding(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model=settings.OPENAI_EMBEDDING_MODEL,
        )
        vector_store = get_vector_store()
        retriever = AdvancedRAGRetriever(embedding, vector_store)

        result = await retriever.retrieve(
            query=state["query"],
            knowledge_base_id=state.get("knowledge_base_id", 1),
            top_k=state.get("top_k", 5),
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
        import logging
        logging.getLogger(__name__).error(f"retrieve_node failed: {e}")
        return {**state, "retrieved_chunks": [], "error": str(e)}


async def generate_node(state: AgentState) -> AgentState:
    """节点 2：LLM 生成（非流式，用于 graph.ainvoke）"""
    from knowledge_qa.agent.llm import OpenAILLM
    from knowledge_qa.core.config import settings

    llm = OpenAILLM(
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_MODEL,
    )

    context = _build_context(state.get("retrieved_chunks", []))
    messages = _build_messages(
        query=state["query"],
        context=context,
        conversation_history=state.get("conversation_history"),
    )

    response = await llm.generate(
        prompt=messages,
        system_prompt=SYSTEM_PROMPT,
        temperature=state.get("temperature", 0.7),
        max_tokens=2000,
    )

    return {
        **state,
        "current_thought": response.content,
        "final_answer": response.content,
    }


async def decide_node(state: AgentState) -> AgentState:
    """节点 3：置信度决策"""
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


def _build_context(chunks: list[dict]) -> str:
    if not chunks:
        return "没有找到相关的上下文信息。"
    parts = ["【参考上下文】\n"]
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] {chunk.get('source', '未知来源')}:\n{chunk.get('content', '')}\n")
    return "\n".join(parts)


def _build_messages(
    query: str,
    context: str,
    conversation_history: list[dict] | None = None,
) -> list[dict]:
    messages = []
    if conversation_history:
        for msg in conversation_history:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": f"{context}\n\n【当前问题】\n{query}"})
    return messages
