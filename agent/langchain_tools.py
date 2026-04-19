"""
LangChain 格式工具定义（供 LangGraph ToolNode 使用）
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class KnowledgeBaseInput(BaseModel):
    query: str = Field(description="用户问题")
    top_k: int = Field(default=5, description="检索数量")
    knowledge_base_id: int = Field(default=1, description="知识库 ID")


@tool("knowledge_base_search", args_schema=KnowledgeBaseInput)
async def knowledge_base_search(query: str, top_k: int = 5, knowledge_base_id: int = 1) -> dict:
    """搜索知识库文档。适用于产品信息、政策、技术文档、FAQ 等问题。"""
    from knowledge_qa.rag import get_vector_store, AdvancedRAGRetriever
    from knowledge_qa.rag.embedding import OpenAIEmbedding
    from knowledge_qa.core.config import settings

    embedding = OpenAIEmbedding(api_key=settings.OPENAI_API_KEY)
    vector_store = get_vector_store()
    retriever = AdvancedRAGRetriever(embedding, vector_store)

    result = await retriever.retrieve(query=query, top_k=top_k, knowledge_base_id=knowledge_base_id)

    return {
        "chunks": [
            {"content": c.content, "source": c.document_name, "relevance": c.score}
            for c in result.chunks
        ],
        "total": result.total_chunks,
    }


@tool("calculator")
def calculator(expression: str) -> dict:
    """计算数学表达式。"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": str(e)}


tools = [knowledge_base_search, calculator]
