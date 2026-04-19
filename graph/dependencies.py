"""
Graph 依赖注入容器

集中管理 LangGraph 节点的依赖项，避免在节点内部重复创建实例。
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
import functools

if TYPE_CHECKING:
    from knowledge_qa.agent.llm import BaseLLM
    from knowledge_qa.rag.retriever import AdvancedRAGRetriever
    from knowledge_qa.rag.embedding import EmbeddingModel


@dataclass
class GraphDependencies:
    """Graph 依赖容器"""
    
    # LLM 实例
    llm: Optional["BaseLLM"] = None
    
    # Embedding 实例
    embedding: Optional["EmbeddingModel"] = None
    
    # Retriever 实例
    retriever: Optional["AdvancedRAGRetriever"] = None
    
    # System prompt
    system_prompt: str = """你是一个专业的企业知识库问答助手。

要求：
1. 基于提供的上下文信息回答问题
2. 如果上下文中没有相关信息，诚实地告知用户
3. 回答要准确、简洁、有条理
4. 如果涉及具体数据或政策，引用来源

回答格式：
- 首先给出直接回答
- 然后列出参考来源（如有）"""
    
    # 默认 top_k
    default_top_k: int = 5
    
    # 默认 temperature
    default_temperature: float = 0.7
    
    # 最大 token 数
    max_tokens: int = 2000


# 全局依赖实例
_dependencies: Optional[GraphDependencies] = None


def get_dependencies() -> GraphDependencies:
    """获取全局依赖容器"""
    global _dependencies
    if _dependencies is None:
        _dependencies = GraphDependencies()
    return _dependencies


def set_dependencies(deps: GraphDependencies) -> None:
    """设置全局依赖容器"""
    global _dependencies
    _dependencies = deps


def with_dependencies(func):
    """
    装饰器：为节点函数自动注入依赖
    
    使用方式：
    @with_dependencies
    async def my_node(state: AgentState, deps: GraphDependencies) -> AgentState:
        retriever = deps.retriever
        ...
    """
    @functools.wraps(func)
    async def wrapper(state: dict, *args, **kwargs):
        deps = get_dependencies()
        return await func(state, deps, *args, **kwargs)
    return wrapper


class DependencyInjector:
    """
    依赖注入器
    
    用于在创建 graph 时绑定依赖。
    支持两种使用方式：
    1. 装饰器模式：@with_dependencies
    2. 函数绑定模式：inject(node_func, deps)
    """
    
    def __init__(self, dependencies: GraphDependencies):
        self.deps = dependencies
    
    def inject(self, node_func):
        """将依赖注入到节点函数"""
        @functools.wraps(node_func)
        async def wrapper(state: dict):
            return await node_func(state, self.deps)
        return wrapper
    
    def inject_all(self, node_dict: dict) -> dict:
        """批量注入依赖"""
        return {
            name: self.inject(func) 
            for name, func in node_dict.items()
        }
