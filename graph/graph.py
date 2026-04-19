"""
LangGraph StateGraph 编译

使用依赖注入模式和条件边实现完整的多轮检索流程。

流程图：
                                    ┌─────────────────────────────────────┐
                                    │                                     │
                                    ▼                                     │
┌──────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┴───┐
│START │───▶│ retrieve │───▶│ generate │───▶│  decide  │◀───┤             │
└──────┘    └──────────┘    └──────────┘    └────┬─────┘    │  clarify    │
                                                  │          │  (等待用户) │
                                                  │          └─────────────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    │                             │                             │
                    ▼                             ▼                             │
            ┌───────────┐               ┌────────────────┐                      │
            │  rewrite  │──────────────▶│    retrieve   │──────────────────────┘
            │ (改写查询) │               │   (重试检索)   │    max_retries 次
            └───────────┘               └────────────────┘
                    │                             ▲
                    │                             │
                    └─────────────────────────────┘
                            (改写失败，进入 clarify)
"""

from typing import Literal, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from knowledge_qa.graph.state import AgentState
from knowledge_qa.graph.dependencies import GraphDependencies, get_dependencies, DependencyInjector
from knowledge_qa.graph.nodes import (
    retrieve_node,
    generate_node,
    decide_node,
    rewrite_node,
    clarify_node,
    route_after_decide,
    route_after_rewrite,
)


def create_graph(deps: Optional[GraphDependencies] = None) -> StateGraph:
    """
    创建带条件边的 LangGraph workflow
    
    支持多轮检索：
    1. retrieve → generate → decide
    2. 如果置信度低 → rewrite → retrieve (重试)
    3. 重试失败 → clarify (等待用户输入)
    
    Args:
        deps: GraphDependencies 实例。如果为 None，使用全局依赖。
    
    Returns:
        编译好的 LangGraph 应用
    """
    if deps is None:
        deps = get_dependencies()
    
    injector = DependencyInjector(deps)
    
    workflow = StateGraph(AgentState)
    
    # 添加所有节点
    workflow.add_node("retrieve", injector.inject(retrieve_node))
    workflow.add_node("generate", injector.inject(generate_node))
    workflow.add_node("decide", injector.inject(decide_node))
    workflow.add_node("rewrite", injector.inject(rewrite_node))
    workflow.add_node("clarify", injector.inject(clarify_node))
    
    # 设置入口
    workflow.set_entry_point("retrieve")
    
    # 正常流程：retrieve → generate → decide
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "decide")
    
    # 条件边：decide 之后根据 current_step 决定下一步
    workflow.add_conditional_edges(
        "decide",
        route_after_decide,
        {
            "rewrite": "rewrite",    # 需要改写重试
            "clarify": "clarify",   # 需要等待用户输入
            "__end__": END,         # 流程结束
        }
    )
    
    # rewrite → retrieve (改写后重新检索)
    workflow.add_conditional_edges(
        "rewrite",
        route_after_rewrite,
        {
            "retrieve": "retrieve",  # 继续检索
            "clarify": "clarify",    # 改写失败
        }
    )
    
    # clarify 是终点
    workflow.add_edge("clarify", END)
    
    # 创建 checkpointer
    checkpointer = MemorySaver()
    
    return workflow.compile(checkpointer=checkpointer)


# 默认 graph 实例
graph = create_graph()


# ============================================================
# 辅助函数
# ============================================================

def get_graph_with_deps(
    llm=None,
    retriever=None,
    embedding=None,
    **kwargs
) -> StateGraph:
    """
    使用指定依赖创建 graph
    
    便捷函数，用于 API 层。
    """
    from knowledge_qa.graph.dependencies import set_dependencies
    
    deps = GraphDependencies(
        llm=llm,
        retriever=retriever,
        embedding=embedding,
        **kwargs
    )
    set_dependencies(deps)
    
    return create_graph(deps)
