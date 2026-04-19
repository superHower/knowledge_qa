"""
Graph 模块

使用依赖注入模式的 LangGraph 实现。
"""

from knowledge_qa.graph.state import AgentState
from knowledge_qa.graph.graph import graph, create_graph, create_graph_with_conditional
from knowledge_qa.graph.dependencies import (
    GraphDependencies,
    get_dependencies,
    set_dependencies,
    DependencyInjector,
)
from knowledge_qa.graph.nodes import retrieve_node, generate_node, decide_node

__all__ = [
    # State
    "AgentState",
    # Graph
    "graph",
    "create_graph",
    "create_graph_with_conditional",
    # Dependencies
    "GraphDependencies",
    "get_dependencies",
    "set_dependencies",
    "DependencyInjector",
    # Nodes
    "retrieve_node",
    "generate_node",
    "decide_node",
]
