"""
LangGraph StateGraph 编译

使用依赖注入模式，避免在节点内部创建依赖实例。
"""

from typing import Literal, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from knowledge_qa.graph.state import AgentState
from knowledge_qa.graph.dependencies import GraphDependencies, get_dependencies, DependencyInjector
from knowledge_qa.graph.nodes import retrieve_node, generate_node, decide_node


def create_graph(deps: Optional[GraphDependencies] = None) -> StateGraph:
    """
    创建 LangGraph workflow
    
    Args:
        deps: GraphDependencies 实例。如果为 None，使用全局依赖。
    
    Returns:
        编译好的 LangGraph 应用
    """
    # 使用提供的依赖或全局依赖
    if deps is None:
        deps = get_dependencies()
    
    # 创建带依赖的节点函数
    injector = DependencyInjector(deps)
    
    workflow = StateGraph(AgentState)
    
    # 添加节点（使用依赖注入）
    workflow.add_node("retrieve", injector.inject(retrieve_node))
    workflow.add_node("generate", injector.inject(generate_node))
    workflow.add_node("decide", injector.inject(decide_node))
    
    # 设置入口和结束点
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "decide")
    workflow.add_edge("decide", END)
    
    # 创建 checkpointer
    checkpointer = MemorySaver()
    
    return workflow.compile(checkpointer=checkpointer)


def create_graph_with_conditional() -> tuple[StateGraph, dict]:
    """
    创建带条件边的更复杂 workflow
    
    Returns:
        (workflow, mapping): workflow 实例和条件边映射
    """
    deps = get_dependencies()
    injector = DependencyInjector(deps)
    
    def should_clarify(state: AgentState) -> Literal["clarify", "end"]:
        """根据置信度决定是否需要澄清"""
        if state.get("needs_clarification"):
            return "clarify"
        return "end"
    
    workflow = StateGraph(AgentState)
    
    # 核心节点
    workflow.add_node("retrieve", injector.inject(retrieve_node))
    workflow.add_node("generate", injector.inject(generate_node))
    workflow.add_node("decide", injector.inject(decide_node))
    
    # 澄清节点（可以重新检索或要求用户提供更多信息）
    async def clarify_node(state: AgentState, deps: GraphDependencies) -> AgentState:
        """澄清节点：标记需要用户输入"""
        return {**state, "_awaiting_clarification": True}
    
    workflow.add_node("clarify", injector.inject(clarify_node))
    
    # 边
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "decide")
    
    # 条件边
    workflow.add_conditional_edges(
        "decide",
        should_clarify,
        {
            "clarify": "clarify",
            "end": END,
        }
    )
    
    workflow.add_edge("clarify", END)
    
    checkpointer = MemorySaver()
    compiled = workflow.compile(checkpointer=checkpointer)
    
    return compiled


# 默认 graph 实例（使用全局依赖）
graph = create_graph()
