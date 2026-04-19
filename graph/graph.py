"""
LangGraph StateGraph 编译
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from knowledge_qa.graph.state import AgentState
from knowledge_qa.graph.nodes import retrieve_node, generate_node, decide_node


def create_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("decide", decide_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "decide")
    workflow.add_edge("decide", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


graph = create_graph()
