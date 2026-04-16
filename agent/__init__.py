"""
Agent 核心架构

真正的 Agent 应该具备：
1. 工具调用 (Tool Use)
2. 规划能力 (Planning)
3. 记忆模块 (Memory)
4. 自主决策 (Self-Decision)
"""

from knowledge_qa.agent.base import (
    BaseAgent, AgentConfig, AgentResponse, AgentStatus, AgentThought, ToolUse
)
from knowledge_qa.agent.tool import (
    BaseTool, ToolResult, ToolDefinition, ToolRegistry,
    KnowledgeBaseTool, CalculatorTool, DateTimeTool, WebSearchTool, DatabaseTool
)
from knowledge_qa.agent.memory import (
    MemoryEntry, UserPreference,
    ShortTermMemory, LongTermMemory, WorkingMemory, EpisodicMemory, ReflectionMemory
)
from knowledge_qa.agent.planner import (
    Task, TaskStatus, TaskPriority, Plan, 
    BasePlanner, SimplePlanner, MultiStepPlanner, HierarchicalPlanner, DynamicPlanner
)
from knowledge_qa.agent.executor import (
    ExecutionStep, ExecutionResult, ReActExecutor, StreamingReActExecutor
)
from knowledge_qa.agent.decision import (
    Decision, ConfidenceResult, ClarificationResult,
    ConfidenceEvaluator, ClarificationGenerator, DecisionEngine, ErrorHandler
)
from knowledge_qa.agent.llm import (
    BaseLLM, LLMResponse, OpenAILLM, ClaudeLLM, LLMFactory
)
from knowledge_qa.agent.agent import (
    KnowledgeQAAgent, AgentFactory, ChatResult
)
from knowledge_qa.agent.prompts import PromptBuilder, PromptContext, RefinedPromptBuilder

__all__ = [
    # Base
    "BaseAgent",
    "AgentConfig",
    "AgentResponse",
    "AgentStatus",
    "AgentThought",
    "ToolUse",
    # Tool
    "BaseTool",
    "ToolResult",
    "ToolDefinition",
    "ToolRegistry",
    "KnowledgeBaseTool",
    "CalculatorTool",
    "DateTimeTool",
    "WebSearchTool",
    "DatabaseTool",
    # Memory
    "MemoryEntry",
    "UserPreference",
    "ShortTermMemory",
    "LongTermMemory",
    "WorkingMemory",
    "EpisodicMemory",
    "ReflectionMemory",
    # Planner
    "Task",
    "TaskStatus",
    "TaskPriority",
    "Plan",
    "BasePlanner",
    "SimplePlanner",
    "MultiStepPlanner",
    "HierarchicalPlanner",
    "DynamicPlanner",
    # Executor
    "ExecutionStep",
    "ExecutionResult",
    "ReActExecutor",
    "StreamingReActExecutor",
    # Decision
    "Decision",
    "ConfidenceResult",
    "ClarificationResult",
    "ConfidenceEvaluator",
    "ClarificationGenerator",
    "DecisionEngine",
    "ErrorHandler",
    # LLM
    "BaseLLM",
    "LLMResponse",
    "OpenAILLM",
    "ClaudeLLM",
    "LLMFactory",
    # Agent
    "KnowledgeQAAgent",
    "AgentFactory",
    "ChatResult",
    # Prompt
    "PromptBuilder",
    "PromptContext",
    "RefinedPromptBuilder",
]
