"""
Agent 基础定义
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class AgentStatus(str, Enum):
    """Agent 状态"""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"       # 等待用户输入
    ERROR = "error"
    DONE = "done"


@dataclass
class AgentConfig:
    """Agent 配置"""
    # LLM 配置
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2000
    
    # 推理配置
    max_iterations: int = 10
    max_execution_steps: int = 5
    timeout: int = 60  # 秒
    
    # 置信度配置
    confidence_threshold: float = 0.7
    low_confidence_ask_clarify: bool = True
    
    # 工具配置
    enable_tools: bool = True
    enabled_tools: list[str] = field(default_factory=list)  # 空表示全部启用
    
    # 记忆配置
    max_short_term_messages: int = 20
    long_term_memory_enabled: bool = True


@dataclass
class AgentThought:
    """Agent 思考步骤"""
    step: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[dict] = None
    observation: Optional[str] = None
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ToolUse:
    """工具调用记录"""
    tool_name: str
    arguments: dict
    result: Any
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class AgentResponse:
    """Agent 响应"""
    content: str
    status: AgentStatus
    thoughts: list[AgentThought] = field(default_factory=list)
    tool_uses: list[ToolUse] = field(default_factory=list)
    confidence: float = 1.0
    citations: list[dict] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None


class BaseAgent(ABC):
    """Agent 基类"""
    
    @abstractmethod
    async def run(self, query: str, **kwargs) -> AgentResponse:
        """运行 Agent"""
        pass
    
    @abstractmethod
    async def plan(self, query: str) -> list["Task"]:
        """规划任务"""
        pass
    
    @abstractmethod
    async def execute(self, task: "Task") -> Any:
        """执行任务"""
        pass
    
    def get_status(self) -> AgentStatus:
        """获取状态"""
        return self._status
    
    def reset(self):
        """重置 Agent"""
        self._status = AgentStatus.IDLE
