"""
企业知识库问答 Agent

真正的 Agent 实现：
1. 工具调用 (Tool Use)
2. 规划能力 (Planning)
3. 记忆模块 (Memory)
4. 自主决策 (Self-Decision)
"""

from dataclasses import dataclass, asdict
from typing import Any, Optional, AsyncGenerator
from datetime import datetime
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from knowledge_qa.agent.base import (
    BaseAgent, AgentConfig, AgentResponse, AgentStatus, AgentThought, ToolUse
)
from knowledge_qa.agent.tool import ToolRegistry, ToolResult
from knowledge_qa.agent.memory import (
    ShortTermMemory, LongTermMemory, WorkingMemory, UserPreference
)
from knowledge_qa.agent.planner import BasePlanner, MultiStepPlanner, Plan
from knowledge_qa.agent.executor import ReActExecutor, StreamingReActExecutor, ExecutionResult
from knowledge_qa.agent.decision import (
    DecisionEngine, ConfidenceEvaluator, ClarificationGenerator, Decision
)
from knowledge_qa.agent.prompts import PromptBuilder, PromptContext
from knowledge_qa.db.models import ChatSession, ChatMessage, KnowledgeBase


class KnowledgeQAAgent(BaseAgent):
    """企业知识库问答 Agent
    
    完整的 Agent 实现，包含：
    - 工具系统 (Tool Use)
    - 规划器 (Planning)
    - 记忆系统 (Memory)
    - 决策引擎 (Self-Decision)
    - ReAct 执行循环
    """

    def __init__(
        self,
        llm,
        tool_registry: ToolRegistry,
        config: Optional[AgentConfig] = None,
        planner: Optional[BasePlanner] = None,
    ):
        self.llm = llm
        self.tool_registry = tool_registry
        self.config = config or AgentConfig()
        
        # 规划器
        self.planner = planner or MultiStepPlanner(llm)
        
        # 执行器
        self.executor = ReActExecutor(llm, tool_registry, self.config)
        self.streaming_executor = StreamingReActExecutor(llm, tool_registry, self.config)
        
        # 记忆系统
        self.short_term = ShortTermMemory(
            max_messages=self.config.max_short_term_messages
        )
        self.long_term = LongTermMemory()
        self.working_memory = WorkingMemory(self.short_term, self.long_term)
        
        # 决策引擎
        confidence_evaluator = ConfidenceEvaluator(
            high_threshold=0.8,
            medium_threshold=0.5,
            low_threshold=0.3,
        )
        clarification_generator = ClarificationGenerator()
        self.decision_engine = DecisionEngine(
            confidence_evaluator,
            clarification_generator,
        )
        
        # Prompt 构建器
        self.prompt_builder = PromptBuilder()
        
        # 状态
        self._status = AgentStatus.IDLE
        self._current_plan: Optional[Plan] = None
    
    async def run(self, query: str, **kwargs) -> AgentResponse:
        """运行 Agent"""
        self._status = AgentStatus.THINKING
        
        try:
            # 1. 记录用户消息
            self.working_memory.add_message("user", query)
            
            # 2. 获取用户 ID（如果有）
            user_id = kwargs.get("user_id")
            
            # 3. 创建执行计划
            self._status = AgentStatus.EXECUTING
            plan = await self.plan(query)
            self._current_plan = plan
            
            # 4. 执行 ReAct 循环
            result = await self.executor.execute(query)
            
            # 5. 置信度评估与决策
            decision, confidence_result, clarification = await self.decision_engine.decide(
                query=query,
                chunks=result.citations,
                answer=result.answer,
                tool_results=result.tool_uses,
            )
            
            # 6. 根据决策处理
            if decision == Decision.REFUSE:
                response = AgentResponse(
                    content=result.answer,
                    status=AgentStatus.DONE,
                    thoughts=self._build_thoughts(result),
                    tool_uses=self._build_tool_uses(result),
                    confidence=confidence_result.confidence,
                    error="refused",
                )
            elif decision == Decision.CLARIFY:
                response = AgentResponse(
                    content=result.answer,
                    status=AgentStatus.WAITING,
                    thoughts=self._build_thoughts(result),
                    tool_uses=self._build_tool_uses(result),
                    confidence=confidence_result.confidence,
                    needs_clarification=True,
                    clarification_question=clarification.question,
                )
            else:
                # 置信度评估通过
                response = AgentResponse(
                    content=result.answer,
                    status=AgentStatus.DONE,
                    thoughts=self._build_thoughts(result),
                    tool_uses=self._build_tool_uses(result),
                    confidence=confidence_result.confidence,
                    citations=result.citations,
                )
            
            # 7. 记录助手回复
            self.working_memory.add_message("assistant", response.content)
            
            # 8. 更新长期记忆
            if user_id:
                await self.long_term.update_interaction(user_id)
            
            return response
            
        except Exception as e:
            self._status = AgentStatus.ERROR
            return AgentResponse(
                content=f"发生错误: {str(e)}",
                status=AgentStatus.ERROR,
                error=str(e),
            )
    
    async def run_stream(self, query: str, **kwargs) -> AsyncGenerator[dict, None]:
        """流式运行 Agent"""
        self._status = AgentStatus.THINKING
        
        try:
            self.working_memory.add_message("user", query)
            user_id = kwargs.get("user_id")
            
            # 发送开始信号
            yield {"type": "start", "status": "thinking"}
            
            # 流式执行
            async for event in self.streaming_executor.execute_stream(query):
                if event["type"] == "step_start":
                    yield {
                        "type": "step",
                        "step": event["step"],
                        "status": "reasoning",
                    }
                elif event["type"] == "thought":
                    yield {
                        "type": "thought",
                        "content": event["content"],
                        "tool": event.get("action"),
                        "confidence": event.get("confidence", 1.0),
                    }
                elif event["type"] == "tool_result":
                    yield {
                        "type": "tool",
                        "tool": event["tool"],
                        "success": event["success"],
                        "observation": event["observation"],
                    }
                elif event["type"] == "final":
                    self._status = AgentStatus.DONE
                    self.working_memory.add_message("assistant", event["answer"])
                    
                    if user_id:
                        await self.long_term.update_interaction(user_id)
                    
                    yield {
                        "type": "final",
                        "answer": event["answer"],
                        "confidence": event.get("confidence", 1.0),
                        "status": "done",
                    }
                    break
            
            yield {"type": "complete"}
            
        except Exception as e:
            self._status = AgentStatus.ERROR
            yield {"type": "error", "message": str(e)}
    
    async def plan(self, query: str) -> Plan:
        """规划任务"""
        tools = self.tool_registry.get_all()
        return await self.planner.create_plan(query, tools)
    
    async def execute(self, task: Any) -> Any:
        """执行任务"""
        raise NotImplementedError("使用 run 方法执行")
    
    def reset(self):
        """重置 Agent"""
        self._status = AgentStatus.IDLE
        self.working_memory.clear()
        self._current_plan = None
    
    def _build_thoughts(self, result: ExecutionResult) -> list[AgentThought]:
        """构建思考记录"""
        thoughts = []
        for i, step in enumerate(result.steps, 1):
            thoughts.append(AgentThought(
                step=i,
                thought=step.thought,
                action=step.action,
                action_input=step.action_input,
                observation=step.observation,
                confidence=step.confidence,
            ))
        return thoughts
    
    def _build_tool_uses(self, result: ExecutionResult) -> list[ToolUse]:
        """构建工具使用记录"""
        return [
            ToolUse(
                tool_name=tu["tool"],
                arguments=tu["input"],
                result=tu["output"],
                success=tu.get("success", True),
            )
            for tu in result.tool_uses
            if tu.get("tool")
        ]

    async def chat(
        self,
        query: str,
        knowledge_base_id: int,
        session_id: Optional[int] = None,
        conversation_history: Optional[list[dict]] = None,
        top_k: int = 5,
        temperature: float = 0.7,
        stream: bool = False,
        db: Optional[AsyncSession] = None,
    ) -> "ChatResult":
        """RAG 对话接口（非流式）
        
        Args:
            query: 用户问题
            knowledge_base_id: 知识库ID
            session_id: 会话ID
            conversation_history: 对话历史
            top_k: 召回数量
            temperature: LLM 温度
            stream: 是否流式（暂不支持）
            db: 数据库会话
            
        Returns:
            ChatResult: 包含 answer, sources, citations, usage
        """
        from knowledge_qa.agent.tool import ToolResult
        
        # 1. 检索相关切片
        retrieved_chunks = []
        if self.tool_registry:
            kb_tool = None
            for tool in self.tool_registry.get_all():
                tool_name = tool.name.lower()
                # 支持: knowledge_base, knowledge_base_search, knowledgebasesearch
                if "knowledge" in tool_name and ("base" in tool_name or "search" in tool_name):
                    kb_tool = tool
                    break
            
            if kb_tool:
                result = await self.tool_registry.execute(
                    kb_tool.name,
                    query=query,
                    top_k=top_k,
                )
                if result.success and result.result:
                    retrieved_chunks = result.result.get("chunks", [])
        
        # 2. 构建上下文
        context = self._build_rag_context(retrieved_chunks)
        
        # 3. 构建对话消息
        messages = self._build_messages(query, context, conversation_history)
        
        # 4. 调用 LLM
        system_prompt = """你是一个专业的企业知识库问答助手。

要求：
1. 基于提供的上下文信息回答问题
2. 如果上下文中没有相关信息，诚实地告知用户
3. 回答要准确、简洁、有条理
4. 如果涉及具体数据或政策，引用来源

回答格式：
- 首先给出直接回答
- 然后列出参考来源（如有）"""

        response = await self.llm.generate(
            prompt=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=2000,
        )
        
        # 5. 构建结果
        sources = [
            {
                "content": chunk.get("content", "")[:200],
                "source": chunk.get("source", ""),
                "relevance": chunk.get("relevance", 0),
            }
            for chunk in retrieved_chunks[:3]
        ]
        
        citations = [
            {
                "content": chunk.get("content", ""),
                "source": chunk.get("source", ""),
                "relevance": chunk.get("relevance", 0),
            }
            for chunk in retrieved_chunks
        ]
        
        return ChatResult(
            answer=response.content,
            sources=sources,
            citations=citations,
            usage={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
            retrieved_chunks=retrieved_chunks,
        )

    async def chat_stream(
        self,
        query: str,
        knowledge_base_id: int,
        session_id: Optional[int] = None,
        conversation_history: Optional[list[dict]] = None,
        top_k: int = 5,
        temperature: float = 0.7,
        db: Optional[AsyncSession] = None,
    ) -> AsyncGenerator[dict, None]:
        """RAG 对话接口（流式 SSE）
        
        Yields:
            dict: 事件字典，包含 type 和相关数据
        """
        from knowledge_qa.agent.tool import ToolResult
        
        def emit_status(status: str, message: str = "", progress: int = 0, detail: dict = None):
            """发送状态事件"""
            return {
                "type": "status",
                "status": status,
                "message": message,
                "progress": progress,
                **(detail or {}),
            }
        
        try:
            # ========== 阶段 1: 意图分析 ==========
            yield emit_status("thinking", "正在分析问题...", 0)
            
            # ========== 阶段 2: 查询改写/意图重写 ==========
            yield emit_status("rewriting", "正在优化查询...", 10)
            
            # ========== 阶段 3: 检索 ==========
            yield emit_status("retrieving", "正在检索相关文档...", 20)
            
            # 1. 检索相关切片
            retrieved_chunks = []
            if self.tool_registry:
                # 查找知识库工具（支持多种命名）
                kb_tool = None
                for tool in self.tool_registry.get_all():
                    tool_name = tool.name.lower()
                    if "knowledge" in tool_name and "base" in tool_name:
                        kb_tool = tool
                        break
                
                if kb_tool:
                    result = await self.tool_registry.execute(
                        kb_tool.name,
                        query=query,
                        top_k=top_k,
                    )
                    if result.success and result.result:
                        retrieved_chunks = result.result.get("chunks", [])
            
            # 发送检索结果（带统计）
            yield {
                "type": "retrieval",
                "chunks": retrieved_chunks,
                "count": len(retrieved_chunks),
            }
            
            # ========== 阶段 4: 重排序 ==========
            if retrieved_chunks:
                yield emit_status("reranking", "正在对结果重排序...", 40)
            
            # ========== 阶段 5: 上下文构建 ==========
            yield emit_status("context", "正在构建上下文...", 50)
            
            # 2. 构建上下文
            context = self._build_rag_context(retrieved_chunks)
            
            # 3. 构建对话消息
            messages = self._build_messages(query, context, conversation_history)
            
            # 4. 系统提示词
            system_prompt = """你是一个专业的企业知识库问答助手。

要求：
1. 基于提供的上下文信息回答问题
2. 如果上下文中没有相关信息，诚实地告知用户
3. 回答要准确、简洁、有条理
4. 如果涉及具体数据或政策，引用来源

回答格式：
- 首先给出直接回答
- 然后列出参考来源（如有）"""
            
            # ========== 阶段 6: LLM 生成 ==========
            yield emit_status("generating", "正在生成回答...", 60)
            
            # 5. 流式生成
            full_answer = ""
            async for delta in self.llm.stream_generate(
                prompt=messages,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=2000,
            ):
                full_answer += delta
                yield {
                    "type": "content",
                    "delta": delta,
                }
            
            # ========== 阶段 7: 置信度评估 ==========
            yield emit_status("evaluating", "正在评估回答质量...", 90)
            
            # ========== 阶段 8: 完成 ==========
            yield {
                "type": "done",
                "answer": full_answer,
                "sources": [
                    {
                        "content": chunk.get("content", "")[:200],
                        "source": chunk.get("source", ""),
                    }
                    for chunk in retrieved_chunks[:3]
                ],
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e),
            }

    def _build_rag_context(self, chunks: list[dict]) -> str:
        """构建 RAG 上下文"""
        if not chunks:
            return "没有找到相关的上下文信息。"
        
        context_parts = ["【参考上下文】\n"]
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "")
            source = chunk.get("source", "未知来源")
            context_parts.append(f"[{i}] {source}:\n{content}\n")
        
        return "\n".join(context_parts)

    def _build_messages(
        self,
        query: str,
        context: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> list[dict]:
        """构建对话消息列表"""
        messages = []
        
        # 添加对话历史
        if conversation_history:
            for msg in conversation_history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })
        
        # 添加当前上下文和问题
        messages.append({
            "role": "user",
            "content": f"{context}\n\n【当前问题】\n{query}",
        })
        
        return messages

    async def save_conversation(
        self,
        query: str,
        answer: str,
        session_id: Optional[int],
        knowledge_base_id: int,
        retrieved_chunks: list[dict],
        db: Optional[AsyncSession],
    ) -> tuple[ChatSession, ChatMessage]:
        """保存对话到数据库
        
        Returns:
            (session, message): 会话和消息对象
        """
        if not db:
            raise ValueError("需要数据库会话才能保存对话")
        
        # 1. 获取或创建会话
        if session_id:
            stmt = select(ChatSession).where(ChatSession.id == session_id)
            result = await db.execute(stmt)
            session = result.scalar_one_or_none()
        else:
            session = None
        
        if not session:
            session = ChatSession(
                knowledge_base_id=knowledge_base_id,
                session_name=f"会话_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            db.add(session)
            await db.flush()
        
        # 2. 保存用户消息
        user_msg = ChatMessage(
            session_id=session.id,
            role="user",
            content=query,
        )
        db.add(user_msg)
        
        # 3. 保存助手消息
        assistant_msg = ChatMessage(
            session_id=session.id,
            role="assistant",
            content=answer,
            citations=retrieved_chunks[:3] if retrieved_chunks else None,
        )
        db.add(assistant_msg)
        
        # 4. 更新会话统计
        session.message_count += 2
        await db.commit()
        await db.refresh(session)
        await db.refresh(assistant_msg)
        
        return session, assistant_msg


@dataclass
class ChatResult:
    """对话结果"""
    answer: str
    sources: list[dict]
    citations: list[dict]
    usage: dict
    retrieved_chunks: list[dict] = None
    
    def __post_init__(self):
        if self.retrieved_chunks is None:
            self.retrieved_chunks = []


class AgentFactory:
    """Agent 工厂
    
    方便创建配置好的 Agent 实例
    """

    @staticmethod
    def create_knowledge_qa_agent(
        llm,
        retriever,
        knowledge_base_id: int,
        config: Optional[AgentConfig] = None,
    ) -> KnowledgeQAAgent:
        """创建知识库问答 Agent"""
        # 创建工具注册表
        registry = ToolRegistry()
        
        # 注册知识库工具
        kb_tool = registry.create_knowledge_base_tool(retriever, knowledge_base_id)
        registry.register(kb_tool)
        
        # 注册辅助工具
        registry.register(CalculatorTool())
        registry.register(DateTimeTool())
        
        # 创建 Agent
        agent = KnowledgeQAAgent(
            llm=llm,
            tool_registry=registry,
            config=config,
        )
        
        return agent


# 导入辅助工具
from knowledge_qa.agent.tool import CalculatorTool, DateTimeTool
