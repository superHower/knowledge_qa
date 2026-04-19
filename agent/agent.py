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
        """运行 Agent（使用 LangGraph graph.ainvoke）"""
        self._status = AgentStatus.THINKING

        try:
            self.working_memory.add_message("user", query)

            from knowledge_qa.graph.graph import graph

            self._status = AgentStatus.EXECUTING
            session_id = kwargs.get("session_id")
            config = {"configurable": {"thread_id": str(session_id) if session_id else "default"}}

            result = await graph.ainvoke(
                {
                    "query": query,
                    "knowledge_base_id": kwargs.get("knowledge_base_id", 1),
                    "session_id": session_id,
                    "conversation_history": kwargs.get("conversation_history"),
                    "top_k": kwargs.get("top_k", 5),
                    "temperature": kwargs.get("temperature", 0.7),
                },
                config=config,
            )

            self._status = AgentStatus.DONE
            answer = result.get("final_answer", "")
            self.working_memory.add_message("assistant", answer)

            if kwargs.get("user_id"):
                await self.long_term.update_interaction(kwargs["user_id"])

            return AgentResponse(
                content=answer,
                status=AgentStatus.WAITING if result.get("needs_clarification") else AgentStatus.DONE,
                confidence=result.get("confidence", 0.85),
                citations=result.get("retrieved_chunks", []),
                needs_clarification=result.get("needs_clarification", False),
                clarification_question=result.get("clarification_question"),
            )

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
        """
        RAG 对话接口（流式 SSE）
        
        支持多轮检索和条件边流程：
        - 检索 → 生成 → 决策
        - 置信度低 → 改写查询 → 重新检索（最多2轮）
        - 无结果 → 澄清节点
        
        Yields:
            dict: 事件字典，包含 type 和相关数据
        """
        from knowledge_qa.graph import get_dependencies
        from knowledge_qa.graph.nodes import (
            retrieve_node, generate_node, decide_node, 
            rewrite_node, clarify_node
        )
        
        deps = get_dependencies()
        
        def emit_status(status: str, message: str = "", progress: int = 0, detail: dict | None = None):
            """发送状态事件"""
            return {
                "type": "status",
                "status": status,
                "message": message,
                "progress": progress,
                **(detail or {}),
            }

        def build_node(
            node_id: str,
            title: str,
            kind: str,
            order: int,
            status: str,
            summary: str = "",
            detail: dict | None = None,
        ) -> dict:
            node = {
                "id": node_id,
                "title": title,
                "kind": kind,
                "order": order,
                "status": status,
            }
            if summary:
                node["summary"] = summary
            if detail:
                node.update(detail)
            return node
        
        try:
            yield {
                "type": "run.start",
                "query": query,
                "knowledge_base_id": knowledge_base_id,
            }

            # 初始化状态
            state = {
                "query": query,
                "knowledge_base_id": knowledge_base_id,
                "session_id": session_id,
                "conversation_history": conversation_history,
                "top_k": top_k,
                "temperature": temperature,
                "retrieved_chunks": [],
                "retrieval_round": 0,
                "retrieved_sources": [],
            }
            
            # ========== 节点 1: 检索 ==========
            yield emit_status("retrieving", "正在检索相关文档...", 10)
            yield {
                "type": "node.start",
                "node": build_node("retrieval", "知识库检索", "retrieval", 1, "running"),
            }
            
            state = await retrieve_node(state, deps)
            retrieved_chunks = state.get("retrieved_chunks", [])
            
            yield {
                "type": "retrieval",
                "chunks": retrieved_chunks,
                "count": len(retrieved_chunks),
                "round": state.get("retrieval_round", 0),
            }
            yield {
                "type": "node.done",
                "node": build_node(
                    "retrieval", "知识库检索", "retrieval", 1, "done",
                    summary=f"已召回 {len(retrieved_chunks)} 条相关片段",
                    detail={
                        "count": len(retrieved_chunks),
                        "sources": [chunk.get("source", "") for chunk in retrieved_chunks[:3] if chunk.get("source")],
                    },
                ),
            }

            # ========== 节点 2: 生成 ==========
            yield emit_status("generating", "正在生成回答...", 50)
            yield {
                "type": "node.start",
                "node": build_node("generate", "答案生成", "generation", 2, "running"),
            }
            
            state = await generate_node(state, deps)
            full_answer = state.get("final_answer", "")
            
            yield {
                "type": "node.done",
                "node": build_node(
                    "generate", "答案生成", "generation", 2, "done",
                    summary="回答生成完成",
                    detail={"answer_length": len(full_answer)},
                ),
            }
            
            # ========== 节点 3: 决策 ==========
            yield emit_status("deciding", "正在评估置信度...", 70)
            yield {
                "type": "node.start",
                "node": build_node("decide", "置信度决策", "decision", 3, "running"),
            }
            
            state = await decide_node(state, deps)
            
            # 处理决策结果
            current_step = state.get("current_step", "")
            confidence = state.get("confidence", 0.85)
            needs_clarification = state.get("needs_clarification", False)
            clarification_type = state.get("clarification_type", "")
            retrieval_round = state.get("retrieval_round", 0)
            
            yield {
                "type": "node.done",
                "node": build_node(
                    "decide", "置信度决策", "decision", 3, "done",
                    summary=f"决策完成，置信度: {confidence:.2f}",
                    detail={
                        "confidence": confidence,
                        "needs_clarification": needs_clarification,
                        "clarification_type": clarification_type,
                    },
                ),
            }
            
            # ========== 条件边处理 ==========
            
            # 需要改写查询重试
            if current_step == "rewrite":
                yield emit_status("rewriting", "正在改写查询...", 30)
                yield {
                    "type": "node.start",
                    "node": build_node("rewrite", "查询改写", "rewrite", 3, "running"),
                }
                
                state = await rewrite_node(state, deps)
                new_query = state.get("query", "")
                
                yield {
                    "type": "node.done",
                    "node": build_node(
                        "rewrite", "查询改写", "rewrite", 3, "done",
                        summary=f"查询已改写为: {new_query[:30]}...",
                        detail={"original_query": query, "rewritten_query": new_query},
                    ),
                }
                
                # 继续回到 retrieve（条件边）
                yield emit_status("retrieving", "正在重新检索...", 40)
                yield {
                    "type": "node.start",
                    "node": build_node("retrieval", "重新检索", "retrieval", 4, "running"),
                }
                
                state = await retrieve_node(state, deps)
                retrieved_chunks = state.get("retrieved_chunks", [])
                
                yield {
                    "type": "retrieval",
                    "chunks": retrieved_chunks,
                    "count": len(retrieved_chunks),
                    "round": state.get("retrieval_round", 0),
                }
                yield {
                    "type": "node.done",
                    "node": build_node(
                        "retrieval", "重新检索", "retrieval", 4, "done",
                        summary=f"重新召回 {len(retrieved_chunks)} 条相关片段",
                    ),
                }
                
                # 再次生成
                yield emit_status("generating", "正在重新生成回答...", 60)
                state = await generate_node(state, deps)
                full_answer = state.get("final_answer", "")
                
                # 再次决策
                yield emit_status("deciding", "正在评估置信度...", 80)
                state = await decide_node(state, deps)
                confidence = state.get("confidence", 0.85)
                current_step = state.get("current_step", "")
            
            # 需要澄清（等待用户输入）
            if current_step == "clarify":
                yield emit_status("clarifying", "需要更多信息...", 90)
                yield {
                    "type": "node.start",
                    "node": build_node("clarify", "澄清", "clarify", 5, "running"),
                }
                
                state = await clarify_node(state, deps)
                clarification_question = state.get("clarification_question", "请提供更多信息")
                
                yield {
                    "type": "node.done",
                    "node": build_node(
                        "clarify", "澄清", "clarify", 5, "done",
                        summary="等待用户输入",
                    ),
                }
                
                # 输出澄清事件
                yield {
                    "type": "clarify",
                    "question": clarification_question,
                    "clarification_type": clarification_type,
                }
                
                # 构建结果
                sources = [
                    {"content": chunk.get("content", "")[:200], "source": chunk.get("source", "")}
                    for chunk in retrieved_chunks[:3]
                ]
                citations = [
                    {"content": chunk.get("content", ""), "source": chunk.get("source", ""), "relevance": chunk.get("relevance", 0)}
                    for chunk in retrieved_chunks
                ]
                
                yield {
                    "type": "answer.done",
                    "answer": full_answer,
                    "sources": sources,
                    "citations": citations,
                    "confidence": confidence,
                    "retrieved_chunks": retrieved_chunks,
                    "needs_clarification": True,
                    "clarification_question": clarification_question,
                }
                return
            
            # 正常结束流程
            yield emit_status("finalizing", "正在整理结果...", 95)
            
            sources = [
                {"content": chunk.get("content", "")[:200], "source": chunk.get("source", "")}
                for chunk in retrieved_chunks[:3]
            ]
            citations = [
                {"content": chunk.get("content", ""), "source": chunk.get("source", ""), "relevance": chunk.get("relevance", 0)}
                for chunk in retrieved_chunks
            ]
            
            yield {
                "type": "answer.done",
                "answer": full_answer,
                "sources": sources,
                "citations": citations,
                "confidence": confidence,
                "retrieved_chunks": retrieved_chunks,
                "needs_clarification": False,
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
