"""
执行器 - ReAct 推理执行循环

ReAct = Reasoning + Acting
核心思想：思考 → 行动 → 观察 → 下一轮思考
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import json

from knowledge_qa.agent.base import AgentThought, AgentResponse, AgentStatus, AgentConfig
from knowledge_qa.agent.tool import ToolRegistry, ToolResult
from knowledge_qa.agent.memory import WorkingMemory, ShortTermMemory
from knowledge_qa.agent.planner import Plan, MultiStepPlanner


@dataclass
class ExecutionStep:
    """执行步骤"""
    step_number: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[dict] = None
    observation: Optional[str] = None
    tool_result: Optional[ToolResult] = None
    is_final: bool = False
    confidence: float = 1.0


@dataclass  
class ExecutionResult:
    """执行结果"""
    success: bool
    answer: str
    steps: list[ExecutionStep]
    confidence: float
    tool_uses: list[dict]
    citations: list[dict]
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    error: Optional[str] = None


class ReActExecutor:
    """ReAct 执行器
    
    实现 ReAct 推理循环：
    Thought → Action → Observation → ... → Final Answer
    """

    SYSTEM_PROMPT = """你是一个专业的企业问答助手，采用 ReAct (Reasoning + Acting) 推理模式。

## 你的能力
1. 使用工具获取信息（知识库搜索、计算、日期查询等）
2. 逐步推理，每一步都要有清晰的思考
3. 根据观察结果决定下一步行动

## 推理框架
每个步骤包含：
1. **Thought**: 你现在的思考过程
2. **Action**: 要执行的行动（如果有）
3. **Observation**: 行动的结果

## 可用工具
{tools}

## 决策规则
- 如果置信度 ≥ 0.7 → 生成最终答案
- 如果置信度 < 0.7 → 考虑追问用户
- 如果工具调用失败 → 尝试备选方案
- 如果无法回答 → 诚实告知

## 输出格式
每次输出 JSON：
```json
{
  "thought": "你的思考过程...",
  "action": "工具名 或 null（如果是最终答案）",
  "action_input": {"参数": "值"},
  "confidence": 0.0-1.0,
  "is_final": true/false
}
```

当 is_final=true 时：
```json
{
  "thought": "综合所有信息，给出最终回答...",
  "action": null,
  "confidence": 0.0-1.0,
  "is_final": true,
  "needs_clarification": false,
  "clarification_question": null
}
```"""

    def __init__(
        self,
        llm,
        tool_registry: ToolRegistry,
        config: Optional[AgentConfig] = None,
    ):
        self.llm = llm
        self.tool_registry = tool_registry
        self.config = config or AgentConfig()
        self._steps: list[ExecutionStep] = []
    
    async def execute(self, query: str) -> ExecutionResult:
        """执行 ReAct 循环"""
        self._steps.clear()
        step_number = 0
        
        # 获取工具定义
        tools_def = self.tool_registry.get_definitions()
        tools_json = json.dumps([t.__dict__ for t in tools_def], ensure_ascii=False)
        
        # 构建上下文
        context = self._build_context()
        
        while step_number < self.config.max_iterations:
            step_number += 1
            
            # LLM 推理
            response = await self._reason(query, context, tools_json)
            
            thought = response.get("thought", "")
            action = response.get("action")
            action_input = response.get("action_input", {})
            confidence = response.get("confidence", 1.0)
            is_final = response.get("is_final", False)
            needs_clarify = response.get("needs_clarification", False)
            clarify_q = response.get("clarification_question")
            
            # 创建执行步骤
            step = ExecutionStep(
                step_number=step_number,
                thought=thought,
                action=action,
                action_input=action_input,
                confidence=confidence,
                is_final=is_final,
            )
            
            # 如果需要最终回答
            if is_final:
                if needs_clarify:
                    return ExecutionResult(
                        success=True,
                        answer=thought,
                        steps=self._steps + [step],
                        confidence=confidence,
                        tool_uses=self._get_tool_uses(),
                        citations=self._get_citations(),
                        needs_clarification=True,
                        clarification_question=clarify_q,
                    )
                else:
                    return ExecutionResult(
                        success=True,
                        answer=thought,
                        steps=self._steps + [step],
                        confidence=confidence,
                        tool_uses=self._get_tool_uses(),
                        citations=self._get_citations(),
                    )
            
            # 执行工具
            if action and action in self.tool_registry.get_names():
                tool_result = await self.tool_registry.execute(action, **action_input)
                step.tool_result = tool_result
                step.observation = self._format_observation(tool_result)
                
                # 记录工具使用
                context += f"\n[Step {step_number}] Action: {action}\nObservation: {step.observation}"
            else:
                step.observation = "工具不存在或参数错误"
            
            self._steps.append(step)
            
            # 检查是否超过执行步骤上限
            if len([s for s in self._steps if s.tool_result]) >= self.config.max_execution_steps:
                break
        
        # 达到最大迭代次数，返回当前结果
        return ExecutionResult(
            success=False,
            answer=self._generate_fallback_answer(),
            steps=self._steps,
            confidence=0.3,
            tool_uses=self._get_tool_uses(),
            citations=self._get_citations(),
            error="达到最大执行步骤限制",
        )
    
    async def _reason(
        self,
        query: str,
        context: str,
        tools_json: str,
    ) -> dict:
        """LLM 推理"""
        prompt = f"""问题：{query}

上下文：
{context}

请按 ReAct 框架推理："""
        
        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT.format(tools=tools_json),
            temperature=0.7,
            max_tokens=800,
        )
        
        try:
            # 尝试解析 JSON
            result = json.loads(response.content.strip())
            return result
        except json.JSONDecodeError:
            # 非 JSON 格式，当作最终回答
            return {
                "thought": response.content,
                "action": None,
                "confidence": 0.8,
                "is_final": True,
            }
    
    def _build_context(self) -> str:
        """构建上下文"""
        return ""
    
    def _format_observation(self, result: ToolResult) -> str:
        """格式化工具观察结果"""
        if not result.success:
            return f"工具执行失败: {result.error}"
        
        if isinstance(result.result, dict):
            if "chunks" in result.result:
                # 知识库结果
                chunks = result.result["chunks"]
                obs_parts = []
                for i, chunk in enumerate(chunks[:3], 1):
                    content = chunk.get("content", "")[:200]
                    source = chunk.get("source", "未知")
                    relevance = chunk.get("relevance", 0)
                    obs_parts.append(f"[{i}] {source} (相关度:{relevance}): {content}...")
                return "\n".join(obs_parts)
            else:
                return str(result.result)[:500]
        
        return str(result.result)[:500]
    
    def _get_tool_uses(self) -> list[dict]:
        """获取工具使用记录"""
        return [
            {
                "tool": s.action,
                "input": s.action_input,
                "output": str(s.tool_result.result)[:200] if s.tool_result else None,
                "success": s.tool_result.success if s.tool_result else False,
            }
            for s in self._steps
            if s.action and s.tool_result
        ]
    
    def _get_citations(self) -> list[dict]:
        """获取引用"""
        citations = []
        for s in self._steps:
            if s.tool_result and s.tool_result.result:
                result = s.tool_result.result
                if isinstance(result, dict) and "chunks" in result:
                    for chunk in result.get("chunks", []):
                        citations.append({
                            "content": chunk.get("content", "")[:300],
                            "source": chunk.get("source", ""),
                            "relevance": chunk.get("relevance", 0),
                        })
        return citations
    
    def _generate_fallback_answer(self) -> str:
        """生成备用回答"""
        if self._steps:
            last_result = self._steps[-1].tool_result
            if last_result and last_result.success:
                return "根据我查询到的信息：\n" + str(last_result.result)[:1000]
        
        return "抱歉，我无法回答您的问题，建议您联系相关人员获取帮助。"


class StreamingReActExecutor(ReActExecutor):
    """流式 ReAct 执行器"""

    async def execute_stream(self, query: str):
        """流式执行"""
        self._steps.clear()
        step_number = 0
        
        tools_def = self.tool_registry.get_definitions()
        tools_json = json.dumps([t.__dict__ for t in tools_def], ensure_ascii=False)
        context = self._build_context()
        
        while step_number < self.config.max_iterations:
            step_number += 1
            
            yield {"type": "step_start", "step": step_number}
            
            # 推理
            response = await self._reason(query, context, tools_json)
            
            thought = response.get("thought", "")
            action = response.get("action")
            is_final = response.get("is_final", False)
            confidence = response.get("confidence", 1.0)
            
            yield {
                "type": "thought",
                "content": thought,
                "action": action,
                "confidence": confidence,
            }
            
            if is_final:
                yield {
                    "type": "final",
                    "answer": thought,
                    "confidence": confidence,
                }
                break
            
            # 执行工具
            if action and action in self.tool_registry.get_names():
                tool_result = await self.tool_registry.execute(
                    action, **response.get("action_input", {})
                )
                
                yield {
                    "type": "tool_result",
                    "tool": action,
                    "success": tool_result.success,
                    "observation": self._format_observation(tool_result),
                }
                
                context += f"\n[Step {step_number}] {action}: {tool_result}"
            
            yield {"type": "step_end", "step": step_number}
        
        yield {"type": "complete"}
