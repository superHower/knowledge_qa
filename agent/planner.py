"""
规划器 - 任务拆解与规划

核心能力：
1. 理解用户意图
2. 拆解多步骤任务
3. 确定执行顺序
4. 处理异常回退
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskPriority(str, Enum):
    """任务优先级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Task:
    """任务"""
    id: str
    description: str
    subtasks: list["Task"] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    dependencies: list[str] = field(default_factory=list)  # 依赖的任务ID
    tool_name: Optional[str] = None  # 需要的工具
    tool_args: dict = field(default_factory=dict)  # 工具参数
    result: Any = None  # 执行结果
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "tool": self.tool_name,
            "result": str(self.result)[:200] if self.result else None,
            "error": self.error,
        }


@dataclass
class Plan:
    """执行计划"""
    query: str
    tasks: list[Task]
    original_query: str
    constraints: dict = field(default_factory=dict)  # 约束条件
    expected_outcome: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        for task in self.tasks:
            if task.id == task_id:
                return task
            for subtask in task.subtasks:
                if subtask.id == task_id:
                    return subtask
        return None
    
    def get_ready_tasks(self) -> list[Task]:
        """获取可执行的任务（依赖都已完成）"""
        ready = []
        for task in self.tasks:
            if task.status != TaskStatus.PENDING:
                continue
            # 检查依赖
            deps_completed = all(
                self.get_task(dep_id).status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
                if self.get_task(dep_id)
            )
            if deps_completed:
                ready.append(task)
        return ready
    
    def is_complete(self) -> bool:
        """计划是否完成"""
        return all(t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED, TaskStatus.FAILED)
                  for t in self.tasks)
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "task_count": len(self.tasks),
            "completed": sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED),
            "tasks": [t.to_dict() for t in self.tasks],
        }


class BasePlanner(ABC):
    """规划器基类"""
    
    @abstractmethod
    async def create_plan(self, query: str, available_tools: list[str]) -> Plan:
        """创建执行计划"""
        pass


class SimplePlanner(BasePlanner):
    """简单规划器（单步任务）"""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def create_plan(self, query: str, available_tools: list[str]) -> Plan:
        """创建单步计划"""
        # 简单情况：直接搜索知识库
        task = Task(
            id="main",
            description=f"回答用户问题: {query}",
            tool_name="knowledge_base_search",
            tool_args={"query": query},
        )
        
        return Plan(
            query=query,
            tasks=[task],
            original_query=query,
        )


class MultiStepPlanner(BasePlanner):
    """多步骤规划器
    
    使用 LLM 分析查询并拆解为多个步骤
    """

    SYSTEM_PROMPT = """你是一个任务规划专家。你的任务是将用户的复杂问题拆解为可执行的步骤。

## 分析原则
1. **理解意图**：理解用户真正想要什么
2. **任务拆解**：将复杂问题拆分为简单子任务
3. **工具匹配**：为每个子任务选择合适的工具
4. **依赖排序**：确定任务的执行顺序
5. **边界处理**：识别需要澄清的问题

## 可用工具
{tools}

## 输出格式
```json
{
  "summary": "问题概述（一句话）",
  "needs_clarification": false,
  "clarification_question": null,
  "constraints": {
    "format": "简洁/详细",
    "style": "正式/口语"
  },
  "tasks": [
    {
      "id": "step_1",
      "description": "任务描述",
      "tool": "工具名称",
      "args": {"参数": "值"},
      "priority": "high/normal/low",
      "depends_on": []
    }
  ]
}
```

## 注意事项
- 如果问题太模糊，需要澄清，不要编造信息
- 优先使用 knowledge_base_search 获取公司信息
- 涉及计算用 calculator
- 日期相关用 datetime_query"""

    def __init__(self, llm):
        self.llm = llm
    
    async def create_plan(self, query: str, available_tools: list[str]) -> Plan:
        """创建多步骤计划"""
        tools_desc = "\n".join([
            f"- {name}: {desc[:100]}" 
            for name, desc in [(t.name, t.description) for t in available_tools]
        ])
        
        response = await self.llm.generate(
            prompt=f"用户问题：{query}\n\n可用工具：\n{tools_desc}",
            system_prompt=self.SYSTEM_PROMPT.format(tools=tools_desc),
            temperature=0.3,
            max_tokens=1500,
        )
        
        try:
            import json
            plan_data = json.loads(response.content)
            
            tasks = []
            for task_data in plan_data.get("tasks", []):
                task = Task(
                    id=task_data["id"],
                    description=task_data["description"],
                    tool_name=task_data.get("tool"),
                    tool_args=task_data.get("args", {}),
                    priority=TaskPriority(task_data.get("priority", "normal")),
                    dependencies=task_data.get("depends_on", []),
                )
                tasks.append(task)
            
            return Plan(
                query=plan_data.get("summary", query),
                tasks=tasks,
                original_query=query,
                constraints=plan_data.get("constraints", {}),
                expected_outcome=plan_data.get("expected_outcome"),
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            # 解析失败，使用简单计划
            task = Task(
                id="main",
                description=f"回答用户问题: {query}",
                tool_name="knowledge_base_search",
                tool_args={"query": query},
            )
            return Plan(
                query=query,
                tasks=[task],
                original_query=query,
            )


class HierarchicalPlanner(MultiStepPlanner):
    """层级规划器
    
    支持多层级任务拆解：
    1. 顶层：理解用户意图
    2. 中间层：主要步骤
    3. 底层：具体操作
    """

    SYSTEM_PROMPT = """你是一个专业的任务规划专家。你的任务是将复杂的企业问答拆解为层次化的执行计划。

## 规划框架

### 第一层：意图理解
- 用户想要完成什么？
- 涉及哪些业务领域？
- 需要什么约束条件？

### 第二层：主要步骤
每个主要步骤应该是一个完整的子任务

### 第三层：具体操作
每个步骤拆解为具体的工具调用

## 可用工具
{tools}

## 输出要求
返回 JSON 格式的层次化计划：

```json
{
  "intent": "用户意图简述",
  "needs_clarification": false,
  "clarification_question": null,
  "main_steps": [
    {
      "id": "step_1",
      "title": "步骤标题",
      "description": "步骤详细描述",
      "subtasks": [
        {
          "id": "step_1_1",
          "tool": "工具名",
          "args": {},
          "expected_result": "期望结果"
        }
      ]
    }
  ],
  "final_answer_template": "最终回答的格式模板"
}
```

## 注意事项
- 步骤之间有依赖关系时，需要标注
- 涉及多系统数据时，需要分别查询后合并
- 涉及计算时，使用 calculator 工具"""

    async def create_plan(self, query: str, available_tools: list[str]) -> Plan:
        """创建层次化计划"""
        return await super().create_plan(query, available_tools)


class DynamicPlanner(MultiStepPlanner):
    """动态规划器
    
    边执行边规划，
    根据中间结果调整后续计划
    """

    def __init__(self, llm):
        self.llm = llm
        self.base_planner = MultiStepPlanner(llm)
    
    async def create_initial_plan(self, query: str, available_tools: list[str]) -> Plan:
        """创建初始计划"""
        return await self.base_planner.create_plan(query, available_tools)
    
    async def revise_plan(
        self,
        current_plan: Plan,
        last_result: Any,
        last_task: Task,
    ) -> Plan:
        """根据执行结果修订计划"""
        # 检查是否需要添加新任务
        if last_result and isinstance(last_result, dict):
            # 根据结果判断是否需要后续处理
            pass
        
        return current_plan


class PlanOptimizer:
    """计划优化器
    
    优化执行计划：
    1. 并行化独立任务
    2. 合并相似任务
    3. 剪枝无效分支
    """

    def optimize(self, plan: Plan) -> Plan:
        """优化计划"""
        # 1. 并行化
        optimized_tasks = self._parallelize(plan.tasks)
        
        # 2. 合并相似任务
        merged_tasks = self._merge_similar(optimized_tasks)
        
        plan.tasks = merged_tasks
        return plan
    
    def _parallelize(self, tasks: list[Task]) -> list[Task]:
        """并行化独立任务"""
        # 简化实现：按依赖关系分组
        return tasks
    
    def _merge_similar(self, tasks: list[Task]) -> list[Task]:
        """合并相似任务"""
        return tasks
