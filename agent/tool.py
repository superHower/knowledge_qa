"""
工具系统
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional
import json
import asyncio
import inspect


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ToolDefinition:
    """工具定义（用于 LLM 调用）"""
    name: str
    description: str
    parameters: dict  # JSON Schema 格式
    
    def to_openai_format(self) -> dict:
        """转换为 OpenAI function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class BaseTool(ABC):
    """工具基类"""
    
    # 工具名称（子类必须定义）
    name: str = ""
    description: str = ""
    
    def __init__(self):
        self._call_count = 0
        self._total_time = 0.0
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """执行工具"""
        pass
    
    def get_definition(self) -> ToolDefinition:
        """获取工具定义"""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self._get_parameters_schema(),
        )
    
    def _get_parameters_schema(self) -> dict:
        """获取参数 Schema（子类可重写）"""
        # 自动从 execute 方法签名提取
        sig = inspect.signature(self.execute)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            
            param_type = "string"
            if param.annotation == int:
                param_type = "integer"
            elif param.annotation == float:
                param_type = "number"
            elif param.annotation == bool:
                param_type = "boolean"
            elif param.annotation == list:
                param_type = "array"
            elif param.annotation == dict:
                param_type = "object"
            
            properties[param_name] = {"type": param_type}
            
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }
    
    def reset_stats(self):
        """重置统计"""
        self._call_count = 0
        self._total_time = 0.0


class KnowledgeBaseTool(BaseTool):
    """知识库检索工具"""
    
    name = "knowledge_base_search"
    description = "在企业知识库中搜索相关信息。当你需要查询公司制度、流程、政策、产品信息时使用。输入是搜索查询，返回相关的知识库内容。"
    
    def __init__(
        self,
        retriever,  # AdvancedRAGRetriever
        knowledge_base_id: int,
    ):
        super().__init__()
        self.retriever = retriever
        self.knowledge_base_id = knowledge_base_id
    
    async def execute(
        self,
        query: str,
        top_k: int = 5,
    ) -> ToolResult:
        """搜索知识库"""
        start_time = datetime.utcnow()
        
        try:
            result = await self.retriever.retrieve(
                query=query,
                knowledge_base_id=self.knowledge_base_id,
                top_k=top_k,
            )
            
            # 格式化结果
            formatted = []
            for chunk in result.chunks:
                formatted.append({
                    "content": chunk.content,
                    "source": chunk.document_name,
                    "relevance": round(chunk.score, 2),
                })
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ToolResult(
                success=True,
                result={
                    "query": query,
                    "chunks": formatted,
                    "total_found": len(formatted),
                    "rewritten_queries": result.rewritten_queries,
                    "metrics": result.metrics,
                },
                execution_time=execution_time,
                metadata={"knowledge_base_id": self.knowledge_base_id},
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
            )


class CalculatorTool(BaseTool):
    """计算器工具"""
    
    name = "calculator"
    description = "执行数学计算。当你需要计算费用、金额、百分比、日期差等数值计算时使用。"
    
    async def execute(
        self,
        expression: str,
    ) -> ToolResult:
        """执行计算"""
        import math
        import re
        
        start_time = datetime.utcnow()
        
        try:
            # 安全计算（限制可用函数）
            allowed_globals = {
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "len": len, "pow": pow, "sqrt": math.sqrt,
            }
            
            # 只允许数字和基本运算符
            if not re.match(r'^[\d\s+\-*/().,]+$', expression):
                # 尝试更宽松但仍安全的表达式
                allowed_names = {"abs": abs, "round": round, "min": min, "max": max,
                               "sum": sum, "len": len, "pow": pow, "sqrt": math.sqrt,
                               "pi": 3.14159, "e": 2.71828}
                result = eval(expression, {"__builtins__": {}}, allowed_names)
            else:
                result = eval(expression, {"__builtins__": {}}, {})
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ToolResult(
                success=True,
                result={
                    "expression": expression,
                    "result": result,
                },
                execution_time=execution_time,
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return ToolResult(
                success=False,
                error=f"计算错误: {str(e)}",
                execution_time=execution_time,
            )


class DateTimeTool(BaseTool):
    """日期时间工具"""
    
    name = "datetime_query"
    description = "查询当前日期时间或计算日期差。当需要知道今天是哪天、某个日期距今多少天等时使用。"
    
    async def execute(
        self,
        operation: str,  # "now", "diff"
        date1: Optional[str] = None,  # YYYY-MM-DD
        date2: Optional[str] = None,
    ) -> ToolResult:
        """查询日期时间"""
        from datetime import datetime as dt
        
        start_time = datetime.utcnow()
        
        try:
            if operation == "now":
                now = dt.now()
                return ToolResult(
                    success=True,
                    result={
                        "date": now.strftime("%Y-%m-%d"),
                        "time": now.strftime("%H:%M:%S"),
                        "weekday": now.strftime("%A"),
                        "timestamp": now.timestamp(),
                    },
                    execution_time=(dt.utcnow() - start_time).total_seconds(),
                )
                
            elif operation == "diff" and date1 and date2:
                d1 = dt.strptime(date1, "%Y-%m-%d")
                d2 = dt.strptime(date2, "%Y-%m-%d")
                diff = abs((d2 - d1).days)
                
                return ToolResult(
                    success=True,
                    result={
                        "date1": date1,
                        "date2": date2,
                        "days_diff": diff,
                        "years_diff": round(diff / 365, 1),
                    },
                    execution_time=(dt.utcnow() - start_time).total_seconds(),
                )
            else:
                return ToolResult(
                    success=False,
                    error="不支持的操作或缺少参数",
                    execution_time=(dt.utcnow() - start_time).total_seconds(),
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
            )


class WebSearchTool(BaseTool):
    """网页搜索工具"""
    
    name = "web_search"
    description = "搜索互联网获取最新信息。当用户询问实时信息、最新政策、新闻等知识库中没有的内容时使用。"
    
    def __init__(self, search_func: Callable):
        super().__init__()
        self.search_func = search_func
    
    async def execute(
        self,
        query: str,
        num_results: int = 5,
    ) -> ToolResult:
        """搜索网页"""
        start_time = datetime.utcnow()
        
        try:
            results = await self.search_func(query, num_results)
            
            return ToolResult(
                success=True,
                result={
                    "query": query,
                    "results": results,
                    "count": len(results),
                },
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
            )


class DatabaseTool(BaseTool):
    """数据库查询工具"""
    
    name = "database_query"
    description = "查询用户相关的业务数据，如保单信息、理赔记录、账户余额等。"
    
    def __init__(self, db_session):
        super().__init__()
        self.db = db_session
    
    async def execute(
        self,
        table: str,
        conditions: dict,
        limit: int = 10,
    ) -> ToolResult:
        """查询数据库"""
        start_time = datetime.utcnow()
        
        try:
            from sqlalchemy import select, and_
            
            # 动态构建查询（简化版，实际需要更严格的参数化）
            model_map = {
                "policies": Policy,
                "claims": Claim,
                "users": User,
            }
            
            model = model_map.get(table)
            if not model:
                return ToolResult(
                    success=False,
                    error=f"未知表: {table}",
                    execution_time=(datetime.utcnow() - start_time).total_seconds(),
                )
            
            # 构建筛选条件
            filters = []
            for key, value in conditions.items():
                if hasattr(model, key):
                    filters.append(getattr(model, key) == value)
            
            query = select(model)
            if filters:
                query = query.where(and_(*filters))
            query = query.limit(limit)
            
            result = await self.db.execute(query)
            rows = result.scalars().all()
            
            return ToolResult(
                success=True,
                result={
                    "table": table,
                    "count": len(rows),
                    "data": [r.__dict__ for r in rows],
                },
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
            )


class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool):
        """注册工具"""
        self._tools[tool.name] = tool
    
    def unregister(self, name: str):
        """注销工具"""
        self._tools.pop(name, None)
    
    def get(self, name: str) -> Optional[BaseTool]:
        """获取工具"""
        return self._tools.get(name)
    
    def get_all(self) -> list[BaseTool]:
        """获取所有工具"""
        return list(self._tools.values())
    
    def get_definitions(self) -> list[ToolDefinition]:
        """获取所有工具定义（用于 LLM）"""
        return [tool.get_definition() for tool in self._tools.values()]
    
    def get_names(self) -> list[str]:
        """获取所有工具名称"""
        return list(self._tools.keys())
    
    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """执行工具"""
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"工具不存在: {tool_name}",
            )
        
        return await tool.execute(**kwargs)
    
    def create_knowledge_base_tool(
        self,
        retriever,
        knowledge_base_id: int,
    ) -> KnowledgeBaseTool:
        """创建知识库工具"""
        return KnowledgeBaseTool(retriever, knowledge_base_id)
