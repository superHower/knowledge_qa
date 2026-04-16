"""
查询改写与增强
"""

from abc import ABC, abstractmethod
from typing import Optional
import re


class QueryRewriter(ABC):
    """查询改写器基类"""
    
    @abstractmethod
    async def rewrite(self, query: str) -> list[str]:
        """改写查询，返回多个变体"""
        pass


class MultiQueryRewriter(QueryRewriter):
    """多查询改写
    
    将用户查询改写成多个不同角度的查询，
    解决用户表述模糊、关键词不匹配等问题
    """
    
    SYSTEM_PROMPT = """你是一个专业的查询改写专家。你的任务是将用户的原始问题改写成多个不同角度的查询。

## 要求
1. 生成 3-5 个不同的查询变体
2. 每个变体从不同角度表达同一问题
3. 包含同义词扩展（如 "续保" → "续保流程"、"如何续保"）
4. 包含口语化转正式（"那个..." → "..."）
5. 包含意图拆分（复杂问题拆成多个简单问题）

## 输出格式
一行一个查询，不要编号，不要解释。"""

    USER_PROMPT = """原始问题：{query}

请改写："""

    def __init__(self, llm):
        self.llm = llm
    
    async def rewrite(self, query: str) -> list[str]:
        """改写查询"""
        response = await self.llm.generate(
            prompt=self.USER_PROMPT.format(query=query),
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.8,
            max_tokens=500,
        )
        
        # 解析结果
        lines = response.content.strip().split("\n")
        queries = [q.strip() for q in lines if q.strip()]
        
        # 确保原始查询也在列表中
        if query not in queries:
            queries.insert(0, query)
        
        return queries[:5]  # 最多5个


class HyDERewriter(QueryRewriter):
    """HyDE (Hypothetical Document Embeddings)
    
    先让 LLM 根据查询生成一个"假设性答案"，
    然后用这个答案去检索，捕捉查询的语义意图
    """

    SYSTEM_PROMPT = """你是一个企业知识库助手。请根据用户问题，生成一个假设性的回答。

## 要求
1. 假设你是一个完全了解相关知识库的专家
2. 生成一个完整、详细、专业的回答
3. 回答要具体，包含可能的步骤、流程、数值等
4. 可以包含"根据公司规定..."、"按照流程..."等开头
5. 如果不确定，生成一个合理的假设性回答

## 注意
这不是最终答案，只是一个检索用的假设性文档。"""

    USER_PROMPT = """问题：{query}

请生成一个假设性的回答："""

    def __init__(self, llm):
        self.llm = llm
    
    async def rewrite(self, query: str) -> list[str]:
        """生成假设性文档"""
        response = await self.llm.generate(
            prompt=self.USER_PROMPT.format(query=query),
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=1000,
        )
        
        # 返回原始查询和假设性回答
        return [query, response.content]


class SubQueryRewriter(QueryRewriter):
    """子查询拆分
    
    将复杂查询拆分为多个简单子查询，
    分别检索后再合并结果
    """
    
    SYSTEM_PROMPT = """分析用户问题，判断是否需要拆分为多个子问题。

## 拆分原则
1. 问题包含多个实体 → 拆（如"张三和李四的入职时间"）
2. 问题包含多个步骤 → 拆（如"如何注册并登录"）
3. 问题包含对比 → 拆（如"A和B的区别"）
4. 单个简单问题 → 不拆

## 输出格式
如果需要拆分：
[子问题1]
[子问题2]
...

如果不需要拆分：
[NONE]
{原问题}"""

    def __init__(self, llm):
        self.llm = llm
    
    async def rewrite(self, query: str) -> list[str]:
        """拆分查询"""
        response = await self.llm.generate(
            prompt=f"问题：{query}\n\n请判断是否需要拆分：",
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=300,
        )
        
        content = response.content.strip()
        
        if content.startswith("[NONE]"):
            return [query]
        
        # 解析子问题
        lines = content.split("\n")
        sub_queries = [l.strip() for l in lines if l.strip() and not l.startswith("[")]
        
        if not sub_queries:
            return [query]
        
        return sub_queries


class QueryExpansionRewriter(QueryRewriter):
    """查询扩展
    
    添加同义词、上下位词、相关概念扩展查询
    """
    
    def __init__(self):
        # 常见同义词映射
        self.synonym_map = {
            "续保": ["续保流程", "续保手续", "续保办理", "保险续期"],
            "报销": ["费用报销", "报销流程", "如何报销"],
            "请假": ["请假申请", "请假流程", "请假制度"],
            "入职": ["入职流程", "入职手续", "新员工入职"],
            "离职": ["离职流程", "离职手续", "办理离职"],
            "加班": ["加班申请", "加班制度", "加班调休"],
            "采购": ["采购流程", "采购申请", "如何采购"],
            "报销": ["报销申请", "费用报销", "报销标准"],
        }
    
    async def rewrite(self, query: str) -> list[str]:
        """扩展查询"""
        queries = [query]
        
        # 查找同义词
        for key, synonyms in self.synonym_map.items():
            if key in query:
                for syn in synonyms:
                    expanded = query.replace(key, syn)
                    if expanded not in queries:
                        queries.append(expanded)
        
        return queries[:5]


class EnsembleQueryRewriter:
    """组合查询改写器
    
    结合多种改写策略
    """
    
    def __init__(
        self,
        llm,
        use_multi_query: bool = True,
        use_hyde: bool = True,
        use_subquery: bool = False,
        use_expansion: bool = True,
    ):
        self.rewriters = []
        
        if use_multi_query:
            self.rewriters.append(MultiQueryRewriter(llm))
        if use_hyde:
            self.rewriters.append(HyDERewriter(llm))
        if use_subquery:
            self.rewriters.append(SubQueryRewriter(llm))
        if use_expansion:
            self.rewriters.append(QueryExpansionRewriter())
    
    async def rewrite(self, query: str) -> list[str]:
        """综合改写"""
        all_queries = set()
        all_queries.add(query)  # 保留原始查询
        
        for rewriter in self.rewriters:
            try:
                variants = await rewriter.rewrite(query)
                all_queries.update(variants)
            except Exception:
                continue
        
        # 去重并限制数量
        return list(all_queries)[:8]
