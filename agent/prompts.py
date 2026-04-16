"""
Prompt 模板管理
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class PromptContext:
    """Prompt 上下文"""
    query: str
    retrieved_chunks: list[dict]  # [{"content": ..., "source": ..., "score": ...}]
    conversation_history: Optional[list[dict]] = None  # [{"role": ..., "content": ...}]
    knowledge_base_name: Optional[str] = None
    user_info: Optional[dict] = None


class PromptBuilder:
    """Prompt 构建器"""
    
    DEFAULT_SYSTEM_PROMPT = """你是一个专业的企业知识库问答助手。你的任务是根据提供的上下文信息，准确、专业地回答用户的问题。

## 回答要求
1. 只根据提供的上下文信息回答，不要编造信息
2. 如果上下文中没有相关信息，诚实地告知用户"抱歉，我在知识库中没有找到相关内容"
3. 回答要条理清晰，使用列表或结构化格式提高可读性
4. 在回答结束时，可以根据上下文提供相关的来源信息
5. 保持专业、友好的语气

## 上下文信息
{context}

## 开始回答
"""

    DEFAULT_USER_PROMPT_TEMPLATE = """用户问题：{query}

请根据上述上下文信息回答用户的问题。"""

    def __init__(
        self,
        system_prompt_template: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
    ):
        self.system_prompt_template = system_prompt_template or self.DEFAULT_SYSTEM_PROMPT
        self.user_prompt_template = user_prompt_template or self.DEFAULT_USER_PROMPT_TEMPLATE
    
    def build_context(self, chunks: list[dict]) -> str:
        """构建上下文内容"""
        if not chunks:
            return "（知识库中暂无相关内容）"
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "未知来源")
            content = chunk.get("content", "")
            score = chunk.get("score", 0)
            
            context_parts.append(
                f"【来源 {i}】{source} (相关度: {score:.2f})\n"
                f"{content}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def build_system_prompt(self, context: PromptContext) -> str:
        """构建系统 Prompt"""
        # 构建上下文
        context_content = self.build_context(context.retrieved_chunks)
        
        # 添加知识库信息
        if context.knowledge_base_name:
            context_content = f"【当前知识库】{context.knowledge_base_name}\n\n" + context_content
        
        return self.system_prompt_template.format(context=context_content)
    
    def build_user_prompt(self, context: PromptContext) -> str:
        """构建用户 Prompt"""
        prompt = self.user_prompt_template.format(query=context.query)
        
        # 添加对话历史
        if context.conversation_history:
            history_parts = []
            for msg in context.conversation_history[-6:]:  # 最近6条
                role = "用户" if msg["role"] == "user" else "助手"
                history_parts.append(f"{role}：{msg['content']}")
            
            if history_parts:
                prompt = "【对话历史】\n" + "\n".join(history_parts) + "\n\n" + prompt
        
        return prompt
    
    def build_messages(
        self,
        context: PromptContext,
    ) -> tuple[str, str]:
        """构建完整的 messages"""
        system_prompt = self.build_system_prompt(context)
        user_prompt = self.build_user_prompt(context)
        
        return system_prompt, user_prompt


class RefinedPromptBuilder(PromptBuilder):
    """增强版 Prompt 构建器（支持追问、引用等）"""
    
    DEFAULT_SYSTEM_PROMPT = """你是一个专业的企业知识库问答助手。你的任务是根据提供的上下文信息，准确、专业地回答用户的问题。

## 核心原则
1. **准确性**：只基于提供的上下文回答，不要编造或臆测信息
2. **诚实性**：如果上下文中没有相关内容，明确告知用户
3. **可追溯性**：在回答中标注信息来源
4. **有帮助性**：除了直接回答，适当补充相关背景知识

## 上下文信息
{context}

## 输出格式
请按以下格式回答：

### 回答
[直接回答用户问题]

### 参考来源
{references}

### 相关追问建议
[提供3个可能的追问方向，帮助用户深入了解]
"""

    def build_system_prompt(self, context: PromptContext) -> str:
        """构建系统 Prompt"""
        context_content = self.build_context(context.retrieved_chunks)
        
        # 构建参考来源
        references = []
        for i, chunk in enumerate(context.retrieved_chunks, 1):
            source = chunk.get("source", "未知来源")
            references.append(f"[{i}] {source}")
        
        system_prompt = self.system_prompt_template.format(
            context=context_content,
            references="\n".join(references) if references else "无",
        )
        
        if context.knowledge_base_name:
            system_prompt = f"【当前知识库】{context.knowledge_base_name}\n\n" + system_prompt
        
        return system_prompt
