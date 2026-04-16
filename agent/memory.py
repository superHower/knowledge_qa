"""
记忆系统

短期记忆：会话上下文
长期记忆：用户偏好、历史交互
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import json
import hashlib


@dataclass
class MemoryEntry:
    """记忆条目"""
    content: str
    timestamp: datetime
    memory_type: str  # "user_message", "assistant_response", "tool_result", "preference"
    metadata: dict = field(default_factory=dict)
    importance: float = 0.5  # 重要性 0-1
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "memory_type": self.memory_type,
            "metadata": self.metadata,
            "importance": self.importance,
        }


@dataclass
class UserPreference:
    """用户偏好"""
    user_id: str
    preferred_style: str = "concise"  # "concise", "detailed", "formal"
    frequent_topics: list[str] = field(default_factory=list)
    interaction_count: int = 0
    last_interaction: Optional[datetime] = None
    custom_preferences: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "preferred_style": self.preferred_style,
            "frequent_topics": self.frequent_topics,
            "interaction_count": self.interaction_count,
            "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None,
            "custom_preferences": self.custom_preferences,
        }


class ShortTermMemory:
    """短期记忆
    
    管理当前会话的上下文，
    包含对话历史、当前任务状态等
    """
    
    def __init__(
        self,
        max_messages: int = 20,
        max_tokens: int = 8000,
    ):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.messages: list[MemoryEntry] = []
        self.current_task: Optional[str] = None
        self.context_variables: dict = {}  # 任务相关的上下文变量
    
    def add(
        self,
        content: str,
        memory_type: str,
        metadata: Optional[dict] = None,
        importance: float = 0.5,
    ):
        """添加记忆"""
        entry = MemoryEntry(
            content=content,
            timestamp=datetime.utcnow(),
            memory_type=memory_type,
            metadata=metadata or {},
            importance=importance,
        )
        self.messages.append(entry)
        
        # 超过上限时，优先保留重要的
        if len(self.messages) > self.max_messages:
            self._prune()
    
    def add_user_message(self, content: str, metadata: Optional[dict] = None):
        """添加用户消息"""
        self.add(content, "user_message", metadata, importance=0.7)
    
    def add_assistant_message(self, content: str, metadata: Optional[dict] = None):
        """添加助手消息"""
        self.add(content, "assistant_response", metadata, importance=0.7)
    
    def add_tool_result(
        self,
        tool_name: str,
        result: str,
        success: bool = True,
    ):
        """添加工具结果"""
        self.add(
            f"[{tool_name}] {result}",
            "tool_result",
            metadata={"tool": tool_name, "success": success},
            importance=0.5,
        )
    
    def add_preference(self, topic: str, importance: float = 0.6):
        """添加偏好"""
        self.add(topic, "preference", metadata={"topic": topic}, importance=importance)
    
    def get_recent(self, n: int = 10) -> list[MemoryEntry]:
        """获取最近 n 条记忆"""
        return self.messages[-n:]
    
    def get_messages_for_llm(self) -> list[dict]:
        """获取用于 LLM 的消息格式"""
        return [
            {"role": "user" if m.memory_type == "user_message" else "assistant",
             "content": m.content}
            for m in self.messages
        ]
    
    def get_context_summary(self) -> str:
        """获取上下文摘要"""
        if not self.messages:
            return ""
        
        recent = self.messages[-5:]
        parts = [f"[{m.memory_type}] {m.content[:100]}" for m in recent]
        return "\n".join(parts)
    
    def _prune(self):
        """修剪低重要性记忆"""
        # 按重要性排序，保留重要的
        self.messages.sort(key=lambda x: x.importance, reverse=True)
        self.messages = self.messages[:self.max_messages]
    
    def clear(self):
        """清空短期记忆"""
        self.messages.clear()
        self.current_task = None
        self.context_variables.clear()
    
    def set_context(self, key: str, value: Any):
        """设置上下文变量"""
        self.context_variables[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """获取上下文变量"""
        return self.context_variables.get(key, default)


class LongTermMemory:
    """长期记忆
    
    持久化存储用户偏好和历史交互模式，
    用于个性化服务和连续对话
    """

    def __init__(self, db_session=None):
        self.db = db_session
        self._preferences_cache: dict[str, UserPreference] = {}
    
    async def save_preference(self, preference: UserPreference):
        """保存用户偏好"""
        self._preferences_cache[preference.user_id] = preference
        
        # 如果有数据库，持久化
        if self.db:
            # TODO: 实现数据库持久化
            pass
    
    async def get_preference(self, user_id: str) -> Optional[UserPreference]:
        """获取用户偏好"""
        if user_id in self._preferences_cache:
            return self._preferences_cache[user_id]
        
        # 从数据库加载
        if self.db:
            # TODO: 从数据库加载
            pass
        
        return None
    
    async def update_interaction(
        self,
        user_id: str,
        topic: Optional[str] = None,
        style: Optional[str] = None,
    ):
        """更新交互记录"""
        pref = await self.get_preference(user_id)
        
        if not pref:
            pref = UserPreference(user_id=user_id)
        
        pref.interaction_count += 1
        pref.last_interaction = datetime.utcnow()
        
        if topic and topic not in pref.frequent_topics:
            pref.frequent_topics.append(topic)
            # 保留最近 20 个高频话题
            pref.frequent_topics = pref.frequent_topics[-20:]
        
        if style:
            pref.preferred_style = style
        
        await self.save_preference(pref)
    
    async def extract_patterns(self, user_id: str) -> dict:
        """提取用户模式"""
        pref = await self.get_preference(user_id)
        if not pref:
            return {}
        
        return {
            "frequent_topics": pref.frequent_topics,
            "preferred_style": pref.preferred_style,
            "interaction_count": pref.interaction_count,
        }
    
    def apply_preferences(
        self,
        preference: Optional[UserPreference],
        default_style: str = "concise",
    ) -> dict:
        """应用偏好到响应配置"""
        if not preference:
            return {"style": default_style}
        
        return {
            "style": preference.preferred_style,
            "frequent_topics": preference.frequent_topics,
        }


class WorkingMemory:
    """工作记忆
    
    整合短期记忆和长期记忆，
    为当前任务提供统一的记忆访问接口
    """
    
    def __init__(
        self,
        short_term: Optional[ShortTermMemory] = None,
        long_term: Optional[LongTermMemory] = None,
    ):
        self.short_term = short_term or ShortTermMemory()
        self.long_term = long_term or LongTermMemory()
    
    def add_message(self, role: str, content: str):
        """添加消息"""
        if role == "user":
            self.short_term.add_user_message(content)
        else:
            self.short_term.add_assistant_message(content)
    
    def add_tool_use(self, tool_name: str, result: str, success: bool = True):
        """添加工具使用记录"""
        self.short_term.add_tool_result(tool_name, result, success)
    
    async def get_context_for_llm(
        self,
        user_id: Optional[str] = None,
    ) -> dict:
        """获取用于 LLM 的完整上下文"""
        # 短期记忆
        recent_messages = self.short_term.get_messages_for_llm()
        context_summary = self.short_term.get_context_summary()
        
        # 长期偏好
        preferences = {}
        if user_id and self.long_term:
            pref = await self.long_term.get_preference(user_id)
            if pref:
                preferences = self.long_term.apply_preferences(pref)
        
        return {
            "recent_messages": recent_messages,
            "context_summary": context_summary,
            "preferences": preferences,
            "current_task": self.short_term.current_task,
        }
    
    def clear(self):
        """清空工作记忆"""
        self.short_term.clear()


class EpisodicMemory:
    """情景记忆
    
    记录完整的交互片段，
    用于回顾和反思学习
    """
    
    def __init__(self, db_session=None):
        self.db = db_session
        self.current_episode: list[MemoryEntry] = []
    
    def start_episode(self, trigger: str):
        """开始一个新片段"""
        self.current_episode = [
            MemoryEntry(
                content=f"Episode started: {trigger}",
                timestamp=datetime.utcnow(),
                memory_type="episode_start",
            )
        ]
    
    def add_to_episode(self, content: str, memory_type: str = "step"):
        """添加步骤到当前片段"""
        self.current_episode.append(
            MemoryEntry(
                content=content,
                timestamp=datetime.utcnow(),
                memory_type=memory_type,
            )
        )
    
    def end_episode(self, outcome: str, success: bool) -> list[MemoryEntry]:
        """结束片段并返回"""
        self.current_episode.append(
            MemoryEntry(
                content=f"Episode ended: {outcome}, success={success}",
                timestamp=datetime.utcnow(),
                memory_type="episode_end",
                importance=1.0 if success else 0.3,
            )
        )
        
        episode = self.current_episode.copy()
        self.current_episode.clear()
        return episode
    
    async def save_episode(self, episode: list[MemoryEntry], user_id: str):
        """保存片段到存储"""
        # TODO: 实现片段持久化
        pass
    
    async def get_recent_episodes(self, n: int = 10) -> list[dict]:
        """获取最近的片段"""
        # TODO: 从存储中获取
        return []


class ReflectionMemory:
    """反思记忆
    
    Agent 自我反思的记录，
    用于改进决策策略
    """
    
    def __init__(self):
        self.reflections: list[dict] = []
    
    def add_reflection(
        self,
        situation: str,
        action: str,
        outcome: str,
        lesson: str,
    ):
        """添加反思"""
        self.reflections.append({
            "situation": situation,
            "action": action,
            "outcome": outcome,
            "lesson": lesson,
            "timestamp": datetime.utcnow(),
        })
    
    def get_relevant_lessons(self, situation: str, n: int = 3) -> list[str]:
        """获取相关教训"""
        # 简单实现，实际可以使用语义搜索
        return [r["lesson"] for r in self.reflections[-n:]]
    
    def get_insights(self) -> list[str]:
        """获取洞察"""
        return list(set(r["lesson"] for r in self.reflections))
