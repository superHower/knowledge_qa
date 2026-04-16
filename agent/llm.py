"""
LLM 服务
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, Optional
import json
import httpx


@dataclass
class LLMResponse:
    """LLM 响应"""
    content: str
    raw_response: any = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    finish_reason: Optional[str] = None


class BaseLLM(ABC):
    """LLM 基类"""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> LLMResponse:
        """生成文本"""
        pass
    
    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """流式生成文本"""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI LLM"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = None,
        model: str = "qwen3.5-omni-flash",
        timeout: float = 120.0,
    ):
        from openai import AsyncOpenAI
        
        if base_url is None:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        # 禁用环境变量代理
        http_client = httpx.AsyncClient(trust_env=False)
        self.client = AsyncOpenAI(
            api_key=api_key, 
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
            http_client=http_client,
        )
        self.model = model
    
    async def generate(
        self,
        prompt: str | list,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> LLMResponse:
        """生成文本
        
        Args:
            prompt: 可以是字符串，也可以是消息列表
            system_prompt: 系统提示词
            temperature: 温度
            max_tokens: 最大 token 数
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 处理 prompt：支持字符串或消息列表
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif isinstance(prompt, list):
            messages.extend(prompt)
        else:
            messages.append({"role": "user", "content": str(prompt)})
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        choice = response.choices[0]
        usage = response.usage
        
        return LLMResponse(
            content=choice.message.content or "",
            raw_response=response.model_dump(),
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason,
        )
    
    async def stream_generate(
        self,
        prompt: str | list,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """流式生成文本
        
        Args:
            prompt: 可以是字符串，也可以是消息列表
            system_prompt: 系统提示词
            temperature: 温度
            max_tokens: 最大 token 数
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 处理 prompt：支持字符串或消息列表
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif isinstance(prompt, list):
            messages.extend(prompt)
        else:
            messages.append({"role": "user", "content": str(prompt)})
        
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class ClaudeLLM(BaseLLM):
    """Anthropic Claude LLM"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-haiku-20240307",
    ):
        from anthropic import AsyncAnthropic
        
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> LLMResponse:
        """生成文本"""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        
        return LLMResponse(
            content=response.content[0].text if response.content else "",
            raw_response=response.model_dump(),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            finish_reason=response.stop_reason,
        )
    
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """流式生成文本"""
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        ) as stream:
            async for text in stream.text_stream:
                yield text


class LLMFactory:
    """LLM 工厂"""
    
    _providers = {
        "openai": OpenAILLM,
        "claude": ClaudeLLM,
    }
    
    @classmethod
    def create(
        cls,
        provider: str = "openai",
        **kwargs,
    ) -> BaseLLM:
        """创建 LLM"""
        provider_class = cls._providers.get(provider)
        if not provider_class:
            raise ValueError(f"不支持的 LLM 提供商: {provider}")
        
        return provider_class(**kwargs)
