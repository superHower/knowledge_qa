"""
Embedding 服务
"""

from abc import ABC, abstractmethod
from typing import Protocol
import httpx


class EmbeddingModel(Protocol):
    """Embedding 模型协议"""
    
    async def embed_text(self, text: str) -> list[float]:
        """单个文本向量化"""
        ...
    
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量文本向量化"""
        ...


class OpenAIEmbedding:
    """OpenAI Embedding"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = None,
        model: str = "text-embedding-v4",
        batch_size: int = 10,  # 通义千问限制最大10
        timeout: float = 60.0,
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
        self.batch_size = batch_size
    
    async def embed_text(self, text: str) -> list[float]:
        """单个文本向量化"""
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding
    
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量文本向量化"""
        results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            results.extend([item.embedding for item in response.data])
        
        return results


class LocalEmbedding:
    """本地 Embedding 模型 (Sentence Transformers)"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
    
    async def embed_text(self, text: str) -> list[float]:
        """单个文本向量化"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量文本向量化"""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()


class EmbeddingFactory:
    """Embedding 工厂"""
    
    _models = {
        "openai": OpenAIEmbedding,
        "local": LocalEmbedding,
    }
    
    @classmethod
    def create(
        cls,
        provider: str = "openai",
        **kwargs,
    ) -> EmbeddingModel:
        """创建 Embedding 模型"""
        model_class = cls._models.get(provider)
        if not model_class:
            raise ValueError(f"不支持的 Embedding 提供商: {provider}")
        
        return model_class(**kwargs)
