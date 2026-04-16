"""
应用配置
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===================
    # 应用配置
    # ===================
    APP_NAME: str
    APP_VERSION: str
    DEBUG: bool

    # ===================
    # 数据库配置
    # ===================
    DATABASE_URL: str

    # ===================
    # LLM 配置
    # ===================
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str
    OPENAI_MODEL: str
    OPENAI_EMBEDDING_MODEL: str

    # LLM 生成参数
    LLM_TEMPERATURE: float
    LLM_MAX_TOKENS: int

    # ===================
    # 向量数据库配置
    # ===================
    QDRANT_HOST: str
    QDRANT_PORT: int
    QDRANT_COLLECTION_NAME: str

    # ===================
    # 文档处理配置
    # ===================
    UPLOAD_DIR: str
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int

    # ===================
    # RAG 配置
    # ===================
    TOP_K: int
    SIMILARITY_THRESHOLD: float

    # ===================
    # 日志配置
    # ===================
    LOG_LEVEL: str
    LOG_FILE: str

    # ===================
    # CORS 配置
    # ===================
    CORS_ORIGINS: list[str]


@lru_cache
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


# 导出 settings 实例
settings = get_settings()
