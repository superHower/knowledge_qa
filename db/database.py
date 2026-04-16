"""
数据库管理
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import make_url

from knowledge_qa.core.config import settings


# 数据库 URL 处理
db_url = settings.DATABASE_URL

# 使用 SQLAlchemy URL 对象来正确解析
url = make_url(db_url)

if url.drivername not in {"mysql", "mysql+pymysql", "mysql+aiomysql"}:
    raise ValueError("DATABASE_URL 必须使用 MySQL 连接串")

# 异步引擎 (用于 FastAPI)
async_db_url = url.set(drivername="mysql+aiomysql")

async_engine = create_async_engine(
    async_db_url,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

# 同步引擎 (用于初始化等场景)
# 启动阶段的 create_all 必须使用同步驱动，避免 MissingGreenlet
sync_db_url = url.set(drivername="mysql+pymysql")

sync_engine = create_engine(
    sync_db_url,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

SyncSessionLocal = sessionmaker(bind=sync_engine, autoflush=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """获取数据库会话的依赖"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """上下文管理器方式的数据库会话"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """初始化数据库表"""
    import pymysql
    from knowledge_qa.db.models import Base

    # 如果是 MySQL，先确保数据库存在
    if url.drivername in ["mysql", "mysql+pymysql", "mysql+aiomysql"]:
        # 1. 确保数据库存在
        conn = pymysql.connect(
            host=url.host or '127.0.0.1',
            port=url.port or 3306,
            user=url.username or 'root',
            password=url.password or '123456',
            charset='utf8mb4'
        )
        try:
            with conn.cursor() as cursor:
                db_name = url.database or 'knowledge_qa'
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            conn.commit()
        finally:
            conn.close()

    # 2. 使用 SQLAlchemy 同步引擎创建所有表
    Base.metadata.create_all(bind=sync_engine)
    sync_engine.dispose()


async def drop_db() -> None:
    """删除所有数据库表"""
    from knowledge_qa.db.models import Base
    
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
