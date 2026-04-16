"""
FastAPI 应用入口
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from knowledge_qa.core.config import settings
from knowledge_qa.db import init_db
from knowledge_qa.api import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    # 启动时
    # 确保数据目录存在
    Path("./data").mkdir(exist_ok=True)
    Path("./uploads").mkdir(exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)
    
    # 初始化数据库
    await init_db()
    
    yield
    
    # 关闭时
    pass


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="企业知识库问答智能体平台 API",
        lifespan=lifespan,
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 路由
    app.include_router(api_router, prefix="/api/v1")
    
    # 健康检查
    @app.get("/health")
    async def health_check():
        return {"status": "ok", "version": settings.APP_VERSION}
    
    return app


# 创建应用实例
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "knowledge_qa.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
