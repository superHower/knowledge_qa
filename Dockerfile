# ===================
# 基础镜像
# ===================
# 使用官方 Python slim 镜像作为基础，体积小且包含必要工具
FROM python:3.11-slim

# ===================
# 系统依赖
# ===================
# 安装系统级依赖：
# - build-essential: 编译工具链（用于安装某些Python包的C扩展）
# - gcc: C编译器
# - curl: 网络请求工具
# - git: 版本控制（某些包可能需要）
# - libpq-dev: PostgreSQL开发库（SQLAlchemy需要）
# - libssl-dev: SSL/TLS开发库
# - libffi-dev: 外部函数接口开发库
# - libxml2-dev, libxslt1-dev: XML/HTML解析库（lxml需要）
# - poppler-utils: PDF处理工具
# - tesseract-ocr: OCR文字识别（图片PDF）
# - libgl1: OpenGL库（某些ML模型需要）
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    git \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ===================
# 工作目录
# ===================
# 设置容器内工作目录
WORKDIR /app

# ===================
# 复制依赖文件
# ===================
# 先复制依赖文件，利用Docker层缓存，避免每次代码变更都重新安装依赖
COPY pyproject.toml .
COPY README.md .

# ===================
# 安装 Python 依赖
# ===================
# 升级pip和构建工具
RUN pip install --no-cache-dir --upgrade pip setuptools wheel hatchling

# 安装 MySQL 驱动（必须在项目依赖之前安装，避免编译问题）
RUN pip install --no-cache-dir aiomysql pymysql cryptography

# 安装项目及其依赖
# -e . 表示可编辑模式安装（开发模式），生产环境可以去掉 -e
RUN pip install --no-cache-dir -e ".[dev]"

# ===================
# 复制源代码
# ===================
# 复制所有Python源代码
COPY agent/ agent/
COPY api/ api/
COPY core/ core/
COPY db/ db/
COPY document/ document/
COPY rag/ rag/
COPY schemas/ schemas/
COPY services/ services/
COPY utils/ utils/
COPY config.py .
COPY main.py .

# ===================
# 创建数据目录
# ===================
# 创建应用运行时需要的目录
RUN mkdir -p /app/data /app/uploads /app/logs

# ===================
# 暴露端口
# ===================
# 声明容器运行时监听的端口（FastAPI默认8000）
EXPOSE 8000

# ===================
# 健康检查
# ===================
# Docker健康检查：每30秒检查一次/api/v1/health接口
# 超时10秒，重试3次失败后标记为unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# ===================
# 启动命令
# ===================
# 使用Uvicorn启动FastAPI应用
# --host 0.0.0.0: 监听所有网络接口（容器外部可访问）
# --port 8000: 服务端口
# --workers 4: 生产环境使用4个worker进程处理并发
# 开发环境可以改为：uvicorn knowledge_qa.main:app --reload --port 8000
CMD ["uvicorn", "knowledge_qa.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
