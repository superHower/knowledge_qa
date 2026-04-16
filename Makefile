# ===================
# Makefile - 便捷操作命令
# ===================
# 用途：简化 Docker 和 Python 项目的常用操作
# 使用：make [目标]
# 示例：make build, make up, make test, make logs
#
# 支持的操作：
# - 环境设置
# - 构建和部署
# - 开发和调试
# - 测试
# - 清理
# - 文档

.PHONY: help build up down restart logs shell test clean lint format dev prod check-health push

# ===================
# 变量定义
# ===================
PROJECT_NAME = knowledge-qa
PYTHON = python3.11
POETRY = poetry
DOCKER_COMPOSE = docker-compose
DOCKER_IMAGE = knowledge-qa-agent
REGISTRY ?= docker.io
NAMESPACE ?= your-username
TAG ?= latest
FULL_IMAGE = $(REGISTRY)/$(NAMESPACE)/$(DOCKER_IMAGE):$(TAG)

# ===================
# 默认目标：显示帮助
# ===================
.DEFAULT_GOAL := help

help: ## 显示此帮助信息
	@echo "Knowledge QA Agent - 可用命令："
	@echo ""
	@echo "环境设置:"
	@echo "  make install         安装 Python 依赖"
	@echo "  make dev-setup       开发环境完整设置"
	@echo ""
	@echo "Docker 构建和部署:"
	@echo "  make build           构建 Docker 镜像"
	@echo "  make up              启动所有服务（后台运行）"
	@echo "  make down            停止所有服务"
	@echo "  make restart         重启服务"
	@echo "  make logs            查看实时日志"
	@echo "  make logs-qa         查看应用日志"
	@echo "  make logs-qdrant     查看Qdrant日志"
	@echo ""
	@echo "调试命令:"
	@echo "  make shell           进入应用容器"
	@echo "  make shell-qdrant    进入Qdrant容器"
	@echo "  make db-shell        进入数据库Shell"
	@echo "  make exec-cmd CMD='...'  在容器中执行命令"
	@echo ""
	@echo "开发模式:"
	@echo "  make dev             开发模式启动（热重载）"
	@echo "  make test            运行测试"
	@echo "  make lint            代码检查"
	@echo "  make format          代码格式化"
	@echo ""
	@echo "生产部署:"
	@echo "  make prod            生产模式启动"
	@echo "  make scale N=4       扩展到N个worker"
	@echo "  make push            构建并推送到镜像仓库"
	@echo "  make deploy          一键部署（构建+推送+启动）"
	@echo ""
	@echo "健康检查和监控:"
	@echo "  make health          检查API健康状态"
	@echo "  make check-all       完整健康检查"
	@echo "  make monitor         启动监控（需要prometheus）"
	@echo ""
	@echo "清理:"
	@echo "  make clean           清理临时文件"
	@echo "  make clean-docker    清理Docker资源"
	@echo "  make clean-all       彻底清理（包括数据卷⚠️）"
	@echo ""
	@echo "文档:"
	@echo "  make docs            生成API文档"
	@echo "  make open-docs       在浏览器打开API文档"
	@echo ""
	@echo "配置:"
	@echo "  make config-edit     编辑环境配置"
	@echo "  make config-show     显示当前配置"
	@echo "  make config-validate 验证配置文件"
	@echo ""
	@echo "更多信息请查看 README-DEPLOY.md"

# ===================
# 环境设置
# ===================
install: ## 安装 Python 依赖
	@echo "📦 安装依赖..."
	$(POETRY) install
	@echo "✓ 安装完成"

dev-setup: ## 开发环境完整设置
	@echo "🚀 设置开发环境..."
	$(POETRY) install
	cp .env.example .env
	@echo "✓ 开发环境就绪！"
	@echo "请编辑 .env 文件配置 OPENAI_API_KEY"

# ===================
# Docker 构建和部署
# ===================
build: ## 构建 Docker 镜像
	@echo "🔨 构建 Docker 镜像..."
	$(DOCKER_COMPOSE) build
	@echo "✓ 构建完成"

up: ## 启动所有服务
	@echo "▶️  启动服务..."
	$(DOCKER_COMPOSE) up -d
	@echo "✓ 服务已启动"
	@echo "访问地址: http://localhost:8000"
	@echo "API文档: http://localhost:8000/docs"

down: ## 停止所有服务
	@echo "⏹️  停止服务..."
	$(DOCKER_COMPOSE) down
	@echo "✓ 服务已停止"

stop: down ## 别名

restart: down up ## 重启服务

logs: ## 查看所有服务日志
	$(DOCKER_COMPOSE) logs -f

logs-qa: ## 查看应用日志
	$(DOCKER_COMPOSE) logs -f $(PROJECT_NAME)

logs-qdrant: ## 查看Qdrant日志
	$(DOCKER_COMPOSE) logs -f qdrant

# ===================
# 调试命令
# ===================
shell: ## 进入应用容器
	$(DOCKER_COMPOSE) exec $(PROJECT_NAME) bash

shell-qdrant: ## 进入Qdrant容器
	$(DOCKER_COMPOSE) exec qdrant bash

db-shell: ## 进入数据库Shell
	$(DOCKER_COMPOSE) exec $(PROJECT_NAME) python -c "from knowledge_qa.db.database import AsyncSessionLocal; import asyncio; async def main(): async with AsyncSessionLocal() as db: print('数据库连接成功'); asyncio.run(main())"

exec-cmd: ## 在容器中执行命令（用法：make exec-cmd CMD="ls -la"）
	$(DOCKER_COMPOSE) exec $(PROJECT_NAME) $(CMD)

# ===================
# 开发模式
# ===================
dev: ## 开发模式启动（热重载）
	@echo "🔧 开发模式启动..."
	$(POETRY) run uvicorn knowledge_qa.main:app --reload --host 0.0.0.0 --port 8000

test: ## 运行测试
	@echo "🧪 运行测试..."
	$(POETRY) run pytest tests/ -v

test-cov: ## 运行测试并生成覆盖率报告
	$(POETRY) run pytest --cov=knowledge_qa --cov-report=html

lint: ## 代码检查
	@echo "🔍 代码检查..."
	$(POETRY) run ruff check .
	$(POETRY) run mypy knowledge_qa/

format: ## 代码格式化
	@echo "✨ 格式化代码..."
	$(POETRY) run ruff format .

# ===================
# 生产部署
# ===================
prod: ## 生产模式启动
	@echo "🏭 生产模式启动..."
	$(DOCKER_COMPOSE) up -d
	@echo "✓ 服务已启动"
	@echo "提示：使用 'make logs' 查看日志"

scale: ## 扩展worker数量（用法：make scale N=4）
	@echo "📈 扩展到 $(N) 个worker..."
	$(DOCKER_COMPOSE) up -d --scale $(PROJECT_NAME)=$(N)

push: build ## 构建并推送到镜像仓库
	@echo "📤 推送镜像到 $(FULL_IMAGE)..."
	docker tag $(DOCKER_IMAGE):latest $(FULL_IMAGE)
	docker push $(FULL_IMAGE)
	@echo "✓ 推送完成"

deploy: push ## 一键部署（构建+推送+启动）
	@echo "🚀 更新部署..."
	# 更新 docker-compose.yml 中的镜像（如果需要）
	@echo "✓ 部署完成"

# ===================
# 健康检查
# ===================
health: ## 检查API健康状态
	@echo "❤️  健康检查..."
	@curl -s http://localhost:8000/api/v1/health | python -m json.tool || echo "❌ 服务未运行"

check-all: ## 完整健康检查
	@echo "🩺 完整健康检查..."
	$(POETRY) run python healthcheck.py

check-json: ## JSON格式健康检查（用于监控）
	$(POETRY) run python healthcheck.py --json

monitor: ## 启动监控（需要prometheus配置）
	@echo "📊 监控功能需要额外配置Prometheus/Grafana"
	@echo "请参考 README-DEPLOY.md 第9节"

# ===================
# 清理
# ===================
clean: ## 清理Python临时文件
	@echo "🧹 清理临时文件..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .coverage htmlcov
	@echo "✓ 清理完成"

clean-docker: ## 清理Docker资源（未使用的镜像、容器、网络）
	@echo "🧹 清理Docker资源..."
	docker system prune -f
	@echo "✓ 清理完成"

clean-all: ## 彻底清理（包括数据卷⚠️）
	@echo "💥 彻底清理（将删除所有数据！）..."
	@read -p "确认删除所有数据？(yes/no): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		$(DOCKER_COMPOSE) down -v; \
		docker system prune -a -f; \
		echo "✓ 已彻底清理"; \
	else \
		echo "取消操作"; \
	fi

# ===================
# 文档
# ===================
docs: ## 生成API文档（需要构建）
	@echo "📚 生成文档..."
	@echo "访问 http://localhost:8000/docs 查看在线文档"

open-docs: ## 在浏览器打开API文档
	@echo "🌐 打开API文档..."
	@if command -v xdg-open > /dev/null; then \
		xdg-open http://localhost:8000/docs; \
	elif command -v open > /dev/null; then \
		open http://localhost:8000/docs; \
	else \
		echo "请手动打开: http://localhost:8000/docs"; \
	fi

# ===================
# 配置
# ===================
config-edit: ## 编辑环境配置
	@echo "✏️  编辑配置文件..."
	@if [ -f ".env" ]; then \
		$$EDITOR .env; \
	else \
		cp .env.example .env; \
		$$EDITOR .env; \
	fi

config-show: ## 显示当前配置（脱敏）
	@echo "⚙️  当前配置（敏感信息已隐藏��："
	@if [ -f ".env" ]; then \
		grep -v "API_KEY\|SECRET\|PASSWORD" .env | grep -v "^#" | grep -v "^$$" || true; \
	else \
		echo "未找到 .env 文件"; \
	fi

config-validate: ## 验证配置文件
	@echo "🔐 验证配置..."
	$(POETRY) run python verify_configs.py

# ===================
# 快速开始
# ===================
quickstart: ## 快速开始指南
	@echo "🎯 快速开始"
	@echo ""
	@echo "1. 配置环境变量："
	@echo "   cp .env.example .env"
	@echo "   vim .env  # 编辑 OPENAI_API_KEY"
	@echo ""
	@echo "2. 启动服务："
	@echo "   make up"
	@echo ""
	@echo "3. 验证："
	@echo "   make health"
	@echo ""
	@echo "4. 访问文档："
	@echo "   http://localhost:8000/docs"
	@echo ""
	@echo "更多信息：make help"

# ===================
# Windows 辅助命令（在WSL中使用）
# ===================
win-ip: ## 获取Windows主机IP（WSL环境）
	@echo "获取Windows主机IP..."
	@cat /etc/resolv.conf | grep nameserver | awk '{print $$2}'

win-build: ## Windows环境构建（使用PowerShell脚本）
	@echo "在Windows上，请使用 PowerShell 运行："
	@echo "  .\\build-and-push.ps1"
	@echo "或使用 Docker Desktop GUI"

# ===================
# 安全检查
# ===================
security-scan: ## 安全扫描（需要trivy）
	@echo "🔒 安全扫描..."
	trivy image $(DOCKER_IMAGE):latest || true

# ===================
# 性能测试
# ===================
load-test: ## 负载测试（需要locust）
	@echo "⚡ 负载测试..."
	locust -f tests/load_test.py --host http://localhost:8000
