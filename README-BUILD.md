# Docer-Build
## 当前部署架构

基于 [docker-compose.yml](file:///d:/MyWork/project-agent/code/knowledge_qa/docker-compose.yml) ，当前项目包含以下服务：

| 服务 | Compose 服务名 | 默认端口 | 作用 |
| --- | --- | --- | --- |
| 应用服务 | `knowledge-qa` | `8000` | FastAPI API、文档处理、RAG、Agent |
| MySQL | `mysql` | `3306` | 业务元数据、知识库、文档、会话 |
| Qdrant | `qdrant` | `6400` | 向量检索与切片索引 |

容器间通过 Docker 网络内服务名通信：

- 应用访问 MySQL：`mysql:3306`
- 应用访问 Qdrant：`qdrant:6400`

## 构建前准备

### 1. 安装 Docker

- Windows/macOS：安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Linux：安装 Docker Engine 和 Compose Plugin

验证命令：

```bash
docker --version
docker compose version
```

### 2. 准备环境变量

在项目根目录执行：

```bash
cp .env.example .env
```

开发人员至少检查这些变量：

```env
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
LOG_LEVEL=INFO
```

说明：

- `docker-compose.yml` 会将 `OPENAI_*`、`LLM_*`、`TOP_K` 等参数注入应用容器
- 当前 Compose 文件中的 `DATABASE_URL` 已固定为容器内 MySQL 地址
- 本地非 Docker 启动使用的 `.env` 与 Docker Compose 环境变量是两套加载方式，不冲突

### 3. 检查目录

确保项目目录存在以下内容：

- [Dockerfile](file:///d:/MyWork/project-agent/code/knowledge_qa/Dockerfile)
- [docker-compose.yml](file:///d:/MyWork/project-agent/code/knowledge_qa/docker-compose.yml)
- [.env.example](file:///d:/MyWork/project-agent/code/knowledge_qa/.env.example)
- [docker/mysql/my.cnf](file:///d:/MyWork/project-agent/code/knowledge_qa/docker/mysql/my.cnf)
- [docker/mysql/init.sql](file:///d:/MyWork/project-agent/code/knowledge_qa/docker/mysql/init.sql)

## 标准构建流程

### 1. 构建应用镜像

```bash
docker compose build knowledge-qa
```

如果需要写入构建元信息：

```bash
set BUILD_DATE=2026-04-17
set VCS_REF=local-dev
docker compose build knowledge-qa
```

PowerShell：

```powershell
$env:BUILD_DATE="2026-04-17"
$env:VCS_REF="local-dev"
docker compose build knowledge-qa
```

### 2. 启动完整编排

```bash
docker compose up -d
```

首次启动建议等待 30 到 60 秒，原因：

- MySQL 首次初始化会创建数据库目录
- Qdrant 需要准备存储目录
- 应用启动时会执行 `init_db()`

### 3. 查看状态

```bash
docker compose ps
```

### 4. 查看日志

```bash
docker compose logs -f knowledge-qa
docker compose logs -f mysql
docker compose logs -f qdrant
```

## 部署验证

### 1. 接口验证

```bash
curl http://localhost:8000/health
```

## 开发环境常用命令

### 启动与停止

```bash
docker compose up -d
docker compose down
docker compose down -v
```

## 升级流程

### 代码升级

```bash
git pull
docker compose up -d --build
```

### 升级前备份 MySQL

```bash
docker compose exec mysql mysqldump -uroot -p123456 knowledge_qa > backup.sql
```

### 恢复 MySQL

```bash
cat backup.sql | docker compose exec -T mysql mysql -uroot -p123456 knowledge_qa
```
# README-DEPLOY

面向客户的运行说明。本文档不讨论源码结构和内部实现，只告诉客户如何在自己的电脑上把项目跑起来并访问可用页面。

## 适用对象

- 最终客户
- 现场实施人员
- 不参与项目开发、只需要运行系统的人

## 客户需要准备什么

客户只需要准备 3 样东西：

1. 一台能安装 Docker 的电脑
2. 项目文件夹 `knowledge_qa`
3. 一个可用的模型 API Key

## 运行方式

当前项目推荐使用 Docker Compose 运行，启动后会自动带起：

- 应用服务
- MySQL 数据库
- Qdrant 向量库

客户不需要单独安装 Python、MySQL 或 Qdrant。

## 第一步：安装 Docker

### Windows

1. 安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. 安装完成后启动 Docker Desktop
3. 打开 PowerShell，执行：

```powershell
docker --version
docker compose version
```

### macOS

1. 安装 Docker Desktop
2. 启动 Docker Desktop
3. 打开终端执行：

```bash
docker --version
docker compose version
```

### Linux

自行安装 Docker Engine 与 Compose Plugin，确认以下命令可用：

```bash
docker --version
docker compose version
```

## 第二步：进入项目目录

```bash
cd knowledge_qa
```

## 第三步：配置 `.env`

如果目录里还没有 `.env`，先从模板复制一份：

```bash
cp .env.example .env
```

Windows 可以直接复制文件后重命名。

客户通常只需要重点检查以下配置：

```env
OPENAI_API_KEY=你的真实密钥
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

如果使用兼容 OpenAI 的其他模型平台，也可以改成对应地址和模型名。

## 第四步：启动项目

在项目根目录执行：

```bash
docker compose up -d
```

## 第五步：确认是否启动成功

```bash
docker compose ps
```
