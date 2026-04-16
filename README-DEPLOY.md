# Knowledge QA Agent 部署指南（三容器架构）

**基于 Docker Compose 的标准化部署：**
- 🚀 **FastAPI 应用** 容器（端口 8000）
- 🗄️ **MySQL 数据库** 容器（端口 3306）
- 🔍 **Qdrant 向量库** 容器（端口 6400）

**所有平台（Windows/macOS/Linux）使用同一套命令：`docker-compose up -d`**

---

## 🎯 客户只需做 3 件事

1. **安装 Docker Desktop**（10分钟）
2. **配置 API 密钥**（`.env` 文件中修改 `OPENAI_API_KEY`）
3. **运行命令**：`docker-compose up -d`（1分钟）

访问：`http://localhost:8000/docs` ✅

---

## 📦 交付文件清单

```
knowledge_qa/
├── Dockerfile                      # FastAPI 应用镜像
├── docker-compose.yml               # 三容器编排（应用+MySQL+Qdrant）
├── .env.example                     # 配置模板
├── DEPLOY.md                        # ⭐ 一页纸快速指南
├── README-DEPLOY.md                # 本文件（详细版）
├── docker/
└── （源代码：agent/, api/, rag/ 等）
```

---

## 🚀 3 步快速部署

### 步骤 1：安装 Docker

**Windows 10/11：**
1. 下载 [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. 安装后启动（等待系统托盘出现鲸鱼图标）
3. PowerShell 验证：`docker --version`

**macOS：**
1. 下载 [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
2. 拖动到 Applications，启动
3. 终端验证：`docker --version`

**Linux (Ubuntu/Debian)：**
```bash
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo systemctl start docker
docker --version
```

### 步骤 2：配置（只需改一行）

```bash
# 1. 进入项目文件夹
cd /path/to/knowledge_qa

# 2. 复制配置模板
cp .env.example .env

# 3. 编辑 .env（只需改 OPENAI_API_KEY）
vim .env   # Windows 用 notepad
```

**.env 关键配置：**
```env
# ⭐ 唯一必须修改
OPENAI_API_KEY=sk-your-actual-api-key-here

# 使用其他 LLM（如通义千问）
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-max
```

**获取 OpenAI API Key：** https://platform.openai.com/api-keys

### 步骤 3：启动所有服务

```bash
# 一键启动：FastAPI + MySQL + Qdrant
docker-compose up -d

# 等待 15-30 秒（MySQL 首次启动较慢）
sleep 20

# 验证
curl http://localhost:8000/api/v1/health
# ✅ 应返回：{"status":"ok","version":"0.1.0"}
```

**🎉 部署完成！**

---

## 📊 三容器架构

```
┌─────────────────────────────────────────────────┐
│          OpenClaw / 浏览器                      │
└─────────────────────┬───────────────────────────┘
                       │ HTTP:8000
                       ▼
┌─────────────────────────────────────────────────┐
│    knowledge-qa 容器 (FastAPI)                   │
│  • 知识库管理 API                                │
│  • 文档上��/解析                                 │
│  • RAG 问答（ReAct 推理）                        │
│  • Agent 决策引擎                               │
│                                                │
│  连接：mysql:3306 → qdrant:6400                │
└─────────────────────┬───────────────────────────┘
                       │ Docker 网络
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   MySQL     │ │   Qdrant    │ │             │
│  容器       │ │  容器       │ │             │
│ port:3306  │ │ port:6400  │ │             │
│ • 用户/会话  │ │ • 向量存储   │ │             │
│ • 知识库元数据│ │ • 相似度检索 │ │             │
│ • 文档元数据 │ │             │ │             │
└─────────────┘ └─────────────┘ └─────────────┘
  卷: mysql-data   卷: qdrant-data
```

**通信方式：**
- 容器间通过 **服务名** 通信（Docker DNS 自动解析）
- `knowledge-qa` → `mysql:3306`（MySQL 容器）
- `knowledge-qa` → `qdrant:6400`（Qdrant 容器）
- 客户访问：`http://localhost:8000`（主机端口映射）

---

## 🔌 OpenClaw 集成

### 配置

| 项目 | 值 |
|------|-----|
| 工具类型 | HTTP API |
| 基础 URL | `http://localhost:8000` |
| 超时 | 60 秒 |
| 认证 | 无（可后续添加） |

### API 端点

| 功能 | 方法 | 端点 | 说明 |
|------|------|------|------|
| 健康检查 | GET | `/api/v1/health` | 检查服务状态 |
| 问答 | POST | `/api/v1/chat` | 普通问答 |
| 流式问答 | POST | `/api/v1/chat/stream` | SSE 实时显示思考过程 |
| 创建知识库 | POST | `/api/v1/knowledge-bases` | 新建知识库 |
| 上传文档 | POST | `/api/v1/documents` | PDF/Docx/PPTX 等 |
| 列出知识库 | GET | `/api/v1/knowledge-bases` | 获取所有知识库 |

### 调用示例

**请求：**
```http
POST http://localhost:8000/api/v1/chat
Content-Type: application/json

{
  "query": "产品的保修期是多久？",
  "knowledge_base_id": 1
}
```

**响应：**
```json
{
  "answer": "根据产品手册，本公司产品提供2年质保服务...",
  "confidence": 0.85,
  "citations": [
    {
      "content": "产品自购买之日起提供2年质保...",
      "source": "产品手册.pdf",
      "relevance": 0.92
    }
  ]
}
```

**流式响应（SSE）：**
```
data: {"type": "thought", "content": "正在查询知识库...", "action": "knowledge_base_search"}
data: {"type": "tool_result", "tool": "knowledge_base_search", "observation": "..."}
data: {"type": "final", "answer": "最终回答...", "confidence": 0.85}
data: [DONE]
```

---

## 🛠️ 常用命令

### 所有平台通用（Docker Compose）

```bash
# 启动所有容器（后台）
docker-compose up -d

# 停止
docker-compose down

# 重启
docker-compose restart

# 查看日志
docker-compose logs -f              # 全部日志
docker-compose logs -f knowledge-qa # 应用日志
docker-compose logs -f mysql        # MySQL 日志
docker-compose logs -f qdrant       # Qdrant 日志

# 进入容器调试
docker-compose exec knowledge-qa bash
docker-compose exec mysql mysql -uroot -p123456

# 查看状态
docker-compose ps

# 重新构建（代码更新后）
docker-compose up -d --build

# 删除数据卷（⚠️ 会删除所有数据）
docker-compose down -v
```

### Makefile（macOS/Linux 可选）

```bash
make up      # 启动（等价于 docker-compose up -d）
make down    # 停止
make logs    # 查看日志
make shell   # 进入容器
make health  # 健康检查
make build   # 仅构建镜像
```

---

## ⚙️ 配置说明

### .env 核心配置

```env
# ⭐ 唯一必填：OpenAI API 密钥
OPENAI_API_KEY=sk-xxxxxxxx

# 可选（保持默认）
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
TOP_K=5

# ⚠️ 生产环境务必修改密码！
MYSQL_ROOT_PASSWORD=123456
```

### 修改端口

编辑 `docker-compose.yml`：

```yaml
services:
  knowledge-qa:
    ports:
      - "8080:8000"   # 主机:容器
  mysql:
    ports:
      - "3307:3306"
  qdrant:
    ports:
      - "6334:6400"
```

重启：`docker-compose up -d`

---

## 🔍 健康检查

```bash
# 快速检查
curl http://localhost:8000/api/v1/health

# 完整检查（包含 MySQL、Qdrant、OpenAI）
docker-compose exec knowledge-qa python healthcheck.py

# JSON 格式（监控用）
docker-compose exec knowledge-qa python healthcheck.py --json

# 浏览器查看 API 文档
http://localhost:8000/docs
```

---

## ❓ 常见问题

### Q1：端口冲突

```bash
# 查看端口占用
netstat -ano | findstr :8000   # Windows
lsof -i :8000                  # macOS/Linux

# 修改 docker-compose.yml 映射到其他端口
# "8080:8000" 改为 "8080:8000"
```

### Q2：OpenAI API 返回 401

```bash
# 1. 检查配置
cat .env | grep OPENAI_API_KEY

# 2. 修改 .env
vim .env

# 3. 重启应用
docker-compose restart knowledge-qa
```

### Q3：MySQL 连接失败

```bash
# 1. 确认容器运行
docker-compose ps

# 2. 查看 MySQL 日志
docker-compose logs mysql

# 3. 等待 MySQL 启动（首次约 30-60 秒）

# 4. 重启
docker-compose restart mysql
```

### Q4：Qdrant 连接失败

```bash
# 测试
curl http://localhost:6400/health

# 查看日志
docker-compose logs qdrant

# 重启
docker-compose restart qdrant
```

### Q5：容器启动后立即退出

```bash
# 查看日志
docker-compose logs knowledge-qa

# 常见原因：.env 配置错误、内存不足、磁盘空间不足
```

---

## 📁 数据管理与备份

### 数据存储

```
本地目录：
├── ./uploads/     # 用户上传的文档
├── ./logs/        # 应用日志

Docker Volume：
├── mysql-data     # MySQL 数据
└── qdrant-data    # 向量数据
```

**重要：** 删除容器不会丢失这些数据。

### 备份 MySQL

```bash
# 完整备份
docker-compose exec mysql mysqldump -uroot -p123456 \
  --single-transaction knowledge_qa > backup.sql

# 恢复
cat backup.sql | docker-compose exec -T mysql \
  mysql -uroot -p123456 knowledge_qa
```

### 备份 Qdrant

```bash
# 停止 Qdrant
docker-compose stop qdrant

# 备份 Volume
docker run --rm -v knowledge-qa_qdrant-data:/data \
  -v $(pwd)/backup:/backup alpine \
  tar czf /backup/qdrant.tar.gz /data

# 重启
docker-compose start qdrant
```

---

## 🔄 升级

```bash
# 1. 备份
docker-compose exec mysql mysqldump -uroot -p123456 knowledge_qa > backup.sql

# 2. 更新代码
git pull

# 3. 重新构建启动
docker-compose up -d --build

# 4. 验证
curl http://localhost:8000/api/v1/health
```

---

## 🗑️ 卸载

```bash
# 停止并删除数据卷（⚠️ 会删除所有数据！）
docker-compose down -v

# 删除镜像
docker rmi knowledge-qa-agent

# 清理
docker system prune -a -f

# 删除项目文件夹
rm -rf /path/to/knowledge_qa
```

---

## 📞 技术支持

**自助排查：**
1. `docker-compose ps` - 查看容器状态
2. `docker-compose logs -f` - 查看错误日志
3. `curl http://localhost:8000/api/v1/health` - 测试 API

**故障诊断流程：**
```
1. 确认 3 个容器都是 Up
2. 查看应用日志
3. 测试 API 响应
4. 检查 MySQL/Qdrant
5. 查看资源占用
```

---

## 📋 部署检查清单

- [ ] Docker Desktop 已安装并运行
- [ ] 已复制 `knowledge_qa` 文件夹
- [ ] 已编辑 `.env`，填入 `OPENAI_API_KEY`
- [ ] 已运行 `docker-compose up -d`
- [ ] `docker-compose ps` 显示 3 个容器均为 Up
- [ ] `curl http://localhost:8000/api/v1/health` 返回 `{"status":"ok"}`
- [ ] 浏览器打开 `http://localhost:8000/docs` 看到 API 文档

---

## 🎓 附录

### 环境变量速查

| 变量名 | 说明 | 默认值 | 必需 |
|--------|------|--------|------|
| `OPENAI_API_KEY` | API 密钥 | 无 | ✅ 必填 |
| `OPENAI_BASE_URL` | API 地址 | `https://api.openai.com/v1` | 否 |
| `MYSQL_ROOT_PASSWORD` | MySQL 密码 | `123456` | ⚠️ 生产改 |
| `QDRANT_HOST` | Qdrant 主机 | `qdrant` | 否 |

### 端口对照表

| 服务 | 容器端口 | 主机端口 | 说明 |
|------|---------|---------|------|
| FastAPI | 8000 | 8000 | HTTP API |
| MySQL | 3306 | 3306 | 数据库 |
| Qdrant | 6400 | 6400 | 向量库 |

### 三容器依赖关系

应用容器等待 MySQL 和 Qdrant 健康后启动：
```yaml
depends_on:
  mysql:
    condition: service_healthy
  qdrant:
    condition: service_healthy
```

---

**文档版本：** v2.0（三容器架构）  
**更新日期：** 2026-04-14  
**适用版本：** Knowledge QA Agent v0.1.0+

**🚀 核心总结：客户只需做 3 件事——装 Docker → 改 API Key → docker-compose up -d**