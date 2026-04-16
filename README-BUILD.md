# Knowledge QA Agent - Docker 快速部署

**三容器架构：FastAPI（8000）+ MySQL（3306）+ Qdrant（6400）**

---

## 🎯 3 步部署（客户操作流程）

### 1️⃣ 安装 Docker Desktop

- **Windows/macOS**：下载 [Docker Desktop](https://www.docker.com/products/docker-desktop/) 并安装
- **Linux**：`sudo apt install docker-ce docker-compose-plugin`

验证：`docker --version`

### 2️⃣ 配置 API 密钥

```bash
cd /path/to/knowledge_qa
cp .env.example .env
vim .env   # 或 notepad .env（Windows）
```

**只需改一行：**
```env
OPENAI_API_KEY=sk-customer-api-key-here
```

### 3️⃣ 启动服务

```bash
docker-compose up -d
sleep 20
curl http://localhost:8000/api/v1/health
# 看到 {"status":"ok"} 即成功 ✅
```

访问：http://localhost:8000/docs

---

## 📦 三容器说明

| 容器 | 端口 | 用途 | 数据持久化 |
|------|------|------|-----------|
| `knowledge-qa` | 8000 | FastAPI 应用 | `./uploads`, `./logs` |
| `mysql` | 3306 | MySQL 数据库 | `mysql-data` Volume |
| `qdrant` | 6400 | 向量数据库 | `qdrant-data` Volume |

**通信方式：** Docker 内部 DNS 自动解析（`mysql`、`qdrant`）

---

## 🔌 OpenClaw 集成

```json
{
  "name": "knowledge_qa_agent",
  "type": "http_api",
  "config": {
    "base_url": "http://localhost:8000",
    "timeout": 60
  }
}
```

**核心接口：**
- 问答：`POST /api/v1/chat`
- 流式：`POST /api/v1/chat/stream`
- 管理：`POST /api/v1/knowledge-bases`、`POST /api/v1/documents`

---

## 🛠️ 常用命令

```bash
docker-compose up -d          # 启动
docker-compose down           # 停止
docker-compose logs -f        # 查看日志
docker-compose restart        # 重启
docker-compose ps             # 查看状态

# 进入容器
docker-compose exec knowledge-qa bash
docker-compose exec mysql mysql -uroot -p123456
```

---

## ❓ 常见问题

### 端口被占用
修改 `docker-compose.yml` 中的端口映射，如 `"8080:8000"`，重启。

### OpenAI API 失败
检查 `.env` 中的 `OPENAI_API_KEY`，确保密钥有效。

### MySQL 连接失败
```bash
docker-compose logs mysql     # 查看日志
docker-compose restart mysql  # 重启
# 首次启动可能需要 30-60 秒
```

### 数据备份
```bash
# 备份 MySQL
docker-compose exec mysql mysqldump -uroot -p123456 knowledge_qa > backup.sql

# 恢复
cat backup.sql | docker-compose exec -T mysql mysql -uroot -p123456 knowledge_qa
```

---

## 📞 技术支持

1. 日志：`docker-compose logs -f`
2. API 文档：http://localhost:8000/docs
3. 健康检查：`curl http://localhost:8000/api/v1/health`

详细文档见 `README-DEPLOY.md`