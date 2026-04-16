-- =====================================================
-- Knowledge QA Agent - MySQL 初始化脚本
-- =====================================================
-- 此脚本在 MySQL 容器首次启动时自动执行
-- 用于初始化数据库和表结构
--
-- 注意：表结构由 SQLAlchemy 在应用启动时自动创建
-- 此脚本主要用于初始化一些默认数据（如需要）

-- 使用 knowledge_qa 数据库
USE knowledge_qa;

-- 设置字符集
SET NAMES utf8mb4;
SET CHARACTER SET utf8mb4;

-- =====================================================
-- 可选：初始化默认数据
-- =====================================================

-- 示例：创建一个默认的知识库（如果需要）
-- INSERT INTO knowledge_bases (name, description, is_active, created_at, updated_at)
-- VALUES ('默认知识库', '系统默认创建的知识库', 1, NOW(), NOW())
-- ON DUPLICATE KEY UPDATE updated_at = NOW();

-- =====================================================
-- 创建索引（优化查询性能）
-- =====================================================

-- 知识库索引
CREATE INDEX IF NOT EXISTS idx_kb_name ON knowledge_bases(name);
CREATE INDEX IF NOT EXISTS idx_kb_active ON knowledge_bases(is_active);

-- 文档索引
CREATE INDEX IF NOT EXISTS idx_doc_kb_id ON documents(knowledge_base_id);
CREATE INDEX IF NOT EXISTS idx_doc_status ON documents(status);

-- 文档切片索引
CREATE INDEX IF NOT EXISTS idx_chunk_doc_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunk_hash ON document_chunks(content_hash);

-- 会话索引
CREATE INDEX IF NOT EXISTS idx_session_kb_id ON chat_sessions(knowledge_base_id);
CREATE INDEX IF NOT EXISTS idx_session_user ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_session_updated ON chat_sessions(updated_at);

-- 消息索引
CREATE INDEX IF NOT EXISTS idx_msg_session ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_msg_created ON chat_messages(created_at);

-- =====================================================
-- 完成提示
-- =====================================================
SELECT 'Knowledge QA Agent 数据库初始化完成！' AS status;
