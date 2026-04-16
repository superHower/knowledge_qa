-- =====================================================
-- Knowledge QA Agent - MySQL 备份脚本
-- =====================================================
-- 用途：备份 knowledge_qa 数据库的所有数据
--
-- 使用方法：
--   docker-compose exec mysql mysqldump -uroot -p123456 knowledge_qa > backup.sql
--   或使用此文件作为参考

-- =====================================================
-- 备份命令说明
-- =====================================================

-- 1. 完整备份（结构和数据）
-- docker-compose exec mysql mysqldump -uroot -p123456 --single-transaction --routines --triggers knowledge_qa > backup_$(date +%Y%m%d).sql

-- 2. 仅结构备份
-- docker-compose exec mysql mysqldump -uroot -p123456 --no-data knowledge_qa > structure.sql

-- 3. 仅数据备份
-- docker-compose exec mysql mysqldump -uroot -p123456 --no-create-info knowledge_qa > data.sql

-- 4. 恢复备份
-- cat backup.sql | docker-compose exec -T mysql mysql -uroot -p123456 knowledge_qa

-- =====================================================
-- 定时备份（Crontab 示例）
-- =====================================================

-- 每天凌晨 3 点自动备份
-- 0 3 * * * docker-compose exec mysql mysqldump -uroot -p123456 --single-transaction knowledge_qa > /opt/backups/kb_qa_$(date +\%Y\%m\%d).sql

-- 保留最近 30 天的备份
-- 0 4 * * * find /opt/backups -name "kb_qa_*.sql" -mtime +30 -delete
