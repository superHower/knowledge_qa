"""
数据库初始化脚本
用于创建测试数据和预填充知识库
"""

import asyncio
from datetime import datetime


async def init_database():
    """初始化数据库和基础数据"""
    from knowledge_qa.db.database import init_db, AsyncSessionLocal
    from knowledge_qa.db.models import KnowledgeBase, Document, DocumentChunk
    from knowledge_qa.rag import get_vector_store
    from knowledge_qa.core.config import settings

    print("正在初始化数据库...")

    # 1. 创建表结构
    await init_db()
    print("✓ 数据库表结构已创建")

    # 2. 创建默认知识库
    async with AsyncSessionLocal() as db:
        # 检查是否已有知识库
        from sqlalchemy import select
        stmt = select(KnowledgeBase).where(KnowledgeBase.id == 1)
        result = await db.execute(stmt)
        kb = result.scalar_one_or_none()
        
        if not kb:
            kb = KnowledgeBase(
                id=1,
                name="默认知识库",
                description="系统默认创建的知识库，用于测试",
            )
            db.add(kb)
            await db.commit()
            print("✓ 默认知识库已创建 (id=1)")
        else:
            print("✓ 知识库已存在 (id=1)")

    # 3. 创建测试文档和向量索引
    await create_sample_documents()

    print("\n初始化完成！")
    print(f"数据库连接: {settings.DATABASE_URL}")


async def create_sample_documents():
    """创建示例文档"""
    from knowledge_qa.db.database import AsyncSessionLocal
    from knowledge_qa.db.models import KnowledgeBase, Document, DocumentChunk
    
    async with AsyncSessionLocal() as db:
        # 检查是否已有文档
        from sqlalchemy import select
        stmt = select(Document).where(Document.knowledge_base_id == 1)
        result = await db.execute(stmt)
        existing_docs = result.scalars().all()
        
        if existing_docs:
            print(f"✓ 文档已存在 ({len(existing_docs)} 个)")
            return
        
        # 创建测试文档
        sample_docs = [
            {
                "title": "公司介绍",
                "content": """北京智联科技有限公司成立于2010年，专注于人工智能和大数据领域。
我们的使命是用AI技术赋能企业智能化转型。
公司总部位于北京中关村高科技园区，在上海、深圳设有分支机构。
核心业务包括：智能客服、知识图谱、数据分析平台等。
公司现有员工500余人，其中研发人员占比超过60%。""",
            },
            {
                "title": "产品介绍",
                "content": """我们的智能问答平台支持以下功能：
1. 文档智能解析：支持PDF、Word、Markdown等格式
2. 语义检索：基于向量数据库的语义搜索
3. 多轮对话：支持上下文感知的对话系统
4. 知识图谱：自动构建实体关系网络
5. 定制化训练：根据企业数据定制专属模型

技术优势：
- 支持私有化部署
- API标准化接入
- 高可用架构设计
- 7x24小时技术支持""",
            },
            {
                "title": "常见问题FAQ",
                "content": """Q: 如何开通服务？
A: 请联系我们的销售团队或访问官网申请试用。

Q: 支持哪些文件格式？
A: 目前支持 PDF、Word、Markdown、TXT、HTML、CSV 等常见格式。

Q: 数据安全性如何保障？
A: 我们采用银行级加密标准，支持私有化部署，数据完全由客户掌控。

Q: 如何进行API对接？
A: 提供 RESTful API 接口，配套详细的开发文档和技术支持。

Q: 计费方式是怎样的？
A: 支持按调用次数和包年包月两种模式，具体请咨询销售。""",
            },
        ]
        
        # 创建文档记录
        for doc_data in sample_docs:
            # 创建文档
            doc = Document(
                knowledge_base_id=1,
                file_name=f"{doc_data['title']}.txt",
                file_path=f"/tmp/{doc_data['title']}.txt",
                file_type=".txt",
                title=doc_data['title'],
                status="completed",
                chunk_count=1,
            )
            db.add(doc)
            await db.flush()
            
            # 创建切片
            chunk = DocumentChunk(
                document_id=doc.id,
                content=doc_data['content'],
                content_hash=f"hash_{doc.id}",
                chunk_index=0,
                metadata_={"source": doc_data['title']},
            )
            db.add(chunk)
            await db.flush()
            
            # 索引向量
            try:
                from knowledge_qa.rag.embedding import OpenAIEmbedding
                embedding_model = OpenAIEmbedding(api_key=settings.OPENAI_API_KEY)
                vector = await embedding_model.embed_text(doc_data['content'])
                
                vs = get_vector_store()
                collection_name = "kb_1"
                await vs.create_collection(collection_name, len(vector))
                
                from knowledge_qa.rag.vector_store import VectorPoint
                import hashlib
                vector_id = hashlib.md5(f"1_{chunk.id}".encode()).hexdigest()
                point = VectorPoint(
                    id=vector_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk.id,
                        "content": doc_data['content'],
                        "document_id": doc.id,
                        "document_name": doc.file_name,
                        "metadata": {"source": doc_data['title']},
                    }
                )
                await vs.upsert(collection_name, [point])
                print(f"✓ 文档已创建并索引: {doc_data['title']}")
            except Exception as e:
                print(f"⚠ 向量索引失败: {e} (文档已创建)")
        
        await db.commit()


async def reset_vector_store():
    """重置向量存储（用于调试）"""
    vs = get_vector_store()
    vs.collections.clear()
    print("✓ 向量存储已重置")


if __name__ == "__main__":
    asyncio.run(init_database())
