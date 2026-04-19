# 代码改进记录

## 1. 为 LangGraph 节点注入依赖

**状态**: ✅ 已完成

**问题描述**:
- 节点函数内部每次调用都创建新的 LLM、Embedding、Retriever 实例
- 难以测试和复用
- 连接资源浪费

**解决方案**:
- 创建 `GraphDependencies` 依赖注入容器
- 节点函数接受 `deps: GraphDependencies` 参数
- 实现单例模式复用 LLM 和 Retriever

**修改文件**:
- `graph/dependencies.py` (新增)
- `graph/nodes.py` (重构)
- `graph/graph.py` (重构)
- `graph/__init__.py` (更新导出)
- `api/chat.py` (单例模式)

**Commit**: `91fd7ab` - 重构: 为 LangGraph 节点注入依赖

---

## 2. 没有条件边

**状态**: 🔄 进行中

**问题描述**:
- graph 是纯线性的：`retrieve → generate → decide → END`
- `decide_node` 设置了 `needs_clarification` 但无法真正分支
- 无法实现澄清流程

**解决方案**:
- 使用 `add_conditional_edges` 实现真正的条件分支
- 当置信度低时回到检索节点重新检索，或进入澄清节点

**待修改文件**:
- `graph/graph.py`
- `graph/nodes.py`
- `graph/state.py` (可能需要添加新字段)

---

## 3. LLM 实例不复用

**状态**: ⏳ 待处理

---

## 4. System prompt 重复

**状态**: ⏳ 待处理

---

## 5. Checkpointer 配置

**状态**: ⏳ 待处理

---

## 6. 错误恢复机制

**状态**: ⏳ 待处理
