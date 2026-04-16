#!/usr/bin/env python3
# ===================
# 综合健康检查脚本
# ===================
# 用途：监控 Knowledge QA Agent 及其依赖服务的健康状态
# 功能：检查API、数据库、Qdrant向量数据库、OpenAI API连通性
#
# 使用方式：
#   python healthcheck.py
#   python healthcheck.py --json  # JSON格式输出（便于监控系统采集）
#   python healthcheck.py --all   # 完整检查（包括OpenAI API）
#
# 集成到监控系统：
#   - Prometheus + Blackbox Exporter
#   - Zabbix / Nagios / Icinga
#   - 自定义告警脚本

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Optional

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# ===================
# 配置
# ===================
# 从环境变量读取配置，提供默认值
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_HEALTH_ENDPOINT = "/api/v1/health"
API_CHAT_ENDPOINT = "/api/v1/chat"

DATABASE_URL = os.getenv("DATABASE_URL", "")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6400"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "knowledge_base")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
CHECK_OPENAI = os.getenv("CHECK_OPENAI", "false").lower() == "true"

TIMEOUT = int(os.getenv("HEALTHCHECK_TIMEOUT", "10"))  # 超时时间（秒）


# ===================
# 数据模型
# ===================
class Status(str, Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentStatus:
    """组件状态"""
    name: str
    status: Status
    message: str = ""
    latency_ms: float = 0.0
    details: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    overall_status: Status
    timestamp: float
    components: list[ComponentStatus]
    summary: dict[str, int]  # healthy, degraded, unhealthy 计数

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.overall_status.value,
            "timestamp": self.timestamp,
            "components": [c.to_dict() for c in self.components],
            "summary": self.summary,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ===================
# 健康检查器基类
# ===================
class HealthChecker:
    """健康检查器基类"""

    def __init__(self, name: str, timeout: float = TIMEOUT):
        self.name = name
        self.timeout = timeout

    async def check(self) -> ComponentStatus:
        """执行检查，子类必须重写"""
        raise NotImplementedError


# ===================
# API 健康检查
# ===================
class APIHealthChecker(HealthChecker):
    """FastAPI 应用健康检查"""

    def __init__(self, base_url: str = API_BASE_URL, timeout: float = TIMEOUT):
        super().__init__("api", timeout)
        self.base_url = base_url.rstrip("/")

    async def check(self) -> ComponentStatus:
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}{API_HEALTH_ENDPOINT}")

            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                version = data.get("version", "unknown")
                return ComponentStatus(
                    name=self.name,
                    status=Status.HEALTHY,
                    message=f"API运行正常 (版本: {version})",
                    latency_ms=latency,
                    details={"version": version, "endpoint": f"{self.base_url}{API_HEALTH_ENDPOINT}"},
                )
            else:
                return ComponentStatus(
                    name=self.name,
                    status=Status.UNHEALTHY,
                    message=f"API返回异常状态码: {response.status_code}",
                    latency_ms=latency,
                    details={"status_code": response.status_code},
                )

        except httpx.ConnectError as e:
            latency = (time.time() - start) * 1000
            return ComponentStatus(
                name=self.name,
                status=Status.UNHEALTHY,
                message=f"无法连接到API服务: {str(e)}",
                latency_ms=latency,
            )
        except httpx.TimeoutException as e:
            latency = (time.time() - start) * 1000
            return ComponentStatus(
                name=self.name,
                status=Status.UNHEALTHY,
                message=f"API请求超时: {str(e)}",
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return ComponentStatus(
                name=self.name,
                status=Status.UNHEALTHY,
                message=f"未知错误: {str(e)}",
                latency_ms=latency,
            )


# ===================
# 数据库健康检查
# ===================
class DatabaseHealthChecker(HealthChecker):
    """MySQL 数据库健康检查"""

    def __init__(self, db_url: str = DATABASE_URL, timeout: float = TIMEOUT):
        super().__init__("database", timeout)
        self.db_url = db_url

    async def check(self) -> ComponentStatus:
        start = time.time()
        try:
            if not self.db_url:
                latency = (time.time() - start) * 1000
                return ComponentStatus(
                    name=self.name,
                    status=Status.UNHEALTHY,
                    message="未配置 DATABASE_URL",
                    latency_ms=latency,
                )

            engine = create_async_engine(self.db_url, echo=False)
            try:
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                    latency = (time.time() - start) * 1000
                    return ComponentStatus(
                        name=self.name,
                        status=Status.HEALTHY,
                        message="数据库连接正常",
                        latency_ms=latency,
                    )
            finally:
                await engine.dispose()

        except Exception as e:
            latency = (time.time() - start) * 1000
            return ComponentStatus(
                name=self.name,
                status=Status.UNHEALTHY,
                message=f"数据库连接失败: {str(e)}",
                latency_ms=latency,
            )


# ===================
# Qdrant 向量数据库健康检查
# ===================
class QdrantHealthChecker(HealthChecker):
    """Qdrant 向量数据库健康检查"""

    def __init__(self, host: str = QDRANT_HOST, port: int = QDRANT_PORT, collection: str = QDRANT_COLLECTION, timeout: float = TIMEOUT):
        super().__init__("qdrant", timeout)
        self.host = host
        self.port = port
        self.collection = collection

    async def check(self) -> ComponentStatus:
        start = time.time()
        try:
            # 使用同步客户端（简单起见）
            client = QdrantClient(host=self.host, port=self.port, timeout=int(self.timeout))

            # 检查服务健康
            health = client.health()
            if health.status not in ("green", "yellow"):
                latency = (time.time() - start) * 1000
                return ComponentStatus(
                    name=self.name,
                    status=Status.DEGRADED,
                    message=f"Qdrant状态: {health.status}",
                    latency_ms=latency,
                    details={"status": health.status},
                )

            # 检查集合是否存在
            try:
                collection_info = client.get_collection(self.collection)
                latency = (time.time() - start) * 1000
                return ComponentStatus(
                    name=self.name,
                    status=Status.HEALTHY,
                    message=f"Qdrant运行正常，集合 '{self.collection}' ���在",
                    latency_ms=latency,
                    details={
                        "status": health.status,
                        "collection": self.collection,
                        "vectors_count": getattr(collection_info, "vectors_count", "N/A"),
                    },
                )
            except UnexpectedResponse as e:
                # 集合不存在不算严重错误
                latency = (time.time() - start) * 1000
                return ComponentStatus(
                    name=self.name,
                    status=Status.DEGRADED,
                    message=f"集合 '{self.collection}' 不存在（需先上传文档）",
                    latency_ms=latency,
                    details={"collection_missing": True},
                )

        except Exception as e:
            latency = (time.time() - start) * 1000
            return ComponentStatus(
                name=self.name,
                status=Status.UNHEALTHY,
                message=f"Qdrant连接失败: {str(e)}",
                latency_ms=latency,
            )


# ===================
# OpenAI API 健康检查
# ===================
class OpenAIHealthChecker(HealthChecker):
    """OpenAI API 连通性检查"""

    def __init__(self, api_key: str = OPENAI_API_KEY, base_url: str = OPENAI_BASE_URL, timeout: float = TIMEOUT):
        super().__init__("openai_api", timeout)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    async def check(self) -> ComponentStatus:
        if not self.api_key:
            return ComponentStatus(
                name=self.name,
                status=Status.UNKNOWN,
                message="未配置API密钥（OPENAI_API_KEY）",
                latency_ms=0.0,
            )

        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )

            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                model_count = len(data.get("data", []))
                return ComponentStatus(
                    name=self.name,
                    status=Status.HEALTHY,
                    message=f"OpenAI API连接正常，可用模型: {model_count}",
                    latency_ms=latency,
                    details={"models_count": model_count},
                )
            elif response.status_code == 401:
                return ComponentStatus(
                    name=self.name,
                    status=Status.UNHEALTHY,
                    message="API密钥无效或已过期",
                    latency_ms=latency,
                    details={"status_code": response.status_code},
                )
            else:
                return ComponentStatus(
                    name=self.name,
                    status=Status.UNHEALTHY,
                    message=f"API返回异常: {response.status_code}",
                    latency_ms=latency,
                    details={"status_code": response.status_code},
                )

        except httpx.ConnectError as e:
            latency = (time.time() - start) * 1000
            return ComponentStatus(
                name=self.name,
                status=Status.UNHEALTHY,
                message=f"无法连接到OpenAI API: {str(e)}",
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return ComponentStatus(
                name=self.name,
                status=Status.UNHEALTHY,
                message=f"未知错误: {str(e)}",
                latency_ms=latency,
            )


# ===================
# 集成测试（可选）
# ===================
class IntegrationHealthChecker(HealthChecker):
    """集成测试：实际调用聊天API"""

    def __init__(self, base_url: str = API_BASE_URL, timeout: float = TIMEOUT):
        super().__init__("integration", timeout)
        self.base_url = base_url.rstrip("/")

    async def check(self) -> ComponentStatus:
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}{API_CHAT_ENDPOINT}",
                    json={
                        "query": "你好",
                        "knowledge_base_id": 1,  # 假设存在ID=1的知识库
                    },
                )

            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                if answer:
                    return ComponentStatus(
                        name=self.name,
                        status=Status.HEALTHY,
                        message="端到端问答测试通过",
                        latency_ms=latency,
                        details={"answer_length": len(answer)},
                    )
                else:
                    return ComponentStatus(
                        name=self.name,
                        status=Status.DEGRADED,
                        message="API响应正常但回答为空",
                        latency_ms=latency,
                    )
            else:
                return ComponentStatus(
                    name=self.name,
                    status=Status.DEGRADED,
                    message=f"集成测试失败，状态码: {response.status_code}",
                    latency_ms=latency,
                    details={"status_code": response.status_code},
                )

        except Exception as e:
            latency = (time.time() - start) * 1000
            return ComponentStatus(
                name=self.name,
                status=Status.DEGRADED,
                message=f"集成测试异常: {str(e)}",
                latency_ms=latency,
            )


# ===================
# 健康检查总控
# ===================
class HealthCheckOrchestrator:
    """健康检查编排器"""

    def __init__(self, check_openai: bool = CHECK_OPENAI, check_integration: bool = False):
        self.checkers: list[HealthChecker] = [
            APIHealthChecker(),
            DatabaseHealthChecker(),
            QdrantHealthChecker(),
        ]
        if check_openai and OPENAI_API_KEY:
            self.checkers.append(OpenAIHealthChecker())
        if check_integration:
            self.checkers.append(IntegrationHealthChecker())

    async def run_all_checks(self) -> HealthCheckResult:
        """并发执行所有检查"""
        tasks = [checker.check() for checker in self.checkers]
        components = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        results = []
        for i, result in enumerate(components):
            if isinstance(result, Exception):
                results.append(
                    ComponentStatus(
                        name=self.checkers[i].name,
                        status=Status.UNHEALTHY,
                        message=f"检查过程异常: {str(result)}",
                        latency_ms=0.0,
                    )
                )
            else:
                results.append(result)

        # 计算总体状态
        statuses = {r.status for r in results}
        if Status.UNHEALTHY in statuses:
            overall = Status.UNHEALTHY
        elif Status.DEGRADED in statuses:
            overall = Status.DEGRADED
        elif Status.HEALTHY in statuses:
            overall = Status.HEALTHY
        else:
            overall = Status.UNKNOWN

        # 统计
        summary = {
            "healthy": sum(1 for r in results if r.status == Status.HEALTHY),
            "degraded": sum(1 for r in results if r.status == Status.DEGRADED),
            "unhealthy": sum(1 for r in results if r.status == Status.UNHEALTHY),
            "unknown": sum(1 for r in results if r.status == Status.UNKNOWN),
            "total": len(results),
        }

        return HealthCheckResult(
            overall_status=overall,
            timestamp=time.time(),
            components=results,
            summary=summary,
        )


# ===================
# 主程序
# ===================
async def main():
    parser = argparse.ArgumentParser(description="Knowledge QA Agent 健康检查")
    parser.add_argument("--json", action="store_true", help="输出JSON格式（便于机器读取）")
    parser.add_argument("--all", action="store_true", help="执行完整检查（包括集成测试）")
    parser.add_argument("--check-openai", action="store_true", help="检查OpenAI API连通性")
    parser.add_argument("--check-integration", action="store_true", help="执行集成测试")
    args = parser.parse_args()

    # 创建编排器
    orchestrator = HealthCheckOrchestrator(
        check_openai=args.check_openai or CHECK_OPENAI,
        check_integration=args.check_integration,
    )

    # 执行检查
    result = await orchestrator.run_all_checks()

    # 输出结果
    if args.json:
        print(result.to_json())
    else:
        # 人类可读格式
        print("=" * 60)
        print(f"Health Check Report - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print(f"\nOverall Status: {result.overall_status.value.upper()}")
        print(f"Timestamp: {result.timestamp}")
        print()

        for comp in result.components:
            status_color = {
                Status.HEALTHY: "\033[92m✓\033[0m",   # 绿
                Status.DEGRADED: "\033[93m⚠\033[0m",  # 黄
                Status.UNHEALTHY: "\033[91m✗\033[0m", # 红
                Status.UNKNOWN: "\033[94m?\033[0m",   # 蓝
            }.get(comp.status, "?")

            print(f"{status_color} {comp.name:20s} [{comp.status.value:10s}] {comp.message}")
            if comp.latency_ms > 0:
                print(f"   Latency: {comp.latency_ms:.2f}ms")
            if comp.details:
                for k, v in comp.details.items():
                    print(f"   {k}: {v}")

        print()
        print("Summary:")
        print(f"  ✓ Healthy:   {result.summary['healthy']}")
        print(f"  ⚠ Degraded:  {result.summary['degraded']}")
        print(f"  ✗ Unhealthy: {result.summary['unhealthy']}")
        print(f"  ? Unknown:   {result.summary['unknown']}")
        print(f"  Total:       {result.summary['total']}")
        print("=" * 60)

    # 返回适当的退出码（用于CI/CD或监控告警）
    if result.overall_status == Status.UNHEALTHY:
        sys.exit(1)  # 严重故障
    elif result.overall_status == Status.DEGRADED:
        sys.exit(2)  # 降级运行
    else:
        sys.exit(0)  # 正常


if __name__ == "__main__":
    asyncio.run(main())
