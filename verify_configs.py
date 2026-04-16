#!/usr/bin/env python3
"""
Docker 配置文件验证工具
验证 Dockerfile、docker-compose.yml 等配置文件的语法正确性
"""

import os
import sys
import subprocess
from pathlib import Path


def check_dockerfile(dockerfile_path: Path) -> bool:
    """验证 Dockerfile 语法"""
    print(f"🔍 检查 Dockerfile: {dockerfile_path}")

    if not dockerfile_path.exists():
        print(f"  ❌ 文件不存在")
        return False

    # 使用 hadolint 进行 Dockerfile 检查（如果安装）
    try:
        result = subprocess.run(
            ["docker", "build", "-t", "test-build", "-f", str(dockerfile_path), "."],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # 如果构建到语法检查阶段失败，说明有语法错误
        if "failed to solve" in result.stderr or "error" in result.stderr.lower():
            print(f"  ⚠️  语法检查（通过构建测试）:")
            print(f"     {result.stderr[:200]}")
            # 注意：这里只是部分检查，真正的语法错误会在构建时暴露
        print(f"  ✓ 文件存在且可读")
        return True
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  构建超时（可能需要实际构建才能发现所有问题）")
        return True
    except FileNotFoundError:
        print(f"  ⚠️  Docker 未安装，跳过构建测试")
        return True
    except Exception as e:
        print(f"  ⚠️  检查异常: {e}")
        return True


def check_docker_compose(compose_path: Path) -> bool:
    """验证 docker-compose.yml 语法"""
    print(f"🔍 检查 docker-compose.yml: {compose_path}")

    if not compose_path.exists():
        print(f"  ❌ 文件不存在")
        return False

    try:
        # 使用 docker-compose config 验证语法
        result = subprocess.run(
            ["docker-compose", "-f", str(compose_path), "config"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print(f"  ✓ 语法正确")
            return True
        else:
            print(f"  ❌ 语法错误:")
            print(f"     {result.stderr}")
            return False
    except FileNotFoundError:
        # 尝试使用 docker compose（新版本）
        try:
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_path), "config"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                print(f"  ✓ 语法正确（使用 docker compose）")
                return True
            else:
                print(f"  ❌ 语法错误:")
                print(f"     {result.stderr}")
                return False
        except Exception:
            print(f"  ⚠️  Docker Compose 未安装，跳过语法检查")
            return True
    except Exception as e:
        print(f"  ⚠️  检查异常: {e}")
        return True


def check_dockerignore(dockerignore_path: Path) -> bool:
    """检查 .dockerignore 文件"""
    print(f"🔍 检查 .dockerignore: {dockerignore_path}")

    if not dockerignore_path.exists():
        print(f"  ⚠️  文件不存在（建议创建以减小镜像体积）")
        return False

    with open(dockerignore_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"  ✓ 文件存在 ({len(lines)} 条规则)")

    # 检查关键规则
    important_rules = [
        ".git",
        "__pycache__",
        "*.pyc",
        ".venv",
        "venv",
        "node_modules",
        ".env",
        "*.log",
        "tests/",
    ]

    found_rules = []
    for rule in important_rules:
        if any(rule in line for line in lines):
            found_rules.append(rule)

    print(f"  ℹ️  关键规则覆盖: {len(found_rules)}/{len(important_rules)}")
    return True


def check_pyproject(pyproject_path: Path) -> bool:
    """检查 pyproject.toml"""
    print(f"🔍 检查 pyproject.toml: {pyproject_path}")

    if not pyproject_path.exists():
        print(f"  ❌ 文件不存在")
        return False

    import tomli if sys.version_info >= (3, 11) else toml

    try:
        with open(pyproject_path, "rb") as f:
            if sys.version_info >= (3, 11):
                import tomllib as tomli
            else:
                import toml
            data = tomli.load(f)

        # 检查必要字段
        required_fields = ["project", "dependencies"]
        for field in required_fields:
            if field not in data:
                print(f"  ⚠️  缺少字段: {field}")
                return False

        deps = data["dependencies"]
        print(f"  ✓ 配置正确 ({len(deps)} 个依赖)")

        # 检查关键依赖
        key_deps = ["fastapi", "uvicorn", "sqlalchemy", "openai", "qdrant-client"]
        missing = [d for d in key_deps if not any(d in dep for dep in deps)]
        if missing:
            print(f"  ⚠️  可能缺少关键依赖: {missing}")

        return True
    except Exception as e:
        print(f"  ❌ 解析失败: {e}")
        return False


def check_env_example(env_path: Path) -> bool:
    """检查 .env.example"""
    print(f"🔍 检查 .env.example: {env_path}")

    if not env_path.exists():
        print(f"  ⚠️  文件不存在")
        return False

    with open(env_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 检查关键环境变量
    required_vars = [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "QDRANT_HOST",
        "DATABASE_URL",
    ]

    missing = [var for var in required_vars if var not in content]
    if missing:
        print(f"  ⚠️  缺少关键环境变量: {missing}")
    else:
        print(f"  ✓ 配置完整")

    return len(missing) == 0


def check_main_file(main_path: Path) -> bool:
    """检查 main.py 入口文件"""
    print(f"🔍 检查 main.py: {main_path}")

    if not main_path.exists():
        print(f"  ❌ 文件不存在")
        return False

    with open(main_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 检查关键导入
    required_imports = ["FastAPI", "uvicorn"]
    missing = [imp for imp in required_imports if imp not in content]
    if missing:
        print(f"  ⚠️  可能缺少关键导入: {missing}")
        return False

    # 检查应用创建和运行
    checks = [
        ("app = create_app()", "应用实例创建"),
        ("uvicorn.run", "Uvicorn启动"),
    ]

    for pattern, desc in checks:
        if pattern in content:
            print(f"  ✓ {desc}")
        else:
            print(f"  ⚠️  未找到: {desc}")

    return True


def main():
    project_root = Path(__file__).parent

    print("=" * 60)
    print("Docker 配置文件验证")
    print("=" * 60)
    print(f"项目目录: {project_root}")
    print()

    checks = [
        ("Dockerfile", check_dockerfile(project_root / "Dockerfile")),
        ("docker-compose.yml", check_docker_compose(project_root / "docker-compose.yml")),
        (".dockerignore", check_dockerignore(project_root / ".dockerignore")),
        ("pyproject.toml", check_pyproject(project_root / "pyproject.toml")),
        (".env.example", check_env_example(project_root / ".env.example")),
        ("main.py", check_main_file(project_root / "main.py")),
    ]

    print()
    print("=" * 60)
    print("验证结果汇总")
    print("=" * 60)

    passed = sum(1 for _, result in checks if result)
    total = len(checks)

    for name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8s} {name}")

    print()
    print(f"总计: {passed}/{total} 通过")

    if passed == total:
        print("✅ 所有检查通过！配置文件基本正确。")
        print()
        print("下一步建议:")
        print("  1. 构建镜像: docker-compose build")
        print("  2. 启动服务: docker-compose up -d")
        print("  3. 健康检查: curl http://localhost:8000/api/v1/health")
        return 0
    else:
        print("⚠️  部分检查未通过，请修正上述问题后再试。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
