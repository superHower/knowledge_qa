"""
工具函数
"""

import hashlib
import uuid
from datetime import datetime
from typing import Any


def generate_uuid() -> str:
    """生成 UUID"""
    return str(uuid.uuid4())


def generate_short_id(prefix: str = "") -> str:
    """生成短 ID"""
    uid = uuid.uuid4().hex[:8]
    return f"{prefix}{uid}" if prefix else uid


def md5_hash(text: str) -> str:
    """MD5 哈希"""
    return hashlib.md5(text.encode()).hexdigest()


def timestamp_to_iso(timestamp: int | None = None) -> str:
    """时间戳转 ISO 格式"""
    if timestamp:
        dt = datetime.fromtimestamp(timestamp)
    else:
        dt = datetime.utcnow()
    return dt.isoformat()


def iso_to_timestamp(iso_str: str) -> int:
    """ISO 格式转时间戳"""
    dt = datetime.fromisoformat(iso_str)
    return int(dt.timestamp())


def safe_get(dictionary: dict, *keys: str, default: Any = None) -> Any:
    """安全获取嵌套字典值"""
    result = dictionary
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key, default)
        else:
            return default
    return result


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """将列表分块"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """截断文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
