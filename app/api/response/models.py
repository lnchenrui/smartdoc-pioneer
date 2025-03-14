"""响应数据模型模块

这个模块定义了API响应相关的数据模型。
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class APIResponse:
    """API响应基础数据模型"""
    success: bool
    message: str
    timestamp: int
    data: Optional[Any] = None
    code: Optional[int] = None
    errors: Optional[List[str]] = None

@dataclass
class ChatCompletionResponse:
    """聊天补全响应数据模型"""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

@dataclass
class StreamChunk:
    """流式响应块数据模型"""
    content: Optional[str] = None
    done: bool = False
    error: Optional[str] = None
    raw: bool = False
    raw_data: Optional[Dict[str, Any]] = None 