"""LLM数据模型模块

这个模块定义了与LLM交互相关的数据模型。
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class LLMRequest:
    """LLM请求数据模型"""
    headers: Dict[str, str]
    payload: Dict[str, Any]

@dataclass
class LLMResponse:
    """LLM响应数据模型"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Dict[str, Any]] = None

@dataclass
class LLMStreamChunk:
    """LLM流式响应块数据模型"""
    content: Optional[str] = None
    done: bool = False
    error: Optional[str] = None
    raw: bool = False
    raw_data: Optional[Dict[str, Any]] = None 