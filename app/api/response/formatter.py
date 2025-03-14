"""响应格式化模块

这个模块提供了API响应的格式化功能。
"""

import time
import json
from typing import Dict, Any, List, Generator

from app.utils.logging.logger import get_logger

logger = get_logger("api.response.formatter")

class ResponseFormatter:
    """响应格式化器基类"""
    
    def format_complete_response(self, response_message: str) -> Dict[str, Any]:
        """格式化完整响应
        
        Args:
            response_message: 响应消息
            
        Returns:
            格式化的响应字典
        """
        raise NotImplementedError
    
    def format_stream_chunk(self, chunk: Dict[str, Any]) -> str:
        """格式化流式响应块
        
        Args:
            chunk: 响应块
            
        Returns:
            格式化的响应块字符串
        """
        raise NotImplementedError


class OpenAIResponseFormatter(ResponseFormatter):
    """OpenAI格式的响应格式化器"""
    
    def format_complete_response(self, response_message: str) -> Dict[str, Any]:
        """格式化完整响应为OpenAI格式
        
        Args:
            response_message: 响应消息
            
        Returns:
            格式化的响应字典
        """
        # 生成唯一ID
        response_id = f"chatcmpl-{int(time.time())}"
        
        # 估算token数量（粗略估计）
        completion_tokens = len(response_message.split()) + 1
        prompt_tokens = completion_tokens * 4  # 假设提示的token数是回复的4倍
        total_tokens = prompt_tokens + completion_tokens
        
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "smartdoc-rag-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_message},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
    
    def format_stream_chunk(self, chunk: Dict[str, Any]) -> str:
        """格式化流式响应块为OpenAI格式
        
        Args:
            chunk: 响应块
            
        Returns:
            格式化的响应块字符串
        """
        # 处理完成标记
        if chunk.get('done', False):
            return "data: [DONE]\n\n"
        
        # 处理错误
        if 'error' in chunk:
            error_json = {
                "error": {
                    "message": chunk['error'],
                    "type": "server_error",
                    "code": 500
                }
            }
            return f"data: {json.dumps(error_json)}\n\n"
        
        # 处理原始文本
        if chunk.get('raw', False):
            return f"data: {json.dumps({'content': chunk.get('content', '')})}\n\n"
        
        # 直接返回原始JSON
        return f"data: {json.dumps(chunk)}\n\n"


class StandardResponseFormatter(ResponseFormatter):
    """标准格式的响应格式化器"""
    
    def format_complete_response(self, response_message: str) -> Dict[str, Any]:
        """格式化完整响应为标准格式
        
        Args:
            response_message: 响应消息
            
        Returns:
            格式化的响应字典
        """
        return {
            "success": True,
            "message": "操作成功",
            "data": response_message,
            "timestamp": int(time.time())
        }
    
    def format_stream_chunk(self, chunk: Dict[str, Any]) -> str:
        """格式化流式响应块为标准格式
        
        Args:
            chunk: 响应块
            
        Returns:
            格式化的响应块字符串
        """
        # 处理完成标记
        if chunk.get('done', False):
            return "event: done\ndata: {}\n\n"
        
        # 处理错误
        if 'error' in chunk:
            error_json = {
                "success": False,
                "message": chunk['error'],
                "timestamp": int(time.time())
            }
            return f"event: error\ndata: {json.dumps(error_json)}\n\n"
        
        # 处理内容
        content = chunk.get('content', '')
        if not content and 'choices' in chunk:
            choices = chunk.get('choices', [])
            if choices and 'delta' in choices[0]:
                content = choices[0]['delta'].get('content', '')
        
        data = {
            "content": content,
            "timestamp": int(time.time())
        }
        
        return f"event: message\ndata: {json.dumps(data)}\n\n" 