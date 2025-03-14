"""聊天API路由模块

这个模块提供了聊天相关的API路由。
"""

from typing import Dict, Any, List, Optional
import time

from flask import Blueprint, request, jsonify, current_app, Response, stream_with_context

from app.di.container import container
from app.utils.error_handler import ValidationError
from app.utils.logging.logger import get_logger

logger = get_logger("api.chat")

# 创建蓝图
chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/completions', methods=['POST'])
def chat_completions():
    """聊天完成API
    
    处理聊天请求并返回完成结果。
    
    Returns:
        JSON响应
    """
    try:
        # 获取请求数据
        data = request.get_json()
        if not data:
            raise ValidationError("请求体不能为空")
        
        # 验证必要字段
        if 'messages' not in data:
            raise ValidationError("缺少必要字段: messages")
        
        messages = data.get('messages', [])
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValidationError("messages必须是非空列表")
        
        # 获取可选参数
        temperature = data.get('temperature')
        max_tokens = data.get('max_tokens')
        stream = data.get('stream', False)
        
        # 获取聊天服务
        chat_service = container.get("chat_service")
        
        # 处理流式响应
        if stream:
            return stream_chat_response(chat_service, messages, temperature, max_tokens)
        
        # 处理普通响应
        start_time = time.time()
        response = chat_service.chat(messages, temperature, max_tokens)
        elapsed = time.time() - start_time
        
        # 构建响应
        result = {
            "status": "success",
            "data": {
                "id": response.get("id", ""),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": response.get("model", ""),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response.get("content", "")
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": response.get("usage", {})
            },
            "meta": {
                "elapsed_time": elapsed
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"聊天完成API错误: {str(e)}", exc_info=True)
        raise

@chat_bp.route('/rag', methods=['POST'])
def rag_chat():
    """RAG聊天API
    
    处理RAG聊天请求并返回结果。
    
    Returns:
        JSON响应
    """
    try:
        # 获取请求数据
        data = request.get_json()
        if not data:
            raise ValidationError("请求体不能为空")
        
        # 验证必要字段
        if 'messages' not in data:
            raise ValidationError("缺少必要字段: messages")
        
        messages = data.get('messages', [])
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValidationError("messages必须是非空列表")
        
        # 获取可选参数
        temperature = data.get('temperature')
        max_tokens = data.get('max_tokens')
        stream = data.get('stream', False)
        include_sources = data.get('include_sources', False)
        
        # 获取聊天服务
        chat_service = container.get("chat_service")
        
        # 处理流式响应
        if stream:
            return stream_rag_response(chat_service, messages, temperature, max_tokens, include_sources)
        
        # 处理普通响应
        start_time = time.time()
        
        if include_sources:
            response, sources = chat_service.rag_chat_with_sources(messages, temperature, max_tokens)
            elapsed = time.time() - start_time
            
            # 构建响应
            result = {
                "status": "success",
                "data": {
                    "id": response.get("id", ""),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": response.get("model", ""),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response.get("content", "")
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": response.get("usage", {}),
                    "sources": sources
                },
                "meta": {
                    "elapsed_time": elapsed
                }
            }
        else:
            response = chat_service.rag_chat(messages, temperature, max_tokens)
            elapsed = time.time() - start_time
            
            # 构建响应
            result = {
                "status": "success",
                "data": {
                    "id": response.get("id", ""),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": response.get("model", ""),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response.get("content", "")
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": response.get("usage", {})
                },
                "meta": {
                    "elapsed_time": elapsed
                }
            }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"RAG聊天API错误: {str(e)}", exc_info=True)
        raise

def stream_chat_response(chat_service, messages, temperature, max_tokens):
    """流式聊天响应
    
    Args:
        chat_service: 聊天服务
        messages: 消息列表
        temperature: 温度参数
        max_tokens: 最大令牌数
        
    Returns:
        流式响应
    """
    @stream_with_context
    def generate():
        try:
            for chunk in chat_service.chat_stream(messages, temperature, max_tokens):
                if isinstance(chunk, dict):
                    # 已经是格式化的SSE消息
                    yield f"data: {chunk}\n\n"
                else:
                    # 格式化为SSE消息
                    formatted_chunk = chat_service.format_sse_message(chunk)
                    yield f"data: {formatted_chunk}\n\n"
            
            # 发送结束标记
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"流式聊天生成错误: {str(e)}", exc_info=True)
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "server_error"
                }
            }
            yield f"data: {error_chunk}\n\n"
            yield "data: [DONE]\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

def stream_rag_response(chat_service, messages, temperature, max_tokens, include_sources=False):
    """流式RAG聊天响应
    
    Args:
        chat_service: 聊天服务
        messages: 消息列表
        temperature: 温度参数
        max_tokens: 最大令牌数
        include_sources: 是否包含来源
        
    Returns:
        流式响应
    """
    @stream_with_context
    def generate():
        try:
            for chunk in chat_service.rag_chat_stream(messages, temperature, max_tokens):
                if isinstance(chunk, dict):
                    # 已经是格式化的SSE消息
                    yield f"data: {chunk}\n\n"
                else:
                    # 格式化为SSE消息
                    formatted_chunk = chat_service.format_sse_message(chunk)
                    yield f"data: {formatted_chunk}\n\n"
            
            # 如果需要包含来源，在最后发送
            if include_sources:
                # 获取最后一条用户消息
                user_message = None
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        user_message = msg.get("content")
                        break
                
                if user_message:
                    # 获取相关文档
                    sources = chat_service.get_relevant_documents(user_message)
                    sources_chunk = {
                        "sources": sources
                    }
                    yield f"data: {sources_chunk}\n\n"
            
            # 发送结束标记
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"流式RAG聊天生成错误: {str(e)}", exc_info=True)
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "server_error"
                }
            }
            yield f"data: {error_chunk}\n\n"
            yield "data: [DONE]\n\n"
    
    return Response(generate(), mimetype='text/event-stream') 