"""聊天API路由模块

这个模块提供了聊天相关的API路由。
"""

from flask import Blueprint, request, jsonify, current_app, Response
from app.services.chat import ChatService
from app.utils.error_handler import handle_api_error, ValidationError
from app.utils.logging.logger import get_logger

logger = get_logger("app.api.routes.chat")
chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/local', methods=['POST'])
def chat_local():
    """本地聊天API
    
    使用本地文档作为上下文进行聊天。
    
    请求格式:
    {
        "messages": [
            {"role": "user", "content": "用户消息"},
            {"role": "assistant", "content": "助手回复"}
        ],
        "stream": true/false
    }
    
    响应格式:
    - 非流式: {"response": "助手回复"}
    - 流式: 服务器发送事件(SSE)格式的数据流
    
    Returns:
        JSON响应或SSE流
    """
    try:
        # 解析请求
        data = request.json
        if not data:
            raise ValidationError("请求体不能为空")
        
        messages = data.get('messages')
        stream = data.get('stream', False)
        
        if not messages or not isinstance(messages, list):
            raise ValidationError("消息格式不正确")
        
        # 记录请求
        logger.info(f"收到聊天请求: {len(messages)}条消息, stream={stream}")
        
        # 获取聊天服务
        chat_service = ChatService()
        
        # 处理聊天请求
        if stream:
            # 流式响应
            response = chat_service.chat_stream(messages)
            return Response(
                response,
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )
        else:
            # 非流式响应
            response = chat_service.chat(messages)
            return jsonify({"response": response})
            
    except Exception as e:
        return handle_api_error(e)

@chat_bp.route('/rag', methods=['POST'])
def chat_rag():
    """RAG聊天API
    
    使用检索增强生成(RAG)进行聊天。
    
    请求格式:
    {
        "messages": [
            {"role": "user", "content": "用户消息"},
            {"role": "assistant", "content": "助手回复"}
        ],
        "stream": true/false
    }
    
    响应格式:
    - 非流式: {"response": "助手回复", "sources": [...]}
    - 流式: 服务器发送事件(SSE)格式的数据流
    
    Returns:
        JSON响应或SSE流
    """
    try:
        # 解析请求
        data = request.json
        if not data:
            raise ValidationError("请求体不能为空")
        
        messages = data.get('messages')
        stream = data.get('stream', False)
        
        if not messages or not isinstance(messages, list):
            raise ValidationError("消息格式不正确")
        
        # 记录请求
        logger.info(f"收到RAG聊天请求: {len(messages)}条消息, stream={stream}")
        
        # 获取聊天服务
        chat_service = ChatService()
        
        # 处理聊天请求
        if stream:
            # 流式响应
            response = chat_service.rag_chat_stream(messages)
            return Response(
                response,
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )
        else:
            # 非流式响应
            response, sources = chat_service.rag_chat(messages)
            return jsonify({"response": response, "sources": sources})
            
    except Exception as e:
        return handle_api_error(e)

# 为了向后兼容，保留旧的API路由
@chat_bp.route('/file_completions', methods=['POST'])
def chat_file_completions():
    """本地文档聊天API（旧版）
    
    为了向后兼容而保留，建议使用 /local 端点。
    """
    logger.warning("使用了旧版API路由 /chat/file_completions，建议使用 /chat/local")
    return chat_local()

@chat_bp.route('/completions', methods=['POST'])
def chat_completions():
    """兼容OpenAI格式的聊天API
    
    这是为了兼容OpenAI API格式的聊天接口。
    
    请求格式:
    {
        "model": "模型名称",
        "messages": [
            {"role": "user", "content": "用户消息"},
            {"role": "assistant", "content": "助手回复"}
        ],
        "stream": true/false
    }
    
    响应格式:
    - 非流式: OpenAI兼容的JSON响应
    - 流式: OpenAI兼容的SSE流
    
    Returns:
        JSON响应或SSE流
    """
    try:
        # 解析请求
        data = request.json
        if not data:
            raise ValidationError("请求体不能为空")
        
        messages = data.get('messages')
        stream = data.get('stream', False)
        
        if not messages or not isinstance(messages, list):
            raise ValidationError("消息格式不正确")
        
        # 记录请求
        logger.info(f"收到兼容OpenAI格式的聊天请求: {len(messages)}条消息, stream={stream}")
        
        # 转发到本地聊天API
        return chat_local()
    except Exception as e:
        return handle_api_error(e) 