"""错误处理模块

这个模块提供了统一的错误处理功能，用于处理API请求中的各种错误情况。
"""

import traceback
from flask import jsonify
from app.utils.logging.logger import get_logger

logger = get_logger("app.utils.error_handler")

class APIError(Exception):
    """API错误基类
    
    所有API相关的自定义错误都应该继承这个类。
    """
    def __init__(self, message, code=400, details=None):
        """初始化API错误
        
        Args:
            message: 错误消息
            code: HTTP状态码
            details: 错误详情
        """
        self.message = message
        self.code = code
        self.details = details
        super().__init__(self.message)

class ValidationError(APIError):
    """验证错误
    
    当请求参数验证失败时抛出。
    """
    def __init__(self, message="请求参数验证失败", details=None):
        super().__init__(message=message, code=400, details=details)

class NotFoundError(APIError):
    """资源不存在错误
    
    当请求的资源不存在时抛出。
    """
    def __init__(self, message="请求的资源不存在", details=None):
        super().__init__(message=message, code=404, details=details)

class ServiceError(APIError):
    """服务错误
    
    当服务处理过程中发生错误时抛出。
    """
    def __init__(self, message="服务处理失败", details=None):
        super().__init__(message=message, code=500, details=details)

def handle_api_error(error, message=None):
    """处理API错误
    
    Args:
        error: 错误对象
        message: 自定义错误消息，如果不提供则使用错误对象的消息
        
    Returns:
        JSON响应和HTTP状态码
    """
    # 如果是API错误，直接使用其属性
    if isinstance(error, APIError):
        response = {
            "error": error.message,
            "code": error.code
        }
        if error.details:
            response["details"] = error.details
        logger.error(f"API错误: {error.message}", exc_info=True)
        return jsonify(response), error.code
    
    # 处理其他类型的错误
    error_message = message or str(error)
    logger.error(f"未处理的错误: {error_message}", exc_info=True)
    logger.debug(traceback.format_exc())
    
    return jsonify({
        "error": error_message,
        "code": 500
    }), 500 