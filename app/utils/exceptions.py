"""异常模块

这个模块定义了应用的自定义异常类。
"""

class AppError(Exception):
    """应用基础异常类"""
    
    def __init__(self, message: str, code: int = 500):
        """初始化异常
        
        Args:
            message: 错误消息
            code: 错误代码
        """
        self.message = message
        self.code = code
        super().__init__(self.message)


class ConfigError(AppError):
    """配置错误异常"""
    
    def __init__(self, message: str, code: int = 500):
        super().__init__(f"配置错误: {message}", code)


class LLMError(AppError):
    """LLM错误异常"""
    
    def __init__(self, message: str, code: int = 500):
        super().__init__(f"LLM错误: {message}", code)


class DocumentError(AppError):
    """文档错误异常"""
    
    def __init__(self, message: str, code: int = 500):
        super().__init__(f"文档错误: {message}", code)


class APIError(AppError):
    """API错误异常"""
    
    def __init__(self, message: str, code: int = 400):
        super().__init__(f"API错误: {message}", code)


class ValidationError(APIError):
    """验证错误异常"""
    
    def __init__(self, message: str, code: int = 400):
        super().__init__(f"验证错误: {message}", code)


class ServiceError(AppError):
    """服务错误异常"""
    
    def __init__(self, message: str, code: int = 500):
        super().__init__(f"服务错误: {message}", code) 