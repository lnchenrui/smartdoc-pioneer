"""流式响应处理模块

这个模块提供了处理流式响应的功能。
"""

from typing import Generator, Dict, Any
from flask import Response

from app.utils.logging.logger import get_logger
from app.api.response.formatter import ResponseFormatter, OpenAIResponseFormatter

logger = get_logger("api.response.stream")

class StreamResponseHandler:
    """流式响应处理器，处理流式响应"""
    
    def __init__(self, formatter: ResponseFormatter = None):
        """初始化流式响应处理器
        
        Args:
            formatter: 响应格式化器，如果为None则使用OpenAI格式
        """
        self.formatter = formatter or OpenAIResponseFormatter()
        logger.info(f"流式响应处理器初始化完成，使用格式化器: {self.formatter.__class__.__name__}")
    
    def create_response(self, generator: Generator[Dict[str, Any], None, None]) -> Response:
        """创建流式响应
        
        Args:
            generator: 响应生成器
            
        Returns:
            Flask响应对象
        """
        logger.info("创建流式响应")
        
        def stream():
            try:
                for chunk in generator:
                    formatted_chunk = self.formatter.format_stream_chunk(chunk)
                    yield formatted_chunk
            except Exception as e:
                logger.error(f"流式响应处理出错: {str(e)}")
                error_chunk = {'error': str(e)}
                yield self.formatter.format_stream_chunk(error_chunk)
                # 确保发送完成标记
                yield self.formatter.format_stream_chunk({'done': True})
        
        return Response(
            stream(),
            mimetype="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'  # 禁用Nginx缓冲
            }
        ) 