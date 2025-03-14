"""LLM客户端模块

这个模块提供了与大语言模型交互的客户端功能。
"""

import time
import traceback
import requests
from typing import List, Dict, Any, Optional, Generator, Union

from app.utils.logging.logger import get_logger
from app.utils.exceptions import LLMError
from app.core.llm.models import LLMRequest

logger = get_logger("llm.client")

class LLMClient:
    """LLM客户端类，提供与大语言模型交互的接口"""
    
    def __init__(
        self,
        model: str,
        api_key: str,
        endpoint: str,
        temperature: float = 0.7,
        max_tokens: int = 10240,
        top_p: float = 0.95
    ):
        """初始化LLM客户端
        
        Args:
            model: 模型名称
            api_key: API密钥
            endpoint: API端点
            temperature: 温度参数
            max_tokens: 最大令牌数
            top_p: top-p参数
        """
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        
        # 构建API URL
        self.url = f"{self.endpoint}"
        if not self.url.endswith('/chat/completions'):
            self.url = f"{self.endpoint}/chat/completions"
        
        logger.info(f"LLM客户端初始化完成，模型: {self.model}, 端点: {self.endpoint}")
    
    def _prepare_request(self, messages: List[Dict[str, str]], stream: bool = True) -> LLMRequest:
        """准备LLM请求
        
        Args:
            messages: 消息列表
            stream: 是否使用流式响应
            
        Returns:
            LLM请求对象
        """
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "model": self.model,
            "stream": stream
        }
        
        logger.debug(f"准备LLM请求参数: stream={stream}, 消息数量={len(messages)}")
        return LLMRequest(headers=headers, payload=payload)
    
    def _handle_error(self, error: Exception) -> str:
        """处理错误
        
        Args:
            error: 异常对象
            
        Returns:
            错误消息
        """
        error_msg = f"LLM服务调用失败 - {str(error)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return error_msg
    
    def generate_response(self, messages: List[Dict[str, str]], stream: bool = True) -> requests.Response:
        """生成响应
        
        Args:
            messages: 消息列表
            stream: 是否使用流式响应
            
        Returns:
            响应对象
            
        Raises:
            LLMError: LLM服务调用失败时抛出
        """
        request = self._prepare_request(messages, stream)
        
        logger.info(f"LLM请求: 模型={self.model}, 流式={stream}, 消息数={len(messages)}")
        logger.debug(f"LLM请求消息: {messages}")
        start_time = time.time()
        
        try:
            logger.info(f"发送请求到LLM服务，请求类型: {'流式' if stream else '非流式'}")
            response = requests.post(
                self.url, 
                headers=request.headers, 
                json=request.payload, 
                stream=stream
            )
            elapsed = time.time() - start_time
            logger.info(f"收到LLM服务响应，状态码: {response.status_code}, 耗时: {elapsed:.2f}秒")
            
            if response.status_code != 200:
                error_msg = f"LLM服务响应错误，状态码: {response.status_code}, 响应: {response.text}"
                logger.error(error_msg)
                raise LLMError(error_msg)
                
            return response
        except Exception as e:
            error_msg = self._handle_error(e)
            raise LLMError(error_msg) 