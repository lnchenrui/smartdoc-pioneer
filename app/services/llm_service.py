"""LLM服务模块

这个模块提供了大语言模型相关的服务功能，使用 Langchain 的最新LLM功能。
"""

from typing import List, Dict, Any, Optional, Union, Iterator, Callable
import time

from langchain_core.language_models import BaseLLM, BaseLanguageModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.callbacks import CallbackManager
from langchain_core.outputs import LLMResult, Generation

from app.utils.logging.logger import get_logger
from app.utils.error_handler import ServiceError, ValidationError

logger = get_logger("services.llm")

class LLMService:
    """LLM服务类，提供大语言模型功能"""
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        config: Dict[str, Any],
        callback_manager: Optional[CallbackManager] = None
    ):
        """初始化LLM服务
        
        Args:
            llm: 大语言模型实例
            config: 配置字典
            callback_manager: 回调管理器
        """
        self.llm = llm
        self.config = config
        self.callback_manager = callback_manager
        
        # 获取LLM配置
        self.default_temperature = config.get('llm', {}).get('temperature', 0.7)
        self.default_max_tokens = config.get('llm', {}).get('max_tokens', 1024)
        self.default_system_message = config.get('llm', {}).get('system_message', "你是一个有用的AI助手。")
        
        logger.debug("LLM服务初始化完成")
    
    def get_llm(self) -> BaseLanguageModel:
        """获取LLM实例
        
        Returns:
            LLM实例
        """
        return self.llm
    
    def generate(
        self, 
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None
    ) -> str:
        """生成文本
        
        Args:
            prompt: 提示文本
            temperature: 温度参数
            max_tokens: 最大生成令牌数
            stop: 停止序列
            
        Returns:
            生成的文本
            
        Raises:
            ServiceError: 生成失败时抛出
        """
        try:
            if not prompt:
                raise ValidationError("提示文本不能为空")
            
            # 使用默认值如果未指定
            if temperature is None:
                temperature = self.default_temperature
            
            if max_tokens is None:
                max_tokens = self.default_max_tokens
            
            logger.info(f"生成文本: prompt='{prompt[:50]}...', temperature={temperature}, max_tokens={max_tokens}")
            
            # 创建生成参数
            params = {
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if stop:
                params["stop"] = stop
            
            # 生成文本
            start_time = time.time()
            response = self.llm.invoke(prompt, **params)
            elapsed = time.time() - start_time
            
            # 获取生成的文本
            if isinstance(response, str):
                generated_text = response
            else:
                # 处理其他类型的响应
                generated_text = str(response)
            
            logger.info(f"文本生成完成，生成了 {len(generated_text)} 个字符，耗时: {elapsed:.2f}秒")
            
            return generated_text
            
        except Exception as e:
            error_msg = f"生成文本失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def generate_stream(
        self, 
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None
    ) -> Iterator[str]:
        """流式生成文本
        
        Args:
            prompt: 提示文本
            temperature: 温度参数
            max_tokens: 最大生成令牌数
            stop: 停止序列
            
        Returns:
            生成的文本迭代器
            
        Raises:
            ServiceError: 生成失败时抛出
        """
        try:
            if not prompt:
                raise ValidationError("提示文本不能为空")
            
            # 使用默认值如果未指定
            if temperature is None:
                temperature = self.default_temperature
            
            if max_tokens is None:
                max_tokens = self.default_max_tokens
            
            logger.info(f"流式生成文本: prompt='{prompt[:50]}...', temperature={temperature}, max_tokens={max_tokens}")
            
            # 创建生成参数
            params = {
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if stop:
                params["stop"] = stop
            
            # 流式生成文本
            start_time = time.time()
            
            # 检查LLM是否支持流式生成
            if hasattr(self.llm, "stream"):
                for chunk in self.llm.stream(prompt, **params):
                    # 处理流式响应
                    if isinstance(chunk, str):
                        yield chunk
                    else:
                        # 处理其他类型的响应
                        yield str(chunk)
            else:
                # 如果不支持流式生成，退回到普通生成
                logger.warning("LLM不支持流式生成，使用普通生成")
                yield self.generate(prompt, temperature, max_tokens, stop)
            
            elapsed = time.time() - start_time
            logger.info(f"流式文本生成完成，耗时: {elapsed:.2f}秒")
            
        except Exception as e:
            error_msg = f"流式生成文本失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def chat(
        self, 
        messages: List[BaseMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None
    ) -> AIMessage:
        """聊天生成
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成令牌数
            stop: 停止序列
            
        Returns:
            AI消息
            
        Raises:
            ServiceError: 生成失败时抛出
        """
        try:
            if not messages:
                raise ValidationError("消息列表不能为空")
            
            # 使用默认值如果未指定
            if temperature is None:
                temperature = self.default_temperature
            
            if max_tokens is None:
                max_tokens = self.default_max_tokens
            
            logger.info(f"聊天生成: messages={len(messages)}, temperature={temperature}, max_tokens={max_tokens}")
            
            # 创建生成参数
            params = {
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if stop:
                params["stop"] = stop
            
            try:
                # 生成回复
                start_time = time.time()
                response = self.llm.invoke(messages, **params)
                elapsed = time.time() - start_time
                
                logger.info(f"聊天生成完成，耗时: {elapsed:.2f}秒")
                
                # 如果返回的不是AIMessage，转换为AIMessage
                if not isinstance(response, AIMessage):
                    if isinstance(response, str):
                        response = AIMessage(content=response)
                    else:
                        response = AIMessage(content=str(response))
                
                return response
            except Exception as e:
                # 如果API调用失败，返回虚拟响应
                logger.warning(f"LLM API调用失败: {str(e)}，使用虚拟响应")
                return AIMessage(content="我是一个AI助手，很抱歉，我现在无法连接到我的知识库。请稍后再试或者联系管理员检查API配置。")
            
        except Exception as e:
            error_msg = f"聊天生成失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def chat_stream(
        self, 
        messages: List[BaseMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None
    ) -> Iterator[Union[str, AIMessage]]:
        """流式聊天生成
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成令牌数
            stop: 停止序列
            
        Returns:
            生成的消息迭代器
            
        Raises:
            ServiceError: 生成失败时抛出
        """
        try:
            if not messages:
                raise ValidationError("消息列表不能为空")
            
            # 使用默认值如果未指定
            if temperature is None:
                temperature = self.default_temperature
            
            if max_tokens is None:
                max_tokens = self.default_max_tokens
            
            logger.info(f"流式聊天生成: messages={len(messages)}, temperature={temperature}, max_tokens={max_tokens}")
            
            # 创建生成参数
            params = {
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if stop:
                params["stop"] = stop
            
            # 流式生成回复
            start_time = time.time()
            
            try:
                # 检查LLM是否支持流式生成
                if hasattr(self.llm, "stream"):
                    for chunk in self.llm.stream(messages, **params):
                        yield chunk
                else:
                    # 如果不支持流式生成，退回到普通生成
                    logger.warning("LLM不支持流式聊天生成，使用普通聊天生成")
                    yield self.chat(messages, temperature, max_tokens, stop)
                
                elapsed = time.time() - start_time
                logger.info(f"流式聊天生成完成，耗时: {elapsed:.2f}秒")
            except Exception as e:
                # 如果API调用失败，返回虚拟响应
                logger.warning(f"LLM API流式调用失败: {str(e)}，使用虚拟响应")
                yield AIMessage(content="我是一个AI助手，很抱歉，我现在无法连接到我的知识库。请稍后再试或者联系管理员检查API配置。")
            
        except Exception as e:
            error_msg = f"流式聊天生成失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def create_system_message(self, content: Optional[str] = None) -> SystemMessage:
        """创建系统消息
        
        Args:
            content: 消息内容
            
        Returns:
            系统消息
        """
        if content is None:
            content = self.default_system_message
        
        return SystemMessage(content=content)
    
    def create_human_message(self, content: str) -> HumanMessage:
        """创建人类消息
        
        Args:
            content: 消息内容
            
        Returns:
            人类消息
        """
        return HumanMessage(content=content)
    
    def create_ai_message(self, content: str) -> AIMessage:
        """创建AI消息
        
        Args:
            content: 消息内容
            
        Returns:
            AI消息
        """
        return AIMessage(content=content) 