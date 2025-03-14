"""LLM 模型模块

这个模块提供了与大语言模型交互的功能，使用 Langchain 的 LLM 和 ChatModel。
"""

from typing import List, Dict, Any, Optional, Union, Generator
import time

from langchain_openai import ChatOpenAI, OpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain_core.language_models import BaseLLM, BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage
)
from langchain_core.outputs import LLMResult, ChatGeneration, Generation

from app.utils.logging.logger import get_logger
from app.utils.error_handler import LLMError

logger = get_logger("core.llms")

class LLMService:
    """LLM 服务类，提供与大语言模型交互的功能"""
    
    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        streaming: bool = False,
        **kwargs
    ):
        """初始化 LLM 服务
        
        Args:
            provider: 模型提供商，支持 "openai", "ollama"
            model_name: 模型名称
            api_key: API 密钥
            api_base: API 基础 URL
            temperature: 温度参数
            max_tokens: 最大令牌数
            streaming: 是否启用流式输出
            **kwargs: 其他参数
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming = streaming
        
        try:
            # 根据提供商创建模型
            if provider.lower() == "openai":
                # 判断是否是聊天模型
                if model_name.startswith(("gpt-", "ft:gpt-")):
                    self.llm = ChatOpenAI(
                        model=model_name,
                        openai_api_key=api_key,
                        openai_api_base=api_base,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        streaming=streaming,
                        **kwargs
                    )
                    self.is_chat_model = True
                else:
                    self.llm = OpenAI(
                        model=model_name,
                        openai_api_key=api_key,
                        openai_api_base=api_base,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        streaming=streaming,
                        **kwargs
                    )
                    self.is_chat_model = False
            
            elif provider.lower() == "ollama":
                # 判断是否是聊天模型
                if kwargs.get("format") == "json" or model_name in ["llama3", "mistral", "gemma"]:
                    self.llm = ChatOllama(
                        model=model_name,
                        base_url=api_base,
                        temperature=temperature,
                        **kwargs
                    )
                    self.is_chat_model = True
                else:
                    self.llm = Ollama(
                        model=model_name,
                        base_url=api_base,
                        temperature=temperature,
                        **kwargs
                    )
                    self.is_chat_model = False
            
            else:
                raise ValueError(f"不支持的模型提供商: {provider}")
            
            logger.info(f"LLM 服务初始化完成，提供商: {provider}, 模型: {model_name}, 是否聊天模型: {self.is_chat_model}")
        
        except Exception as e:
            error_msg = f"初始化 LLM 服务失败: {str(e)}"
            logger.error(error_msg)
            raise LLMError(error_msg)
    
    def _convert_to_messages(self, messages: List[Dict[str, str]]) -> List[BaseMessage]:
        """将消息字典列表转换为 Langchain 消息对象列表
        
        Args:
            messages: 消息字典列表，每个字典包含 "role" 和 "content" 字段
            
        Returns:
            Langchain 消息对象列表
        """
        langchain_messages = []
        
        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "")
            
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                logger.warning(f"未知的消息角色: {role}，将作为用户消息处理")
                langchain_messages.append(HumanMessage(content=content))
        
        return langchain_messages
    
    def generate_text(self, prompt: str) -> str:
        """生成文本
        
        Args:
            prompt: 提示文本
            
        Returns:
            生成的文本
            
        Raises:
            LLMError: 生成文本失败时抛出
        """
        try:
            start_time = time.time()
            
            # 如果是聊天模型，需要将提示转换为消息
            if self.is_chat_model:
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                result = response.content
            else:
                result = self.llm.invoke(prompt)
            
            elapsed = time.time() - start_time
            logger.info(f"生成文本完成，提示长度: {len(prompt)}, 结果长度: {len(result)}, 耗时: {elapsed:.2f}秒")
            
            return result
        
        except Exception as e:
            error_msg = f"生成文本失败: {str(e)}"
            logger.error(error_msg)
            raise LLMError(error_msg)
    
    def generate_chat_response(self, messages: List[Dict[str, str]]) -> str:
        """生成聊天响应
        
        Args:
            messages: 消息列表，每个消息是一个包含 "role" 和 "content" 字段的字典
            
        Returns:
            生成的响应文本
            
        Raises:
            LLMError: 生成响应失败时抛出
        """
        try:
            start_time = time.time()
            
            # 将消息字典列表转换为 Langchain 消息对象列表
            langchain_messages = self._convert_to_messages(messages)
            
            # 生成响应
            response = self.llm.invoke(langchain_messages)
            
            # 提取响应内容
            if hasattr(response, "content"):
                result = response.content
            else:
                result = str(response)
            
            elapsed = time.time() - start_time
            logger.info(f"生成聊天响应完成，消息数量: {len(messages)}, 结果长度: {len(result)}, 耗时: {elapsed:.2f}秒")
            
            return result
        
        except Exception as e:
            error_msg = f"生成聊天响应失败: {str(e)}"
            logger.error(error_msg)
            raise LLMError(error_msg)
    
    def stream_chat_response(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """流式生成聊天响应
        
        Args:
            messages: 消息列表，每个消息是一个包含 "role" 和 "content" 字段的字典
            
        Returns:
            生成的响应文本的生成器
            
        Raises:
            LLMError: 生成响应失败时抛出
        """
        try:
            # 确保模型支持流式输出
            if not self.streaming:
                logger.warning("模型未启用流式输出，将使用非流式生成")
                result = self.generate_chat_response(messages)
                yield result
                return
            
            # 将消息字典列表转换为 Langchain 消息对象列表
            langchain_messages = self._convert_to_messages(messages)
            
            # 流式生成响应
            for chunk in self.llm.stream(langchain_messages):
                if hasattr(chunk, "content"):
                    yield chunk.content
                else:
                    yield str(chunk)
        
        except Exception as e:
            error_msg = f"流式生成聊天响应失败: {str(e)}"
            logger.error(error_msg)
            raise LLMError(error_msg)
    
    def get_llm(self) -> Union[BaseLLM, BaseChatModel]:
        """获取 LLM 模型实例
        
        Returns:
            LLM 模型实例
        """
        return self.llm 