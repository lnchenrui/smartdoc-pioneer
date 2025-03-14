"""聊天服务模块

这个模块提供了聊天相关的业务逻辑，使用 Langchain 的 LCEL 和聊天功能。
"""

from typing import List, Dict, Any, Generator, Optional, Tuple, Union
import json
import time

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.utils.logging.logger import get_logger
from app.utils.error_handler import ValidationError, ServiceError
from app.core.components import (
    document_processor,
    llm_client,
    llm_response_handler,
    rag_service
)

logger = get_logger("services.chat")


class ChatService:
    """聊天服务类，处理聊天相关的业务逻辑"""
    
    def __init__(
        self,
        llm_service,
        document_loader,
        rag_service,
        prompt_service
    ):
        """初始化聊天服务
        
        Args:
            llm_service: LLM 服务实例
            document_loader: 文档加载器实例
            rag_service: RAG 服务实例
            prompt_service: 提示模板服务实例
        """
        self.llm_service = llm_service
        self.document_loader = document_loader
        self.rag_service = rag_service
        self.prompt_service = prompt_service
        
        logger.info("聊天服务初始化完成")
    
    def validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """验证消息格式
        
        Args:
            messages: 消息列表
            
        Raises:
            ValidationError: 消息格式无效时抛出
        """
        if not messages or not isinstance(messages, list):
            raise ValidationError("消息格式无效")
        
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValidationError("消息格式无效，每条消息必须包含 role 和 content 字段")
            
            if msg["role"] not in ["system", "user", "assistant"]:
                raise ValidationError(f"消息角色无效: {msg['role']}，必须是 system、user 或 assistant")
    
    def extract_user_message(self, messages: List[Dict[str, str]]) -> str:
        """从消息列表中提取最后一条用户消息
        
        Args:
            messages: 消息列表
            
        Returns:
            用户消息内容
            
        Raises:
            ValidationError: 未找到用户消息时抛出
        """
        user_message = None
        for message in reversed(messages):
            if message.get('role') == 'user':
                user_message = message.get('content', '')
                break
        
        if not user_message:
            raise ValidationError("未找到用户消息")
        
        return user_message
    
    def process_local_chat_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """处理本地聊天消息
        
        Args:
            messages: 原始消息列表
            
        Returns:
            处理后的消息列表
        """
        # 暂时不加载本地文档内容作为上下文
        context = "目前没有可用的本地文档内容。"
        
        # 构建完整的消息列表，保留多轮对话历史
        processed_messages = []
        
        # 添加系统消息
        system_message = {
            "role": "system",
            "content": "你是一个智能助手，可以回答用户关于文档内容的问题。请基于提供的文档内容回答问题，如果无法从文档中找到答案，请明确告知用户。"
        }
        
        # 检查是否已有系统消息
        has_system_message = False
        for msg in messages:
            if msg.get("role") == "system":
                has_system_message = True
                # 将文档内容添加到系统消息中
                msg["content"] = f"{msg['content']}\n\n文档内容：\n{context}"
                processed_messages.append(msg)
                break
        
        # 如果没有系统消息，添加一个
        if not has_system_message:
            system_message["content"] = f"{system_message['content']}\n\n文档内容：\n{context}"
            processed_messages.append(system_message)
        
        # 添加用户的所有消息，保持对话历史
        for msg in messages:
            if msg["role"] == "system" and has_system_message:
                # 跳过系统消息，因为我们已经处理过了
                continue
            processed_messages.append(msg)
        
        logger.debug(f"处理后的消息数量: {len(processed_messages)}")
        return processed_messages
    
    def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """普通聊天
        
        使用本地文档作为上下文进行聊天。
        
        Args:
            messages: 消息列表，包含用户和助手的对话历史
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成令牌数
            
        Returns:
            助手的回复，包含id、model、content等信息
        """
        try:
            # 验证消息格式
            self.validate_messages(messages)
            
            # 处理消息
            processed_messages = self.process_local_chat_messages(messages)
            
            # 转换为 Langchain 消息格式
            langchain_messages = []
            for msg in processed_messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "system":
                    langchain_messages.append(self.llm_service.create_system_message(content))
                elif role == "user":
                    langchain_messages.append(self.llm_service.create_human_message(content))
                elif role == "assistant":
                    langchain_messages.append(self.llm_service.create_ai_message(content))
            
            # 生成回复
            ai_message = self.llm_service.chat(
                langchain_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 构建响应
            response = {
                "id": f"chatcmpl-{int(time.time())}",
                "model": getattr(self.llm_service.get_llm(), "model_name", "unknown"),
                "content": ai_message.content,
                "usage": {
                    "prompt_tokens": 0,  # 暂时无法获取
                    "completion_tokens": 0,  # 暂时无法获取
                    "total_tokens": 0  # 暂时无法获取
                }
            }
            
            logger.info("生成聊天回复成功")
            return response
            
        except Exception as e:
            logger.error(f"生成聊天回复失败: {str(e)}")
            raise ServiceError(f"聊天服务错误: {str(e)}")
    
    def chat_stream(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Generator[str, None, None]:
        """流式聊天
        
        使用本地文档作为上下文进行流式聊天。
        
        Args:
            messages: 消息列表，包含用户和助手的对话历史
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成令牌数
            
        Returns:
            助手回复的生成器
        """
        try:
            # 验证消息格式
            self.validate_messages(messages)
            
            # 处理消息
            processed_messages = self.process_local_chat_messages(messages)
            
            # 转换为 Langchain 消息格式
            langchain_messages = []
            for msg in processed_messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "system":
                    langchain_messages.append(self.llm_service.create_system_message(content))
                elif role == "user":
                    langchain_messages.append(self.llm_service.create_human_message(content))
                elif role == "assistant":
                    langchain_messages.append(self.llm_service.create_ai_message(content))
            
            # 生成流式回复
            for chunk in self.llm_service.chat_stream(
                langchain_messages,
                temperature=temperature,
                max_tokens=max_tokens
            ):
                # 如果是字符串，直接返回
                if isinstance(chunk, str):
                    yield chunk
                # 如果是消息对象，返回内容
                elif hasattr(chunk, "content"):
                    yield chunk.content
                # 其他情况，尝试转换为字符串
                else:
                    yield str(chunk)
                
        except Exception as e:
            logger.error(f"生成流式聊天回复失败: {str(e)}")
            raise ServiceError(f"聊天服务错误: {str(e)}")
    
    def rag_chat(self, messages: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, Any]]]:
        """RAG 聊天
        
        使用检索增强生成 (RAG) 进行聊天。
        
        Args:
            messages: 消息列表，包含用户和助手的对话历史
            
        Returns:
            (助手的回复, 引用的来源列表)
        """
        try:
            # 验证消息格式
            self.validate_messages(messages)
            
            # 获取最后一条用户消息
            query = self.extract_user_message(messages)
            
            # 使用 RAG 服务处理查询
            response, sources = self.rag_service.process_with_sources(query)
            
            logger.info("生成 RAG 聊天回复成功")
            return response, sources
            
        except Exception as e:
            logger.error(f"生成 RAG 聊天回复失败: {str(e)}")
            raise ServiceError(f"RAG 聊天服务错误: {str(e)}")
    
    def rag_chat_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """RAG 流式聊天
        
        使用检索增强生成 (RAG) 进行流式聊天。
        
        Args:
            messages: 消息列表，包含用户和助手的对话历史
            
        Returns:
            助手回复的生成器
        """
        try:
            # 验证消息格式
            self.validate_messages(messages)
            
            # 获取最后一条用户消息
            query = self.extract_user_message(messages)
            
            # 使用 RAG 服务处理流式查询
            for chunk in self.rag_service.process_stream(query):
                yield chunk
                
        except Exception as e:
            logger.error(f"生成 RAG 流式聊天回复失败: {str(e)}")
            raise ServiceError(f"RAG 聊天服务错误: {str(e)}")
    
    def format_sse_message(self, text: str, event_type: str = "message") -> str:
        """格式化服务器发送事件 (SSE) 消息
        
        Args:
            text: 消息文本
            event_type: 事件类型
            
        Returns:
            格式化的 SSE 消息
        """
        return f"event: {event_type}\ndata: {text}\n\n"
    
    def format_openai_sse_chunk(self, chunk: str, message_id: str, model: str) -> str:
        """格式化 OpenAI 兼容的 SSE 块
        
        Args:
            chunk: 消息块
            message_id: 消息 ID
            model: 模型名称
            
        Returns:
            格式化的 OpenAI 兼容 SSE 块
        """
        data = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": chunk
                    },
                    "finish_reason": None
                }
            ]
        }
        return f"data: {json.dumps(data)}\n\n"


# 创建全局服务实例
chat_service = ChatService(
    llm_service=llm_client,
    document_loader=document_processor,
    rag_service=rag_service,
    prompt_service=None  # 暂时设为None，后续可以添加
) 