"""聊天服务模块

这个模块提供了聊天相关的服务功能，包括普通聊天和RAG聊天。
"""

from app.di.container import get_container
from app.utils.logging.logger import get_logger
from app.utils.error_handler import ServiceError, ValidationError

logger = get_logger("app.services.chat")

class ChatService:
    """聊天服务类
    
    提供聊天和RAG聊天功能。
    """
    
    def __init__(self, llm_service=None, document_loader=None, rag_service=None, prompt_service=None):
        """初始化聊天服务
        
        Args:
            llm_service: LLM服务实例
            document_loader: 文档加载器实例
            rag_service: RAG服务实例
            prompt_service: 提示模板服务实例
        """
        # 如果没有提供服务实例，则从容器中获取
        container = get_container()
        self.llm_client = llm_service or container.get("llm_service")
        self.document_processor = document_loader or container.get("document_loader_service")
        self.rag_service = rag_service or container.get("rag_service")
        self.prompt_service = prompt_service
        
        logger.debug("聊天服务初始化完成")
    
    def chat(self, messages):
        """普通聊天
        
        使用本地文档作为上下文进行聊天。
        
        Args:
            messages: 消息列表，包含用户和助手的对话历史
            
        Returns:
            助手的回复
        """
        try:
            # 验证消息格式
            self._validate_messages(messages)
            
            # 读取本地文档内容作为上下文
            context = self._get_local_context()
            
            # 构建处理后的消息列表
            processed_messages = self._build_messages_with_context(messages, context)
            
            # 生成回复
            response = self.llm_client.generate_response(processed_messages, stream=False)
            logger.info("生成聊天回复成功")
            
            return response
        except Exception as e:
            logger.error(f"生成聊天回复失败: {str(e)}", exc_info=True)
            raise ServiceError(f"聊天服务错误: {str(e)}")
    
    def chat_stream(self, messages):
        """流式聊天
        
        使用本地文档作为上下文进行流式聊天。
        
        Args:
            messages: 消息列表，包含用户和助手的对话历史
            
        Returns:
            助手回复的生成器
        """
        try:
            # 验证消息格式
            self._validate_messages(messages)
            
            # 读取本地文档内容作为上下文
            context = self._get_local_context()
            
            # 构建处理后的消息列表
            processed_messages = self._build_messages_with_context(messages, context)
            
            # 生成流式回复
            return self.llm_client.generate_response(processed_messages, stream=True)
        except Exception as e:
            logger.error(f"生成流式聊天回复失败: {str(e)}", exc_info=True)
            raise ServiceError(f"聊天服务错误: {str(e)}")
    
    def rag_chat(self, messages):
        """RAG聊天
        
        使用检索增强生成(RAG)进行聊天。
        
        Args:
            messages: 消息列表，包含用户和助手的对话历史
            
        Returns:
            (助手的回复, 引用的来源列表)
        """
        try:
            # 验证消息格式
            self._validate_messages(messages)
            
            # 获取最后一条用户消息
            last_user_message = self._get_last_user_message(messages)
            
            # 检索相关文档
            retrieved_docs = self.rag_service.retrieve(last_user_message)
            
            if not retrieved_docs:
                logger.warning("未检索到相关文档")
            
            # 构建上下文
            context = self._build_rag_context(retrieved_docs)
            
            # 构建处理后的消息列表
            processed_messages = self._build_messages_with_context(messages, context)
            
            # 生成回复
            response = self.llm_client.generate_response(processed_messages, stream=False)
            
            # 提取来源信息
            sources = self._extract_sources(retrieved_docs)
            
            logger.info("生成RAG聊天回复成功")
            return response, sources
        except Exception as e:
            logger.error(f"生成RAG聊天回复失败: {str(e)}", exc_info=True)
            raise ServiceError(f"RAG聊天服务错误: {str(e)}")
    
    def rag_chat_stream(self, messages):
        """RAG流式聊天
        
        使用检索增强生成(RAG)进行流式聊天。
        
        Args:
            messages: 消息列表，包含用户和助手的对话历史
            
        Returns:
            助手回复的生成器
        """
        try:
            # 验证消息格式
            self._validate_messages(messages)
            
            # 获取最后一条用户消息
            last_user_message = self._get_last_user_message(messages)
            
            # 检索相关文档
            retrieved_docs = self.rag_service.retrieve(last_user_message)
            
            if not retrieved_docs:
                logger.warning("未检索到相关文档")
            
            # 构建上下文
            context = self._build_rag_context(retrieved_docs)
            
            # 构建处理后的消息列表
            processed_messages = self._build_messages_with_context(messages, context)
            
            # 生成流式回复
            return self.llm_client.generate_response(processed_messages, stream=True)
        except Exception as e:
            logger.error(f"生成RAG流式聊天回复失败: {str(e)}", exc_info=True)
            raise ServiceError(f"RAG聊天服务错误: {str(e)}")
    
    def _validate_messages(self, messages):
        """验证消息格式
        
        Args:
            messages: 消息列表
            
        Raises:
            ValidationError: 如果消息格式不正确
        """
        if not messages:
            raise ValidationError("消息列表不能为空")
        
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValidationError("消息必须是字典格式")
            
            if 'role' not in msg or 'content' not in msg:
                raise ValidationError("消息必须包含role和content字段")
            
            if msg['role'] not in ['system', 'user', 'assistant']:
                raise ValidationError("消息role必须是system、user或assistant")
    
    def _get_local_context(self):
        """获取本地文档上下文
        
        Returns:
            本地文档内容
        """
        try:
            documents = self.document_processor.load_local_documents()
            if not documents:
                logger.warning("未找到本地文档")
                return ""
            
            # 合并文档内容
            context = "\n\n".join([doc.content for doc in documents])
            return context
        except Exception as e:
            logger.error(f"获取本地文档上下文失败: {str(e)}", exc_info=True)
            return ""
    
    def _get_last_user_message(self, messages):
        """获取最后一条用户消息
        
        Args:
            messages: 消息列表
            
        Returns:
            最后一条用户消息的内容
        """
        for msg in reversed(messages):
            if msg['role'] == 'user':
                return msg['content']
        
        raise ValidationError("消息列表中没有用户消息")
    
    def _build_rag_context(self, retrieved_docs):
        """构建RAG上下文
        
        Args:
            retrieved_docs: 检索到的文档列表
            
        Returns:
            构建的上下文
        """
        if not retrieved_docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            context_parts.append(f"文档[{i+1}]: {doc.content}")
        
        return "\n\n".join(context_parts)
    
    def _build_messages_with_context(self, messages, context):
        """构建带有上下文的消息列表
        
        Args:
            messages: 原始消息列表
            context: 上下文内容
            
        Returns:
            处理后的消息列表
        """
        # 复制消息列表，避免修改原始列表
        processed_messages = messages.copy()
        
        # 如果没有系统消息，添加一个
        has_system_message = any(msg['role'] == 'system' for msg in processed_messages)
        if not has_system_message:
            processed_messages.insert(0, {
                "role": "system",
                "content": "你是一个智能助手，请根据提供的上下文回答用户的问题。"
            })
        
        # 如果有上下文，添加上下文消息
        if context:
            # 在系统消息之后，其他消息之前插入上下文
            context_message = {
                "role": "system",
                "content": f"以下是相关的上下文信息，请参考这些信息回答用户的问题：\n\n{context}"
            }
            processed_messages.insert(1, context_message)
        
        return processed_messages
    
    def _extract_sources(self, retrieved_docs):
        """提取来源信息
        
        Args:
            retrieved_docs: 检索到的文档列表
            
        Returns:
            来源信息列表
        """
        sources = []
        for doc in retrieved_docs:
            source = {
                "title": doc.metadata.get("title", "未知标题"),
                "source": doc.metadata.get("source", "未知来源"),
                "relevance": doc.metadata.get("relevance", 0)
            }
            sources.append(source)
        
        return sources 