"""提示服务模块

这个模块提供了提示模板相关的服务功能，使用 Langchain 的最新提示功能。
"""

from typing import List, Dict, Any, Optional, Union, Callable
import os
import json

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage

from app.utils.logging.logger import get_logger
from app.utils.error_handler import ServiceError, ValidationError

logger = get_logger("services.prompt")

class PromptService:
    """提示服务类，提供提示模板功能"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        prompt_templates_dir: Optional[str] = None
    ):
        """初始化提示服务
        
        Args:
            config: 配置字典
            prompt_templates_dir: 提示模板目录
        """
        self.config = config
        
        # 获取提示配置
        if prompt_templates_dir is None:
            prompt_templates_dir = config.get('prompt', {}).get('templates_dir', 'prompts')
        
        self.prompt_templates_dir = prompt_templates_dir
        
        # 加载提示模板
        self.templates = {}
        self.load_templates()
        
        logger.debug("提示服务初始化完成")
    
    def load_templates(self):
        """加载提示模板
        
        从提示模板目录加载所有JSON格式的提示模板
        """
        try:
            if not os.path.exists(self.prompt_templates_dir):
                logger.warning(f"提示模板目录不存在: {self.prompt_templates_dir}")
                return
            
            logger.info(f"从目录加载提示模板: {self.prompt_templates_dir}")
            
            # 遍历目录中的所有JSON文件
            for filename in os.listdir(self.prompt_templates_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.prompt_templates_dir, filename)
                    template_name = os.path.splitext(filename)[0]
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            template_data = json.load(f)
                        
                        # 存储模板数据
                        self.templates[template_name] = template_data
                        logger.debug(f"加载提示模板: {template_name}")
                        
                    except Exception as e:
                        logger.error(f"加载提示模板失败: {file_path}, 错误: {str(e)}")
            
            logger.info(f"提示模板加载完成，共加载了 {len(self.templates)} 个模板")
            
        except Exception as e:
            error_msg = f"加载提示模板失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def get_template(self, template_name: str) -> Dict[str, Any]:
        """获取提示模板
        
        Args:
            template_name: 模板名称
            
        Returns:
            模板数据
            
        Raises:
            ValidationError: 模板不存在时抛出
        """
        if template_name not in self.templates:
            raise ValidationError(f"提示模板不存在: {template_name}")
        
        return self.templates[template_name]
    
    def create_prompt_template(
        self, 
        template: str,
        input_variables: List[str],
        template_format: str = "f-string",
        validate_template: bool = True
    ) -> PromptTemplate:
        """创建提示模板
        
        Args:
            template: 模板字符串
            input_variables: 输入变量列表
            template_format: 模板格式
            validate_template: 是否验证模板
            
        Returns:
            提示模板实例
            
        Raises:
            ServiceError: 创建失败时抛出
        """
        try:
            logger.debug(f"创建提示模板: input_variables={input_variables}")
            
            prompt_template = PromptTemplate(
                template=template,
                input_variables=input_variables,
                template_format=template_format,
                validate_template=validate_template
            )
            
            return prompt_template
            
        except Exception as e:
            error_msg = f"创建提示模板失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def create_chat_prompt_template(
        self, 
        system_message: str,
        human_message_template: Optional[str] = None,
        input_variables: Optional[List[str]] = None,
        include_history: bool = True
    ) -> ChatPromptTemplate:
        """创建聊天提示模板
        
        Args:
            system_message: 系统消息
            human_message_template: 人类消息模板
            input_variables: 输入变量列表
            include_history: 是否包含历史消息
            
        Returns:
            聊天提示模板实例
            
        Raises:
            ServiceError: 创建失败时抛出
        """
        try:
            logger.debug(f"创建聊天提示模板: include_history={include_history}")
            
            # 创建消息列表
            messages = []
            
            # 添加系统消息
            messages.append({"role": "system", "content": system_message})
            
            # 添加历史消息占位符
            if include_history:
                messages.append(MessagesPlaceholder(variable_name="chat_history"))
            
            # 添加人类消息模板
            if human_message_template:
                if input_variables is None:
                    # 自动提取输入变量
                    input_variables = []
                    for var in human_message_template.split("{"):
                        if "}" in var:
                            var_name = var.split("}")[0].strip()
                            if var_name and var_name != "chat_history":
                                input_variables.append(var_name)
                
                # 直接使用模板字符串
                messages.append({"role": "human", "content": human_message_template})
            else:
                # 使用默认的人类消息占位符
                messages.append({"role": "human", "content": "{input}"})
                input_variables = ["input"]
            
            # 创建聊天提示模板
            chat_prompt_template = ChatPromptTemplate.from_messages(messages)
            
            return chat_prompt_template
            
        except Exception as e:
            error_msg = f"创建聊天提示模板失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def create_rag_prompt_template(
        self, 
        system_message: str,
        human_message_template: Optional[str] = None,
        include_history: bool = True
    ) -> ChatPromptTemplate:
        """创建RAG提示模板
        
        Args:
            system_message: 系统消息
            human_message_template: 人类消息模板
            include_history: 是否包含历史消息
            
        Returns:
            RAG提示模板实例
            
        Raises:
            ServiceError: 创建失败时抛出
        """
        try:
            logger.debug(f"创建RAG提示模板: include_history={include_history}")
            
            # 使用默认的RAG人类消息模板
            if human_message_template is None:
                human_message_template = """
                请根据以下上下文回答我的问题。如果上下文中没有相关信息，请说明你不知道，不要编造答案。

                上下文:
                {context}

                问题: {question}
                """
            
            # 创建RAG提示模板
            input_variables = ["context", "question"]
            if include_history:
                input_variables.append("chat_history")
            
            return self.create_chat_prompt_template(
                system_message=system_message,
                human_message_template=human_message_template,
                input_variables=input_variables,
                include_history=include_history
            )
            
        except Exception as e:
            error_msg = f"创建RAG提示模板失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def create_rag_prompt(self) -> ChatPromptTemplate:
        """创建RAG提示
        
        使用默认的系统消息创建RAG提示模板
        
        Returns:
            RAG提示模板实例
            
        Raises:
            ServiceError: 创建失败时抛出
        """
        try:
            logger.debug("创建RAG提示")
            
            # 使用默认的系统消息
            system_message = """你是一个智能助手，负责根据提供的上下文回答用户问题。
            请仅使用提供的上下文信息回答问题。如果上下文中没有足够的信息，请诚实地告知用户你不知道，不要编造答案。
            回答应该简洁、准确、有帮助。"""
            
            return self.create_rag_prompt_template(
                system_message=system_message,
                include_history=False
            )
            
        except Exception as e:
            error_msg = f"创建RAG提示失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def format_prompt(
        self, 
        prompt_template: Union[PromptTemplate, ChatPromptTemplate],
        **kwargs
    ) -> Union[str, List[BaseMessage]]:
        """格式化提示
        
        Args:
            prompt_template: 提示模板
            **kwargs: 格式化参数
            
        Returns:
            格式化后的提示
            
        Raises:
            ServiceError: 格式化失败时抛出
        """
        try:
            logger.debug(f"格式化提示: kwargs={kwargs.keys()}")
            
            # 格式化提示
            formatted_prompt = prompt_template.format(**kwargs)
            
            return formatted_prompt
            
        except Exception as e:
            error_msg = f"格式化提示失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def format_document_context(self, documents: List[Any]) -> str:
        """格式化文档上下文
        
        将文档列表格式化为字符串上下文
        
        Args:
            documents: 文档列表
            
        Returns:
            格式化后的上下文字符串
        """
        if not documents:
            return ""
        
        # 格式化文档
        formatted_docs = []
        for i, doc in enumerate(documents):
            # 处理不同类型的文档
            if hasattr(doc, "page_content"):
                # Langchain Document
                content = doc.page_content
                metadata = getattr(doc, "metadata", {})
                
                # 格式化元数据
                meta_str = ""
                if metadata:
                    meta_items = []
                    for key, value in metadata.items():
                        if key in ["source", "title", "url", "page"]:
                            meta_items.append(f"{key}: {value}")
                    if meta_items:
                        meta_str = f" ({', '.join(meta_items)})"
                
                formatted_docs.append(f"[文档 {i+1}{meta_str}]\n{content}\n")
            elif isinstance(doc, dict) and "content" in doc:
                # 字典格式
                content = doc["content"]
                metadata = doc.get("metadata", {})
                
                # 格式化元数据
                meta_str = ""
                if metadata:
                    meta_items = []
                    for key, value in metadata.items():
                        if key in ["source", "title", "url", "page"]:
                            meta_items.append(f"{key}: {value}")
                    if meta_items:
                        meta_str = f" ({', '.join(meta_items)})"
                
                formatted_docs.append(f"[文档 {i+1}{meta_str}]\n{content}\n")
            else:
                # 其他格式
                formatted_docs.append(f"[文档 {i+1}]\n{str(doc)}\n")
        
        # 合并文档
        return "\n".join(formatted_docs) 