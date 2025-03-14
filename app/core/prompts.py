"""提示模板模块

这个模块提供了提示模板功能，使用 Langchain 的提示模板。
"""

from typing import Dict, Any, List, Optional

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

from app.utils.logging.logger import get_logger

logger = get_logger("core.prompts")

# 系统提示模板
DEFAULT_SYSTEM_TEMPLATE = """你是一个智能助手，可以回答用户关于文档内容的问题。
请基于提供的文档内容回答问题，如果无法从文档中找到答案，请明确告知用户。
保持回答简洁、准确，并引用相关的文档内容作为依据。"""

# RAG 提示模板
DEFAULT_RAG_TEMPLATE = """使用以下检索到的上下文信息来回答用户的问题。
如果你无法从上下文中找到答案，请直接说"我无法从提供的信息中找到答案"，不要试图编造答案。
保持回答简洁、准确，并引用相关的文档内容作为依据。

上下文信息:
{context}

用户问题: {question}

回答:"""

# 聊天历史提示模板
DEFAULT_CHAT_TEMPLATE = """你是一个智能助手，可以回答用户关于文档内容的问题。
请基于提供的文档内容和聊天历史回答问题，如果无法从文档中找到答案，请明确告知用户。

文档内容:
{context}

聊天历史:
{chat_history}

用户问题: {question}

回答:"""

class PromptTemplateService:
    """提示模板服务类，提供提示模板功能"""
    
    @staticmethod
    def create_rag_prompt(
        template: Optional[str] = None,
        context_variable: str = "context",
        question_variable: str = "question"
    ) -> PromptTemplate:
        """创建 RAG 提示模板
        
        Args:
            template: 提示模板文本，如果为 None 则使用默认模板
            context_variable: 上下文变量名
            question_variable: 问题变量名
            
        Returns:
            提示模板对象
        """
        if template is None:
            template = DEFAULT_RAG_TEMPLATE
        
        prompt = PromptTemplate.from_template(
            template,
            template_format="f-string",
            input_variables=[context_variable, question_variable]
        )
        
        logger.debug(f"创建 RAG 提示模板: {template}")
        return prompt
    
    @staticmethod
    def create_chat_prompt(
        system_template: Optional[str] = None,
        human_template: str = "{input}",
        include_history: bool = True
    ) -> ChatPromptTemplate:
        """创建聊天提示模板
        
        Args:
            system_template: 系统提示模板文本，如果为 None 则使用默认模板
            human_template: 人类提示模板文本
            include_history: 是否包含聊天历史
            
        Returns:
            聊天提示模板对象
        """
        if system_template is None:
            system_template = DEFAULT_SYSTEM_TEMPLATE
        
        messages = []
        
        # 添加系统消息
        messages.append(
            SystemMessagePromptTemplate.from_template(system_template)
        )
        
        # 添加聊天历史
        if include_history:
            messages.append(MessagesPlaceholder(variable_name="chat_history"))
        
        # 添加人类消息
        messages.append(
            HumanMessagePromptTemplate.from_template(human_template)
        )
        
        # 创建聊天提示模板
        chat_prompt = ChatPromptTemplate.from_messages(messages)
        
        logger.debug(f"创建聊天提示模板: 系统={system_template}, 人类={human_template}, 包含历史={include_history}")
        return chat_prompt
    
    @staticmethod
    def create_rag_chat_prompt(
        system_template: Optional[str] = None,
        include_history: bool = True,
        context_variable: str = "context",
        question_variable: str = "question",
        history_variable: str = "chat_history"
    ) -> ChatPromptTemplate:
        """创建 RAG 聊天提示模板
        
        Args:
            system_template: 系统提示模板文本，如果为 None 则使用默认模板
            include_history: 是否包含聊天历史
            context_variable: 上下文变量名
            question_variable: 问题变量名
            history_variable: 历史变量名
            
        Returns:
            聊天提示模板对象
        """
        if system_template is None:
            system_template = """你是一个智能助手，可以回答用户关于文档内容的问题。
请基于提供的文档内容回答问题，如果无法从文档中找到答案，请明确告知用户。

文档内容:
{context}"""
        
        messages = []
        
        # 添加系统消息
        messages.append(
            SystemMessagePromptTemplate.from_template(system_template)
        )
        
        # 添加聊天历史
        if include_history:
            messages.append(MessagesPlaceholder(variable_name=history_variable))
        
        # 添加人类消息
        messages.append(
            HumanMessagePromptTemplate.from_template("{" + question_variable + "}")
        )
        
        # 创建聊天提示模板
        chat_prompt = ChatPromptTemplate.from_messages(messages)
        
        logger.debug(f"创建 RAG 聊天提示模板: 系统={system_template}, 包含历史={include_history}")
        return chat_prompt 