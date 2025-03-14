"""服务管理模块

这个模块提供了一个简单的服务管理器，用于管理应用的各种服务组件。
"""

from typing import Dict, Any, Optional
import logging

from app.utils.config.loader import get_config
from app.core.document.processor import DocumentProcessor
from app.core.llm.client import LLMClient
from app.core.llm.response import LLMResponseHandler
from app.api.response.formatter import OpenAIResponseFormatter
from app.api.response.stream import StreamResponseHandler
from app.services.rag.service import RAGService
from app.core.search import ElasticsearchClient
from app.core.embedding import OpenAIEmbeddingClient

logger = logging.getLogger("services.manager")

class ServiceManager:
    """服务管理器，用于管理应用的各种服务组件"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if ServiceManager._initialized:
            return
        
        logger.info("初始化服务管理器")
        
        # 加载配置
        self.config = get_config()
        
        # 初始化服务组件
        self._init_services()
        
        ServiceManager._initialized = True
    
    def _init_services(self):
        """初始化所有服务组件"""
        # 文档处理器
        self._document_processor = DocumentProcessor(
            datasets_dir=self.config['document']['datasets_dir']
        )
        
        # LLM客户端
        self._llm_client = LLMClient(
            model=self.config['llm']['model'],
            api_key=self.config['llm']['api_key'],
            endpoint=self.config['llm']['endpoint'],
            temperature=self.config['llm']['temperature'],
            max_tokens=self.config['llm']['max_tokens'],
            top_p=self.config['llm']['top_p']
        )
        
        # LLM响应处理器
        self._llm_response_handler = LLMResponseHandler()
        
        # 响应格式化器
        self._response_formatter = OpenAIResponseFormatter()
        
        # 流式响应处理器
        self._stream_handler = StreamResponseHandler(
            formatter=self._response_formatter
        )
        
        # 搜索客户端
        self._search_client = ElasticsearchClient()
        
        # 嵌入客户端
        self._embedding_client = OpenAIEmbeddingClient()
        
        # RAG服务
        self._rag_service = RAGService(
            document_processor=self._document_processor,
            llm_client=self._llm_client,
            llm_response_handler=self._llm_response_handler,
            system_message=self.config['rag']['system_message'],
            search_client=self._search_client,
            embedding_client=self._embedding_client
        )
        
        logger.info("服务组件初始化完成")
    
    @property
    def document_processor(self) -> DocumentProcessor:
        """获取文档处理器"""
        return self._document_processor
    
    @property
    def llm_client(self) -> LLMClient:
        """获取LLM客户端"""
        return self._llm_client
    
    @property
    def llm_response_handler(self) -> LLMResponseHandler:
        """获取LLM响应处理器"""
        return self._llm_response_handler
    
    @property
    def response_formatter(self) -> OpenAIResponseFormatter:
        """获取响应格式化器"""
        return self._response_formatter
    
    @property
    def stream_handler(self) -> StreamResponseHandler:
        """获取流式响应处理器"""
        return self._stream_handler
    
    @property
    def search_client(self) -> ElasticsearchClient:
        """获取搜索客户端"""
        return self._search_client
    
    @property
    def embedding_client(self) -> OpenAIEmbeddingClient:
        """获取嵌入客户端"""
        return self._embedding_client
    
    @property
    def rag_service(self) -> RAGService:
        """获取RAG服务"""
        return self._rag_service


# 创建全局服务管理器实例
_service_manager = None

def get_service_manager() -> ServiceManager:
    """获取服务管理器实例
    
    Returns:
        服务管理器实例
    """
    global _service_manager
    if _service_manager is None:
        _service_manager = ServiceManager()
    return _service_manager 