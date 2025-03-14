"""核心组件模块

这个模块提供了应用所需的所有核心组件实例，采用懒加载模式。
"""

from typing import Optional, Dict, Any
from app.utils.config.loader import get_config as load_config
from app.core.document.processor import DocumentProcessor
from app.core.llm.client import LLMClient
from app.core.llm.response import LLMResponseHandler
from app.api.response.formatter import OpenAIResponseFormatter
from app.api.response.stream import StreamResponseHandler
from app.services.rag.service import RAGService
from app.core.search import ElasticsearchClient, DummySearchClient
from app.core.embedding import OpenAIEmbeddingClient, DummyEmbeddingClient
from app.utils.logging.logger import get_logger

logger = get_logger("core.components")

# 组件实例缓存
_components = {}
_config = None


def get_app_config() -> Dict[str, Any]:
    """获取配置，确保只加载一次"""
    global _config
    if _config is None:
        _config = load_config()
        logger.info("配置加载完成")
    return _config


def get_document_processor() -> DocumentProcessor:
    """获取文档处理器"""
    if 'document_processor' not in _components:
        config = get_app_config()
        # 使用index.upload_dir作为datasets_dir
        datasets_dir = config.get('index', {}).get('upload_dir', 'uploads')
        _components['document_processor'] = DocumentProcessor(
            datasets_dir=datasets_dir
        )
        logger.info("文档处理器初始化完成")
    return _components['document_processor']


def get_llm_client() -> LLMClient:
    """获取LLM客户端"""
    if 'llm_client' not in _components:
        config = get_app_config()
        llm_config = config.get('llm', {})
        _components['llm_client'] = LLMClient(
            model=llm_config.get('model_name', 'gpt-3.5-turbo'),
            api_key=llm_config.get('api_key', ''),
            endpoint=llm_config.get('api_base', ''),
            temperature=llm_config.get('temperature', 0.7),
            max_tokens=llm_config.get('max_tokens', 1024),
            top_p=llm_config.get('top_p', 1.0)
        )
        logger.info("LLM客户端初始化完成")
    return _components['llm_client']


def get_llm_response_handler() -> LLMResponseHandler:
    """获取LLM响应处理器"""
    if 'llm_response_handler' not in _components:
        _components['llm_response_handler'] = LLMResponseHandler()
        logger.info("LLM响应处理器初始化完成")
    return _components['llm_response_handler']


def get_response_formatter() -> OpenAIResponseFormatter:
    """获取响应格式化器"""
    if 'response_formatter' not in _components:
        _components['response_formatter'] = OpenAIResponseFormatter()
        logger.info("响应格式化器初始化完成")
    return _components['response_formatter']


def get_stream_handler() -> StreamResponseHandler:
    """获取流式响应处理器"""
    if 'stream_handler' not in _components:
        formatter = get_response_formatter()
        _components['stream_handler'] = StreamResponseHandler(formatter=formatter)
        logger.info("流式响应处理器初始化完成")
    return _components['stream_handler']


def get_search_client() -> Optional[ElasticsearchClient]:
    """获取搜索客户端"""
    if 'search_client' not in _components:
        try:
            _components['search_client'] = ElasticsearchClient()
            logger.info("搜索客户端初始化完成")
        except Exception as e:
            logger.warning(f"搜索客户端初始化失败: {str(e)}，使用虚拟搜索客户端")
            _components['search_client'] = DummySearchClient()
            logger.info("虚拟搜索客户端初始化完成")
    return _components['search_client']


def get_embedding_client() -> Optional[OpenAIEmbeddingClient]:
    """获取嵌入客户端"""
    if 'embedding_client' not in _components:
        try:
            _components['embedding_client'] = OpenAIEmbeddingClient()
            logger.info("嵌入客户端初始化完成")
        except Exception as e:
            logger.warning(f"嵌入客户端初始化失败: {str(e)}，使用虚拟嵌入客户端")
            _components['embedding_client'] = DummyEmbeddingClient()
            logger.info("虚拟嵌入客户端初始化完成")
    return _components['embedding_client']


def get_rag_service() -> RAGService:
    """获取RAG服务"""
    if 'rag_service' not in _components:
        config = get_app_config()
        rag_config = config.get('rag', {})
        _components['rag_service'] = RAGService(
            document_processor=get_document_processor(),
            llm_client=get_llm_client(),
            llm_response_handler=get_llm_response_handler(),
            system_message=rag_config.get('system_message', "你是一个智能助手，可以回答用户关于文档内容的问题。"),
            search_client=get_search_client(),
            embedding_client=get_embedding_client()
        )
        logger.info("RAG服务初始化完成")
    return _components['rag_service']


# 为了向后兼容，提供直接访问的属性
document_processor = get_document_processor()
llm_client = get_llm_client()
llm_response_handler = get_llm_response_handler()
response_formatter = get_response_formatter()
stream_handler = get_stream_handler()
search_client = get_search_client()
embedding_client = get_embedding_client()
rag_service = get_rag_service()

logger.info("核心组件模块加载完成") 