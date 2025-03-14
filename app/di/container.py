"""依赖管理模块

这个模块提供了简单的依赖管理功能，使用单例模式管理应用程序的服务实例。
"""

from typing import Dict, Any, Optional, Callable
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import CallbackManager
from langchain_core.language_models.fake import FakeListLLM

from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService
from app.services.document_loader_service import DocumentLoaderService
from app.services.prompt_service import PromptService
from app.services.search_service import SearchService
from app.services.index_service import IndexService
from app.services.chat_service import ChatService
from app.services.rag_service import RAGService

from app.utils.logging.logger import get_logger
from app.config import load_config

logger = get_logger("di.container")

class ServiceContainer:
    """服务容器类
    
    这个类使用单例模式管理应用程序的服务实例。
    """
    
    _instance = None
    
    def __new__(cls, config_path: Optional[str] = None):
        """创建单例实例
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
            
        Returns:
            ServiceContainer实例
        """
        if cls._instance is None:
            cls._instance = super(ServiceContainer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化容器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        # 避免重复初始化
        if self._initialized:
            return
            
        # 加载配置
        self._config = load_config(config_path)
        
        # 初始化服务实例字典
        self._services = {}
        
        # 标记为已初始化
        self._initialized = True
        logger.debug("服务容器初始化完成")
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置
        
        Returns:
            配置字典
        """
        return self._config
    
    def register(self, name: str, instance: Any) -> None:
        """注册服务实例
        
        Args:
            name: 服务名称
            instance: 服务实例
        """
        self._services[name] = instance
        logger.debug(f"注册服务: {name}")
    
    def get(self, name: str) -> Any:
        """获取服务实例
        
        Args:
            name: 服务名称
            
        Returns:
            服务实例
            
        Raises:
            KeyError: 服务不存在时抛出
        """
        if name not in self._services:
            raise KeyError(f"服务不存在: {name}")
        return self._services[name]
    
    def has(self, name: str) -> bool:
        """检查服务是否存在
        
        Args:
            name: 服务名称
            
        Returns:
            服务是否存在
        """
        return name in self._services
    
    def initialize_services(self) -> None:
        """初始化所有服务
        
        初始化应用程序所需的所有服务。
        """
        logger.info("初始化服务...")
        
        # 初始化LLM服务
        self._initialize_llm_service()
        
        # 初始化嵌入服务
        self._initialize_embedding_service()
        
        # 初始化向量存储服务
        self._initialize_vector_store_service()
        
        # 初始化文档加载服务
        self._initialize_document_loader_service()
        
        # 初始化提示服务
        self._initialize_prompt_service()
        
        # 初始化搜索服务
        self._initialize_search_service()
        
        # 初始化索引服务
        self._initialize_index_service()
        
        # 初始化RAG服务
        self._initialize_rag_service()
        
        # 初始化聊天服务
        self._initialize_chat_service()
        
        logger.info("服务初始化完成")
    
    def _initialize_llm_service(self) -> None:
        """初始化LLM服务"""
        if self.has("llm_service"):
            return
            
        logger.info("初始化LLM服务...")
        
        # 获取LLM配置
        llm_config = self._config.get("llm", {})
        model_name = llm_config.get("model_name", "gpt-3.5-turbo")
        temperature = llm_config.get("temperature", 0.7)
        streaming = llm_config.get("streaming", True)
        api_key = llm_config.get("api_key")
        api_base = llm_config.get("api_base")
        
        try:
            # 创建LLM
            llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                streaming=streaming,
                openai_api_key=api_key,
                openai_api_base=api_base
            )
            
            logger.info(f"LLM服务初始化完成，使用模型: {model_name}")
        except Exception as e:
            # 如果初始化失败，使用虚拟LLM
            logger.warning(f"LLM初始化失败: {str(e)}，使用虚拟LLM")
            llm = FakeListLLM(responses=["这是一个虚拟的LLM响应，因为没有有效的API密钥。"])
        
        # 创建LLM服务
        llm_service = LLMService(
            llm=llm,
            config=llm_config
        )
        
        # 注册LLM服务
        self.register("llm_service", llm_service)
    
    def _initialize_embedding_service(self) -> None:
        """初始化嵌入服务"""
        if self.has("embedding_service"):
            return
        
        logger.info("初始化嵌入服务...")
        
        # 获取嵌入配置
        embedding_config = self._config.get("embedding", {})
        model_name = embedding_config.get("model_name", "text-embedding-ada-002")
        api_key = embedding_config.get("api_key")
        api_base = embedding_config.get("api_base")
        
        try:
            # 创建嵌入模型
            embedding_model = OpenAIEmbeddings(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=api_base
            )
            
            logger.info(f"嵌入服务初始化完成，使用模型: {model_name}")
        except Exception as e:
            # 如果初始化失败，使用虚拟嵌入模型
            logger.warning(f"嵌入服务初始化失败: {str(e)}，使用虚拟嵌入模型")
            from langchain_core.embeddings import FakeEmbeddings
            embedding_model = FakeEmbeddings(size=1536)
        
        # 创建嵌入服务
        embedding_service = EmbeddingService(
            embedding_model=embedding_model,
            config=self._config
        )
        
        # 注册嵌入服务
        self.register("embedding_service", embedding_service)
    
    def _initialize_vector_store_service(self) -> None:
        """初始化向量存储服务"""
        if self.has("vector_store_service"):
            return
        
        logger.info("初始化向量存储服务...")
        
        # 获取嵌入服务
        embedding_service = self.get("embedding_service")
        
        # 获取向量存储配置
        vector_store_config = self._config.get("vector_store", {})
        persist_directory = vector_store_config.get("persist_directory", "chroma_db")
        
        # 确保目录存在
        os.makedirs(persist_directory, exist_ok=True)
        
        try:
            # 尝试导入Chroma
            try:
                from langchain_community.vectorstores import Chroma
                
                # 创建向量存储
                vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embedding_service.get_embedding_model()
                )
                
                logger.info(f"向量存储服务初始化完成，使用存储目录: {persist_directory}")
            except ImportError:
                # 如果Chroma不可用，尝试使用FAISS
                logger.warning("Chroma不可用，尝试使用FAISS作为替代")
                try:
                    from langchain_community.vectorstores import FAISS
                    
                    # 创建内存向量存储
                    vector_store = FAISS(
                        embedding_function=embedding_service.get_embedding_model(),
                        index_name="default_index"
                    )
                    
                    logger.info("向量存储服务初始化完成，使用FAISS内存存储")
                except ImportError:
                    # 如果FAISS也不可用，使用虚拟向量存储
                    logger.warning("FAISS也不可用，使用虚拟向量存储")
                    from langchain_core.vectorstores import VectorStore
                    
                    # 创建一个简单的虚拟向量存储
                    class DummyVectorStore(VectorStore):
                        def add_documents(self, documents, **kwargs):
                            logger.info(f"添加了 {len(documents)} 个文档到虚拟向量存储")
                            return ["dummy_id"] * len(documents)
                        
                        def similarity_search(self, query, k=4, **kwargs):
                            logger.info(f"在虚拟向量存储中搜索: {query}")
                            return []
                        
                        @classmethod
                        def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
                            logger.info(f"从 {len(texts)} 个文本创建虚拟向量存储")
                            return cls()
                    
                    vector_store = DummyVectorStore()
                    logger.info("向量存储服务初始化完成，使用虚拟向量存储")
        except Exception as e:
            # 如果初始化失败，使用内存向量存储
            logger.warning(f"向量存储服务初始化失败: {str(e)}，使用内存向量存储")
            
            # 创建一个简单的虚拟向量存储
            from langchain_core.vectorstores import VectorStore
            
            class DummyVectorStore(VectorStore):
                def add_documents(self, documents, **kwargs):
                    logger.info(f"添加了 {len(documents)} 个文档到虚拟向量存储")
                    return ["dummy_id"] * len(documents)
                
                def similarity_search(self, query, k=4, **kwargs):
                    logger.info(f"在虚拟向量存储中搜索: {query}")
                    return []
                
                @classmethod
                def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
                    logger.info(f"从 {len(texts)} 个文本创建虚拟向量存储")
                    return cls()
            
            vector_store = DummyVectorStore()
            logger.info("向量存储服务初始化完成，使用虚拟向量存储")
        
        # 创建向量存储服务
        vector_store_service = VectorStoreService(
            vector_store=vector_store,
            embedding_service=embedding_service,
            config=self._config
        )
        
        # 注册向量存储服务
        self.register("vector_store_service", vector_store_service)
    
    def _initialize_document_loader_service(self) -> None:
        """初始化文档加载服务"""
        if self.has("document_loader_service"):
            return
        
        logger.info("初始化文档加载服务...")
        
        # 创建文档加载服务
        document_loader_service = DocumentLoaderService(
            config=self._config
        )
        
        # 注册文档加载服务
        self.register("document_loader_service", document_loader_service)
        
        logger.info("文档加载服务初始化完成")
    
    def _initialize_prompt_service(self) -> None:
        """初始化提示服务"""
        if self.has("prompt_service"):
            return
        
        logger.info("初始化提示服务...")
        
        # 获取提示配置
        prompt_config = self._config.get("prompt", {})
        templates_dir = prompt_config.get("templates_dir", "prompts")
        
        # 创建提示服务
        prompt_service = PromptService(
            config=self._config,
            prompt_templates_dir=templates_dir
        )
        
        # 注册提示服务
        self.register("prompt_service", prompt_service)
        
        logger.info(f"提示服务初始化完成，使用模板目录: {templates_dir}")
    
    def _initialize_search_service(self) -> None:
        """初始化搜索服务"""
        if self.has("search_service"):
            return
        
        logger.info("初始化搜索服务...")
        
        # 获取依赖服务
        vector_store_service = self.get("vector_store_service")
        embedding_service = self.get("embedding_service")
        
        # 创建搜索服务
        search_service = SearchService(
            vector_store_service=vector_store_service,
            embedding_service=embedding_service,
            config=self._config
        )
        
        # 注册搜索服务
        self.register("search_service", search_service)
        
        logger.info("搜索服务初始化完成")
    
    def _initialize_index_service(self) -> None:
        """初始化索引服务"""
        if self.has("index_service"):
            return
        
        logger.info("初始化索引服务...")
        
        # 获取依赖服务
        vector_store_service = self.get("vector_store_service")
        embedding_service = self.get("embedding_service")
        document_loader_service = self.get("document_loader_service")
        
        # 创建索引服务
        index_service = IndexService(
            vector_store_service=vector_store_service,
            embedding_service=embedding_service,
            document_loader_service=document_loader_service,
            config=self._config
        )
        
        # 注册索引服务
        self.register("index_service", index_service)
        
        logger.info("索引服务初始化完成")
    
    def _initialize_rag_service(self) -> None:
        """初始化RAG服务"""
        if self.has("rag_service"):
            return
        
        logger.info("初始化RAG服务...")
        
        # 获取依赖服务
        llm_service = self.get("llm_service")
        vector_store_service = self.get("vector_store_service")
        prompt_service = self.get("prompt_service")
        
        # 获取RAG配置
        rag_config = self._config.get("rag", {})
        system_message = rag_config.get("system_message", "你是一个有用的AI助手，能够根据提供的上下文回答问题。")
        top_k = rag_config.get("top_k", 5)
        search_type = rag_config.get("search_type", "similarity")
        
        # 创建RAG服务
        rag_service = RAGService(
            llm_service=llm_service,
            vector_store_service=vector_store_service,
            prompt_service=prompt_service,
            system_message=system_message,
            top_k=top_k,
            search_type=search_type
        )
        
        # 注册RAG服务
        self.register("rag_service", rag_service)
        
        logger.info("RAG服务初始化完成")
    
    def _initialize_chat_service(self) -> None:
        """初始化聊天服务"""
        if self.has("chat_service"):
            return
        
        logger.info("初始化聊天服务...")
        
        # 获取依赖服务
        llm_service = self.get("llm_service")
        document_loader_service = self.get("document_loader_service")
        rag_service = self.get("rag_service")
        prompt_service = self.get("prompt_service")
        
        # 创建聊天服务
        chat_service = ChatService(
            llm_service=llm_service,
            document_loader=document_loader_service,
            rag_service=rag_service,
            prompt_service=prompt_service
        )
        
        # 注册聊天服务
        self.register("chat_service", chat_service)
        
        logger.info("聊天服务初始化完成")

# 全局容器实例
_container = None

def get_container(config_path: Optional[str] = None) -> ServiceContainer:
    """获取容器实例
    
    如果容器已创建，则返回现有实例，否则创建新实例
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认路径
    
    Returns:
        容器实例
    """
    global _container
    
    if _container is None:
        _container = ServiceContainer(config_path)
    
    return _container

def initialize_container(config: Dict[str, Any]) -> ServiceContainer:
    """初始化容器并注册所有服务
    
    Args:
        config: 配置字典
        
    Returns:
        初始化后的容器实例
    """
    global _container
    
    # 创建容器实例
    if _container is None:
        _container = ServiceContainer()
    
    # 手动设置配置
    _container._config = config
    
    # 初始化所有服务
    _container.initialize_services()
    
    return _container

# 公共容器实例，供其他模块导入
container = get_container() 