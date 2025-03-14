"""向量存储服务模块

这个模块提供了向量存储相关的服务功能，使用 Langchain 的最新向量存储功能。
"""

from typing import List, Dict, Any, Optional, Union, Callable
import time

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever

from app.utils.logging.logger import get_logger
from app.utils.error_handler import ServiceError, ValidationError

logger = get_logger("services.vector_store")

class VectorStoreService:
    """向量存储服务类，提供向量存储功能"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service,
        config: Dict[str, Any]
    ):
        """初始化向量存储服务
        
        Args:
            vector_store: 向量存储实例
            embedding_service: 嵌入服务实例
            config: 配置字典
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.config = config
        
        # 获取向量存储配置
        self.default_top_k = config.get('search', {}).get('default_top_k', 5)
        self.default_search_type = config.get('search', {}).get('default_search_type', 'similarity')
        
        logger.debug("向量存储服务初始化完成")
    
    def get_vector_store(self) -> VectorStore:
        """获取向量存储实例
        
        Returns:
            向量存储实例
        """
        return self.vector_store
    
    def as_retriever(
        self,
        search_type: Optional[str] = None,
        search_kwargs: Optional[Dict[str, Any]] = None
    ) -> BaseRetriever:
        """将向量存储转换为检索器
        
        Args:
            search_type: 搜索类型，支持 "similarity", "mmr"
            search_kwargs: 搜索参数
            
        Returns:
            检索器实例
            
        Raises:
            ServiceError: 转换失败时抛出
        """
        try:
            # 使用默认值如果未指定
            if search_type is None:
                search_type = self.default_search_type
            
            if search_kwargs is None:
                search_kwargs = {"k": self.default_top_k}
            
            logger.debug(f"创建检索器: search_type={search_type}, search_kwargs={search_kwargs}")
            
            # 创建检索器
            retriever = self.vector_store.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
            
            return retriever
            
        except Exception as e:
            error_msg = f"创建检索器失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到向量存储
        
        Args:
            documents: 文档列表
            
        Returns:
            文档ID列表
            
        Raises:
            ServiceError: 添加失败时抛出
        """
        try:
            if not documents:
                logger.warning("没有文档需要添加")
                return []
            
            logger.info(f"添加 {len(documents)} 个文档到向量存储")
            
            # 添加文档
            start_time = time.time()
            ids = self.vector_store.add_documents(documents)
            elapsed = time.time() - start_time
            
            logger.info(f"文档添加完成，添加了 {len(ids)} 个文档，耗时: {elapsed:.2f}秒")
            
            return ids
            
        except Exception as e:
            error_msg = f"添加文档失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def add_texts(
        self, 
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """添加文本到向量存储
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
            
        Returns:
            文档ID列表
            
        Raises:
            ServiceError: 添加失败时抛出
        """
        try:
            if not texts:
                logger.warning("没有文本需要添加")
                return []
            
            logger.info(f"添加 {len(texts)} 个文本到向量存储")
            
            # 添加文本
            start_time = time.time()
            ids = self.vector_store.add_texts(texts, metadatas)
            elapsed = time.time() - start_time
            
            logger.info(f"文本添加完成，添加了 {len(ids)} 个文本，耗时: {elapsed:.2f}秒")
            
            return ids
            
        except Exception as e:
            error_msg = f"添加文本失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def similarity_search(
        self, 
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """相似度搜索
        
        Args:
            query: 搜索查询
            top_k: 返回的最大结果数
            filter_dict: 过滤条件字典
            
        Returns:
            搜索结果列表
            
        Raises:
            ServiceError: 搜索失败时抛出
        """
        try:
            if not query:
                raise ValidationError("搜索查询不能为空")
            
            # 使用默认值如果未指定
            if top_k is None:
                top_k = self.default_top_k
            
            logger.info(f"执行相似度搜索: query='{query}', top_k={top_k}")
            
            # 执行搜索
            start_time = time.time()
            results = self.vector_store.similarity_search(
                query=query,
                k=top_k,
                filter=filter_dict
            )
            elapsed = time.time() - start_time
            
            # 记录搜索结果
            logger.info(f"相似度搜索完成，找到 {len(results)} 个结果，耗时: {elapsed:.2f}秒")
            
            return results
            
        except Exception as e:
            error_msg = f"相似度搜索失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def similarity_search_with_score(
        self, 
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """带分数的相似度搜索
        
        Args:
            query: 搜索查询
            top_k: 返回的最大结果数
            filter_dict: 过滤条件字典
            
        Returns:
            搜索结果列表，每个结果是 (文档, 分数) 的元组
            
        Raises:
            ServiceError: 搜索失败时抛出
        """
        try:
            if not query:
                raise ValidationError("搜索查询不能为空")
            
            # 使用默认值如果未指定
            if top_k is None:
                top_k = self.default_top_k
            
            logger.info(f"执行带分数的相似度搜索: query='{query}', top_k={top_k}")
            
            # 执行搜索
            start_time = time.time()
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                filter=filter_dict
            )
            elapsed = time.time() - start_time
            
            # 记录搜索结果
            logger.info(f"带分数的相似度搜索完成，找到 {len(results)} 个结果，耗时: {elapsed:.2f}秒")
            
            return results
            
        except Exception as e:
            error_msg = f"带分数的相似度搜索失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def max_marginal_relevance_search(
        self, 
        query: str,
        top_k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """最大边际相关性搜索
        
        Args:
            query: 搜索查询
            top_k: 返回的最大结果数
            fetch_k: 初始获取的结果数
            lambda_mult: 多样性权重
            filter_dict: 过滤条件字典
            
        Returns:
            搜索结果列表
            
        Raises:
            ServiceError: 搜索失败时抛出
        """
        try:
            if not query:
                raise ValidationError("搜索查询不能为空")
            
            # 使用默认值如果未指定
            if top_k is None:
                top_k = self.default_top_k
            
            if fetch_k is None:
                fetch_k = 4 * top_k
            
            if lambda_mult is None:
                lambda_mult = 0.5
            
            logger.info(f"执行MMR搜索: query='{query}', top_k={top_k}, fetch_k={fetch_k}, lambda_mult={lambda_mult}")
            
            # 执行搜索
            start_time = time.time()
            results = self.vector_store.max_marginal_relevance_search(
                query=query,
                k=top_k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter_dict
            )
            elapsed = time.time() - start_time
            
            # 记录搜索结果
            logger.info(f"MMR搜索完成，找到 {len(results)} 个结果，耗时: {elapsed:.2f}秒")
            
            return results
            
        except Exception as e:
            error_msg = f"MMR搜索失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def hybrid_search(
        self, 
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """混合搜索
        
        结合向量搜索和关键词搜索的混合搜索方法。
        
        Args:
            query: 搜索查询
            top_k: 返回的最大结果数
            filter_dict: 过滤条件字典
            
        Returns:
            搜索结果列表
            
        Raises:
            ServiceError: 搜索失败时抛出
        """
        try:
            if not query:
                raise ValidationError("搜索查询不能为空")
            
            # 使用默认值如果未指定
            if top_k is None:
                top_k = self.default_top_k
            
            logger.info(f"执行混合搜索: query='{query}', top_k={top_k}")
            
            # 检查向量存储是否支持混合搜索
            if hasattr(self.vector_store, "hybrid_search"):
                # 执行混合搜索
                start_time = time.time()
                results = self.vector_store.hybrid_search(
                    query=query,
                    k=top_k,
                    filter=filter_dict
                )
                elapsed = time.time() - start_time
                
                # 记录搜索结果
                logger.info(f"混合搜索完成，找到 {len(results)} 个结果，耗时: {elapsed:.2f}秒")
                
                return results
            else:
                # 如果不支持混合搜索，退回到普通搜索
                logger.warning("向量存储不支持混合搜索，使用普通搜索")
                return self.similarity_search(query, top_k, filter_dict)
            
        except Exception as e:
            error_msg = f"混合搜索失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def delete(self, filter_dict: Dict[str, Any]) -> int:
        """删除文档
        
        Args:
            filter_dict: 过滤条件字典
            
        Returns:
            删除的文档数量
            
        Raises:
            ServiceError: 删除失败时抛出
        """
        try:
            if not filter_dict:
                raise ValidationError("过滤条件不能为空")
            
            logger.info(f"删除文档: filter={filter_dict}")
            
            # 检查向量存储是否支持删除
            if hasattr(self.vector_store, "delete"):
                # 执行删除
                start_time = time.time()
                deleted_count = self.vector_store.delete(filter=filter_dict)
                elapsed = time.time() - start_time
                
                # 记录删除结果
                logger.info(f"文档删除完成，删除了 {deleted_count} 个文档，耗时: {elapsed:.2f}秒")
                
                return deleted_count
            else:
                error_msg = "向量存储不支持删除操作"
                logger.error(error_msg)
                raise ServiceError(error_msg)
            
        except Exception as e:
            error_msg = f"删除文档失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def get_document_count(self, filter_dict: Optional[Dict[str, Any]] = None) -> int:
        """获取文档数量
        
        Args:
            filter_dict: 过滤条件字典
            
        Returns:
            文档数量
            
        Raises:
            ServiceError: 获取失败时抛出
        """
        try:
            logger.info(f"获取文档数量: filter={filter_dict}")
            
            # 检查向量存储是否支持获取文档数量
            if hasattr(self.vector_store, "get_document_count"):
                # 获取文档数量
                count = self.vector_store.get_document_count(filter=filter_dict)
                
                logger.info(f"获取文档数量完成: {count}")
                
                return count
            else:
                # 如果不支持获取文档数量，返回-1表示未知
                logger.warning("向量存储不支持获取文档数量")
                return -1
            
        except Exception as e:
            error_msg = f"获取文档数量失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg) 