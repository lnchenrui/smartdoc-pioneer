"""搜索服务模块

这个模块提供了文档搜索相关的服务功能，使用 Langchain 的搜索功能。
"""

from typing import List, Dict, Any, Optional, Union
import time

from langchain_core.documents import Document

from app.utils.logging.logger import get_logger
from app.utils.error_handler import ServiceError, ValidationError

logger = get_logger("services.search")

class SearchService:
    """搜索服务类，提供文档搜索功能"""
    
    def __init__(
        self,
        vector_store_service,
        embedding_service,
        config: Dict[str, Any]
    ):
        """初始化搜索服务
        
        Args:
            vector_store_service: 向量存储服务实例
            embedding_service: 嵌入服务实例
            config: 配置字典
        """
        self.vector_store_service = vector_store_service
        self.embedding_service = embedding_service
        self.config = config
        
        # 获取搜索配置
        self.default_top_k = config.get('search', {}).get('default_top_k', 5)
        self.default_search_type = config.get('search', {}).get('default_search_type', 'similarity')
        
        logger.debug("搜索服务初始化完成")
    
    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        search_type: Optional[str] = None
    ) -> List[Document]:
        """搜索文档
        
        Args:
            query: 搜索查询
            top_k: 返回的最大结果数
            filter_dict: 过滤条件字典，如{"source": "某来源", "type": "某类型"}
            search_type: 搜索类型，支持 "similarity", "mmr"
            
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
            
            if search_type is None:
                search_type = self.default_search_type
            
            logger.info(f"执行搜索: query='{query}', top_k={top_k}, search_type={search_type}")
            
            # 创建检索参数
            search_kwargs = {"k": top_k}
            if filter_dict:
                search_kwargs["filter"] = filter_dict
            
            # 创建检索器
            retriever = self.vector_store_service.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
            
            # 执行检索
            start_time = time.time()
            results = retriever.invoke(query)
            elapsed = time.time() - start_time
            
            # 记录搜索结果
            logger.info(f"搜索完成，找到 {len(results)} 个结果，耗时: {elapsed:.2f}秒")
            
            return results
            
        except Exception as e:
            error_msg = f"搜索失败: {str(e)}"
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
            
            # 执行混合搜索
            start_time = time.time()
            results = self.vector_store_service.hybrid_search(
                query=query,
                top_k=top_k,
                filter_dict=filter_dict
            )
            elapsed = time.time() - start_time
            
            # 记录搜索结果
            logger.info(f"混合搜索完成，找到 {len(results)} 个结果，耗时: {elapsed:.2f}秒")
            
            return results
            
        except Exception as e:
            error_msg = f"混合搜索失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def keyword_search(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """关键词搜索
        
        使用关键词匹配进行搜索。
        
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
            
            logger.info(f"执行关键词搜索: query='{query}', top_k={top_k}")
            
            # 检查向量存储是否支持关键词搜索
            if hasattr(self.vector_store_service.get_vector_store(), "similarity_search"):
                # 执行关键词搜索
                start_time = time.time()
                results = self.vector_store_service.similarity_search(
                    query=query,
                    top_k=top_k,
                    filter_dict=filter_dict
                )
                elapsed = time.time() - start_time
                
                # 记录搜索结果
                logger.info(f"关键词搜索完成，找到 {len(results)} 个结果，耗时: {elapsed:.2f}秒")
                
                return results
            else:
                # 如果不支持关键词搜索，退回到普通搜索
                logger.warning("向量存储不支持关键词搜索，使用普通搜索")
                return self.search(query, top_k, filter_dict)
            
        except Exception as e:
            error_msg = f"关键词搜索失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def format_search_results(self, results: List[Document]) -> List[Dict[str, Any]]:
        """格式化搜索结果
        
        Args:
            results: 搜索结果列表
            
        Returns:
            格式化后的搜索结果列表
        """
        formatted_results = []
        
        for doc in results:
            result = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            formatted_results.append(result)
        
        return formatted_results 