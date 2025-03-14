"""搜索服务模块

这个模块提供了文档搜索相关的服务功能。
"""

from app.di.container import container
from app.utils.logging.logger import get_logger
from app.utils.error_handler import ServiceError, ValidationError

logger = get_logger("app.services.search")

class SearchService:
    """搜索服务类
    
    提供文档搜索功能。
    """
    
    def __init__(self):
        """初始化搜索服务
        
        初始化向量存储和嵌入服务。
        """
        self.vector_store = container.vector_store()
        self.embedding_service = container.embedding_service()
        self.config = container.config()
        logger.debug("搜索服务初始化完成")
    
    def search(self, query, top_k=5, filter_dict=None):
        """搜索文档
        
        Args:
            query: 搜索查询
            top_k: 返回的最大结果数
            filter_dict: 过滤条件字典，如{"source": "某来源", "type": "某类型"}
            
        Returns:
            搜索结果列表
        """
        try:
            if not query:
                raise ValidationError("搜索查询不能为空")
            
            logger.info(f"执行搜索: query='{query}', top_k={top_k}")
            
            # 创建查询的向量嵌入
            query_embedding = self.embedding_service.create_embedding(query)
            
            # 执行向量搜索
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_dict=filter_dict
            )
            
            # 记录搜索结果
            logger.info(f"搜索完成，找到 {len(results)} 个结果")
            
            return results
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}", exc_info=True)
            raise ServiceError(f"搜索服务错误: {str(e)}")
    
    def hybrid_search(self, query, top_k=5, filter_dict=None):
        """混合搜索
        
        结合向量搜索和关键词搜索的混合搜索方法。
        
        Args:
            query: 搜索查询
            top_k: 返回的最大结果数
            filter_dict: 过滤条件字典
            
        Returns:
            搜索结果列表
        """
        try:
            if not query:
                raise ValidationError("搜索查询不能为空")
            
            logger.info(f"执行混合搜索: query='{query}', top_k={top_k}")
            
            # 创建查询的向量嵌入
            query_embedding = self.embedding_service.create_embedding(query)
            
            # 执行混合搜索
            results = self.vector_store.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                filter_dict=filter_dict
            )
            
            # 记录搜索结果
            logger.info(f"混合搜索完成，找到 {len(results)} 个结果")
            
            return results
        except Exception as e:
            logger.error(f"混合搜索失败: {str(e)}", exc_info=True)
            raise ServiceError(f"搜索服务错误: {str(e)}")
    
    def keyword_search(self, query, top_k=5, filter_dict=None):
        """关键词搜索
        
        使用关键词匹配进行搜索。
        
        Args:
            query: 搜索查询
            top_k: 返回的最大结果数
            filter_dict: 过滤条件字典
            
        Returns:
            搜索结果列表
        """
        try:
            if not query:
                raise ValidationError("搜索查询不能为空")
            
            logger.info(f"执行关键词搜索: query='{query}', top_k={top_k}")
            
            # 执行关键词搜索
            results = self.vector_store.keyword_search(
                query=query,
                top_k=top_k,
                filter_dict=filter_dict
            )
            
            # 记录搜索结果
            logger.info(f"关键词搜索完成，找到 {len(results)} 个结果")
            
            return results
        except Exception as e:
            logger.error(f"关键词搜索失败: {str(e)}", exc_info=True)
            raise ServiceError(f"搜索服务错误: {str(e)}") 