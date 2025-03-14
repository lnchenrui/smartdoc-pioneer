"""RAG服务模块

这个模块提供了检索增强生成(RAG)相关的服务功能。
"""

from app.di.container import container
from app.utils.logging.logger import get_logger
from app.utils.error_handler import ServiceError, ValidationError

logger = get_logger("app.services.rag")

class RAGService:
    """RAG服务类
    
    提供检索增强生成功能。
    """
    
    def __init__(self):
        """初始化RAG服务
        
        初始化搜索服务和配置。
        """
        self.search_service = container.search_service()
        self.config = container.config()
        self.top_k = self.config.get('rag', {}).get('top_k', 5)
        self.search_type = self.config.get('rag', {}).get('search_type', 'vector')
        logger.debug("RAG服务初始化完成")
    
    def retrieve(self, query, top_k=None, filter_dict=None):
        """检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回的最大结果数，如果为None则使用配置中的值
            filter_dict: 过滤条件字典
            
        Returns:
            检索到的文档列表
        """
        try:
            if not query:
                raise ValidationError("查询不能为空")
            
            # 使用配置中的top_k如果未指定
            if top_k is None:
                top_k = self.top_k
            
            logger.info(f"RAG检索: query='{query}', top_k={top_k}, search_type={self.search_type}")
            
            # 根据配置的搜索类型执行检索
            if self.search_type == 'hybrid':
                results = self.search_service.hybrid_search(query, top_k, filter_dict)
            elif self.search_type == 'keyword':
                results = self.search_service.keyword_search(query, top_k, filter_dict)
            else:  # 默认使用向量搜索
                results = self.search_service.search(query, top_k, filter_dict)
            
            # 记录检索结果
            logger.info(f"RAG检索完成，找到 {len(results)} 个结果")
            
            return results
        except Exception as e:
            logger.error(f"RAG检索失败: {str(e)}", exc_info=True)
            raise ServiceError(f"RAG服务错误: {str(e)}")
    
    def rerank(self, query, documents):
        """重新排序检索结果
        
        Args:
            query: 查询文本
            documents: 检索到的文档列表
            
        Returns:
            重新排序后的文档列表
        """
        try:
            if not documents:
                return []
            
            logger.info(f"RAG重排序: {len(documents)} 个文档")
            
            # 简单实现，未来可以添加更复杂的重排序逻辑
            # 例如使用交叉编码器或其他重排序模型
            
            # 当前仅返回原始文档，保持原有顺序
            return documents
        except Exception as e:
            logger.error(f"RAG重排序失败: {str(e)}", exc_info=True)
            # 如果重排序失败，返回原始文档
            return documents 