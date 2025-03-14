"""索引服务模块

这个模块提供了文档索引相关的服务功能。
"""

import os
from pathlib import Path
from app.di.container import container
from app.utils.logging.logger import get_logger
from app.utils.error_handler import ServiceError, ValidationError

logger = get_logger("app.services.index")

class IndexService:
    """索引服务类
    
    提供文档索引功能。
    """
    
    def __init__(self):
        """初始化索引服务
        
        初始化文档处理器和向量存储。
        """
        self.document_processor = container.document_processor()
        self.vector_store = container.vector_store()
        self.embedding_service = container.embedding_service()
        self.config = container.config()
        logger.debug("索引服务初始化完成")
    
    def index_documents(self, path=None, recursive=True, file_types=None):
        """索引文档
        
        Args:
            path: 文档路径，如果为None则使用配置中的路径
            recursive: 是否递归索引子目录
            file_types: 要索引的文件类型列表，如[".txt", ".pdf", ".docx"]
            
        Returns:
            (已索引文档数, 失败文档数)
        """
        try:
            # 如果未指定路径，使用配置中的路径
            if path is None:
                path = self.config.get('document', {}).get('datasets_dir')
                if not path:
                    raise ValidationError("未指定文档路径，且配置中未设置默认路径")
            
            # 确保路径存在
            if not os.path.exists(path):
                raise ValidationError(f"指定的路径不存在: {path}")
            
            logger.info(f"开始索引文档: path='{path}', recursive={recursive}")
            
            # 加载文档
            documents = self.document_processor.load_documents(
                path=path,
                recursive=recursive,
                file_types=file_types
            )
            
            if not documents:
                logger.warning(f"在路径 '{path}' 中未找到文档")
                return 0, 0
            
            logger.info(f"加载了 {len(documents)} 个文档，开始创建向量嵌入")
            
            # 创建向量嵌入并存储
            indexed_count = 0
            failed_count = 0
            
            # 批量处理文档，避免一次处理过多
            batch_size = 10
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                try:
                    # 创建向量嵌入
                    embeddings = self.embedding_service.create_embeddings([doc.content for doc in batch])
                    
                    # 存储到向量数据库
                    self.vector_store.add_documents(batch, embeddings)
                    
                    indexed_count += len(batch)
                    logger.info(f"已索引 {indexed_count}/{len(documents)} 个文档")
                except Exception as e:
                    logger.error(f"批次 {i//batch_size + 1} 索引失败: {str(e)}", exc_info=True)
                    failed_count += len(batch)
            
            logger.info(f"索引完成: 成功={indexed_count}, 失败={failed_count}")
            return indexed_count, failed_count
            
        except Exception as e:
            logger.error(f"索引文档失败: {str(e)}", exc_info=True)
            raise ServiceError(f"索引服务错误: {str(e)}")
    
    def clear_index(self):
        """清空索引
        
        Returns:
            是否成功清空
        """
        try:
            logger.info("开始清空索引")
            result = self.vector_store.clear()
            logger.info("索引已清空")
            return result
        except Exception as e:
            logger.error(f"清空索引失败: {str(e)}", exc_info=True)
            raise ServiceError(f"清空索引失败: {str(e)}")
    
    def get_index_stats(self):
        """获取索引统计信息
        
        Returns:
            索引统计信息
        """
        try:
            logger.info("获取索引统计信息")
            stats = self.vector_store.get_stats()
            return stats
        except Exception as e:
            logger.error(f"获取索引统计信息失败: {str(e)}", exc_info=True)
            raise ServiceError(f"获取索引统计信息失败: {str(e)}") 