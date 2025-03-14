"""嵌入服务模块

这个模块提供了文本嵌入相关的服务功能，使用 Langchain 的最新嵌入功能。
"""

from typing import List, Dict, Any, Optional, Union
import time
import numpy as np

from langchain_core.embeddings import Embeddings

from app.utils.logging.logger import get_logger
from app.utils.error_handler import ServiceError, ValidationError

logger = get_logger("services.embedding")

class EmbeddingService:
    """嵌入服务类，提供文本嵌入功能"""
    
    def __init__(
        self,
        embedding_model: Embeddings,
        config: Dict[str, Any]
    ):
        """初始化嵌入服务
        
        Args:
            embedding_model: 嵌入模型实例
            config: 配置字典
        """
        self.embedding_model = embedding_model
        self.config = config
        
        # 获取嵌入配置
        self.batch_size = config.get('embedding', {}).get('batch_size', 32)
        
        logger.debug("嵌入服务初始化完成")
    
    def get_embedding_model(self) -> Embeddings:
        """获取嵌入模型实例
        
        Returns:
            嵌入模型实例
        """
        return self.embedding_model
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询文本
        
        Args:
            text: 查询文本
            
        Returns:
            嵌入向量
            
        Raises:
            ServiceError: 嵌入失败时抛出
        """
        try:
            if not text:
                raise ValidationError("查询文本不能为空")
            
            logger.debug(f"嵌入查询文本: '{text[:50]}...' (长度: {len(text)})")
            
            # 嵌入文本
            start_time = time.time()
            embedding = self.embedding_model.embed_query(text)
            elapsed = time.time() - start_time
            
            logger.debug(f"查询文本嵌入完成，向量维度: {len(embedding)}，耗时: {elapsed:.2f}秒")
            
            return embedding
            
        except Exception as e:
            error_msg = f"嵌入查询文本失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档文本
        
        Args:
            texts: 文档文本列表
            
        Returns:
            嵌入向量列表
            
        Raises:
            ServiceError: 嵌入失败时抛出
        """
        try:
            if not texts:
                logger.warning("没有文档需要嵌入")
                return []
            
            logger.info(f"嵌入 {len(texts)} 个文档")
            
            # 嵌入文档
            start_time = time.time()
            embeddings = self.embedding_model.embed_documents(texts)
            elapsed = time.time() - start_time
            
            logger.info(f"文档嵌入完成，嵌入了 {len(embeddings)} 个文档，每个向量维度: {len(embeddings[0])}，耗时: {elapsed:.2f}秒")
            
            return embeddings
            
        except Exception as e:
            error_msg = f"嵌入文档失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def embed_documents_in_batches(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """分批嵌入多个文档文本
        
        Args:
            texts: 文档文本列表
            batch_size: 批处理大小
            
        Returns:
            嵌入向量列表
            
        Raises:
            ServiceError: 嵌入失败时抛出
        """
        try:
            if not texts:
                logger.warning("没有文档需要嵌入")
                return []
            
            # 使用默认值如果未指定
            if batch_size is None:
                batch_size = self.batch_size
            
            logger.info(f"分批嵌入 {len(texts)} 个文档，批大小: {batch_size}")
            
            # 分批处理
            all_embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            start_time = time.time()
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_num = i // batch_size + 1
                
                logger.debug(f"处理批次 {batch_num}/{total_batches}，包含 {len(batch_texts)} 个文档")
                
                batch_start_time = time.time()
                batch_embeddings = self.embedding_model.embed_documents(batch_texts)
                batch_elapsed = time.time() - batch_start_time
                
                logger.debug(f"批次 {batch_num} 处理完成，耗时: {batch_elapsed:.2f}秒")
                
                all_embeddings.extend(batch_embeddings)
            
            total_elapsed = time.time() - start_time
            logger.info(f"所有文档嵌入完成，共 {len(all_embeddings)} 个嵌入向量，总耗时: {total_elapsed:.2f}秒")
            
            return all_embeddings
            
        except Exception as e:
            error_msg = f"分批嵌入文档失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """计算两个嵌入向量之间的余弦相似度
        
        Args:
            embedding1: 第一个嵌入向量
            embedding2: 第二个嵌入向量
            
        Returns:
            余弦相似度
            
        Raises:
            ServiceError: 计算失败时抛出
        """
        try:
            # 转换为numpy数组
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # 计算余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            similarity = dot_product / (norm1 * norm2)
            
            return float(similarity)
            
        except Exception as e:
            error_msg = f"计算相似度失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg) 