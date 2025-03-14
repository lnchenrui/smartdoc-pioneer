"""嵌入模型模块

这个模块提供了文本嵌入功能，使用 Langchain 的嵌入模型。
"""

from typing import List, Dict, Any, Optional, Union
import time

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    CohereEmbeddings,
    OllamaEmbeddings
)
from langchain_core.embeddings import Embeddings

from app.utils.logging.logger import get_logger
from app.utils.error_handler import EmbeddingError

logger = get_logger("core.embeddings")

class EmbeddingService:
    """嵌入服务类，提供文本嵌入功能"""
    
    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs
    ):
        """初始化嵌入服务
        
        Args:
            provider: 嵌入提供商，支持 "openai", "huggingface", "cohere", "ollama"
            model_name: 模型名称
            api_key: API密钥
            api_base: API基础URL
            **kwargs: 其他参数
        """
        self.provider = provider
        self.model_name = model_name
        
        try:
            # 根据提供商创建嵌入模型
            if provider.lower() == "openai":
                self.embedding_model = OpenAIEmbeddings(
                    model=model_name,
                    openai_api_key=api_key,
                    openai_api_base=api_base,
                    **kwargs
                )
            elif provider.lower() == "huggingface":
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=model_name,
                    **kwargs
                )
            elif provider.lower() == "cohere":
                self.embedding_model = CohereEmbeddings(
                    model=model_name,
                    cohere_api_key=api_key,
                    **kwargs
                )
            elif provider.lower() == "ollama":
                self.embedding_model = OllamaEmbeddings(
                    model=model_name,
                    base_url=api_base,
                    **kwargs
                )
            else:
                raise ValueError(f"不支持的嵌入提供商: {provider}")
            
            logger.info(f"嵌入服务初始化完成，提供商: {provider}, 模型: {model_name}")
        except Exception as e:
            error_msg = f"初始化嵌入服务失败: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg)
    
    def create_embedding(self, text: str) -> List[float]:
        """创建单个文本的嵌入向量
        
        Args:
            text: 文本内容
            
        Returns:
            嵌入向量
            
        Raises:
            EmbeddingError: 创建嵌入失败时抛出
        """
        try:
            start_time = time.time()
            
            # 创建嵌入
            embedding = self.embedding_model.embed_query(text)
            
            elapsed = time.time() - start_time
            logger.debug(f"创建嵌入完成，文本长度: {len(text)}, 向量维度: {len(embedding)}, 耗时: {elapsed:.2f}秒")
            
            return embedding
        except Exception as e:
            error_msg = f"创建嵌入失败: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg)
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量创建文本的嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
            
        Raises:
            EmbeddingError: 创建嵌入失败时抛出
        """
        try:
            if not texts:
                logger.warning("没有文本需要创建嵌入")
                return []
            
            start_time = time.time()
            
            # 批量创建嵌入
            embeddings = self.embedding_model.embed_documents(texts)
            
            elapsed = time.time() - start_time
            logger.info(f"批量创建嵌入完成，文本数量: {len(texts)}, 耗时: {elapsed:.2f}秒")
            
            return embeddings
        except Exception as e:
            error_msg = f"批量创建嵌入失败: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg)
    
    def get_embedding_model(self) -> Embeddings:
        """获取嵌入模型实例
        
        Returns:
            嵌入模型实例
        """
        return self.embedding_model 