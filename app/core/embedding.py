"""向量嵌入模块

这个模块提供了文本向量嵌入功能，支持多种嵌入模型。
"""

import time
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import requests

# 尝试导入voyageai，如果不可用则跳过
try:
    import voyageai
    VOYAGE_AVAILABLE = True
except ImportError:
    VOYAGE_AVAILABLE = False
    

from app.utils.logging.logger import get_logger
from app.utils.config.loader import get_config

logger = get_logger("core.embedding")


class EmbeddingClient(ABC):
    """嵌入客户端抽象基类"""
    
    @abstractmethod
    def get_embedding(self, texts: List[str]) -> List[List[float]]:
        """获取文本的向量嵌入
        
        Args:
            texts: 文本列表
            
        Returns:
            向量嵌入列表
        """
        pass


class OpenAIEmbeddingClient(EmbeddingClient):
    """OpenAI嵌入客户端"""
    
    def __init__(self):
        """初始化OpenAI嵌入客户端"""
        # 从配置中获取API密钥和端点
        self.api_key = get_config().get('embedding.openai.api_key', '')
        self.endpoint = get_config().get('embedding.openai.endpoint', '')
        self.api_version = get_config().get('embedding.openai.api_version', '2023-05-15')
        self.url = f"{self.endpoint}?api-version={self.api_version}"
        logger.info("OpenAI嵌入客户端初始化完成")
    
    def get_embedding(self, texts: List[str]) -> List[List[float]]:
        """获取文本的向量嵌入
        
        Args:
            texts: 文本列表
            
        Returns:
            向量嵌入列表
        """
        total_tokens = sum(len(text) for text in texts)
        logger.info(f"开始调用OpenAI嵌入服务，文本数: {len(texts)}, 总token数: {total_tokens}")
        
        try:
            headers = {
                "api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                url=self.url,
                headers=headers,
                json={"input": texts}
            )
            
            if response.status_code != 200:
                logger.error(f"OpenAI嵌入服务调用失败，状态码: {response.status_code}, 响应: {response.text}")
                return []
            
            result = response.json()
            embeddings = []
            
            for item in result.get("data", []):
                embedding = item.get("embedding", [])
                embeddings.append(embedding)
            
            logger.info(f"OpenAI嵌入服务调用成功，获取到 {len(embeddings)} 个嵌入向量")
            return embeddings
            
        except Exception as e:
            logger.error(f"OpenAI嵌入服务调用异常: {str(e)}")
            return []
    
    def embedding(self, texts: List[str]) -> Dict[str, Any]:
        """兼容旧版API的嵌入方法
        
        Args:
            texts: 文本列表
            
        Returns:
            包含嵌入向量的字典
        """
        try:
            headers = {
                "api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                url=self.url,
                headers=headers,
                json={"input": texts}
            )
            
            return response.json()
            
        except Exception as e:
            logger.error(f"OpenAI嵌入服务调用异常: {str(e)}")
            return {"data": []}


class DummyEmbeddingClient(EmbeddingClient):
    """虚拟嵌入客户端，用于在其他嵌入服务不可用时提供基本功能"""
    
    def __init__(self, embedding_dim: int = 1536):
        """初始化虚拟嵌入客户端
        
        Args:
            embedding_dim: 嵌入向量维度
        """
        self.embedding_dim = embedding_dim
        logger.info(f"虚拟嵌入客户端初始化完成，向量维度: {embedding_dim}")
    
    def get_embedding(self, texts: List[str]) -> List[List[float]]:
        """获取虚拟嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            随机生成的嵌入向量列表
        """
        logger.info(f"生成虚拟嵌入向量，文本数: {len(texts)}")
        
        # 为每个文本生成随机向量
        embeddings = []
        for text in texts:
            # 生成随机向量并归一化
            vector = [random.uniform(-1, 1) for _ in range(self.embedding_dim)]
            # 简单归一化
            magnitude = sum(x*x for x in vector) ** 0.5
            if magnitude > 0:
                vector = [x/magnitude for x in vector]
            embeddings.append(vector)
        
        return embeddings


# 如果voyageai可用，则定义VoyageEmbeddingClient类
if VOYAGE_AVAILABLE:
    class VoyageEmbeddingClient(EmbeddingClient):
        """Voyage嵌入客户端"""
        
        def __init__(self):
            """初始化Voyage嵌入客户端"""
            # 从配置中获取API密钥和模型
            self.api_key = get_config().get('embedding.voyage.api_key', '')
            self.model = get_config().get('embedding.voyage.model', 'voyage-3-large')
            self.client = voyageai.Client(api_key=self.api_key)
            logger.info(f"Voyage嵌入客户端初始化完成，模型: {self.model}")
        
        def get_embedding(self, texts: List[str]) -> List[List[float]]:
            """获取文本的向量嵌入
            
            Args:
                texts: 文本列表
                
            Returns:
                向量嵌入列表
            """
            total_tokens = sum(len(text) for text in texts)
            logger.info(f"开始调用Voyage嵌入服务，文本数: {len(texts)}, 总token数: {total_tokens}")
            
            try:
                result = self.client.embed(
                    texts=texts,
                    model=self.model,
                    input_type="document",
                    truncation=False
                )
                
                embeddings = result.embeddings
                logger.info(f"Voyage嵌入服务调用成功，获取到 {len(embeddings)} 个嵌入向量")
                return embeddings
                
            except Exception as e:
                logger.error(f"Voyage嵌入服务调用异常: {str(e)}")
                return []


def batch_embedding(texts: List[str], batch_size: int = 10, client: Optional[EmbeddingClient] = None) -> List[List[float]]:
    """批量获取文本的向量嵌入
    
    Args:
        texts: 文本列表
        batch_size: 每批处理的文本数量
        client: 嵌入客户端，如果为None则使用默认客户端
        
    Returns:
        向量嵌入列表
    """
    if not texts:
        return []
    
    # 使用默认客户端
    if client is None:
        client = openai_embedding_client
    
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    logger.info(f"开始批量嵌入处理，总文本数: {len(texts)}, 批次大小: {batch_size}, 总批次: {total_batches}")
    start_time = time.time()
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        logger.info(f"处理第 {batch_num}/{total_batches} 批，包含 {len(batch_texts)} 个文本")
        batch_start_time = time.time()
        
        # 获取当前批次的嵌入向量
        batch_embeddings = client.get_embedding(batch_texts)
        all_embeddings.extend(batch_embeddings)
        
        batch_elapsed = time.time() - batch_start_time
        logger.info(f"第 {batch_num} 批处理完成，耗时: {batch_elapsed:.2f}秒")
        
        # 添加延迟，避免API限制
        if i + batch_size < len(texts):
            time.sleep(0.5)
    
    total_elapsed = time.time() - start_time
    logger.info(f"批量嵌入处理完成，总耗时: {total_elapsed:.2f}秒，平均每批耗时: {total_elapsed/total_batches:.2f}秒")
    
    return all_embeddings


# 创建默认嵌入客户端实例，方便直接导入使用
openai_embedding_client = OpenAIEmbeddingClient()

# 如果voyageai可用，则创建Voyage嵌入客户端，否则使用虚拟客户端
if VOYAGE_AVAILABLE:
    try:
        voyage_embedding_client = VoyageEmbeddingClient()
    except Exception as e:
        logger.warning(f"Voyage嵌入客户端初始化失败: {str(e)}，使用虚拟嵌入客户端")
        voyage_embedding_client = DummyEmbeddingClient()
else:
    logger.warning("Voyage嵌入服务不可用，使用虚拟嵌入客户端")
    voyage_embedding_client = DummyEmbeddingClient() 