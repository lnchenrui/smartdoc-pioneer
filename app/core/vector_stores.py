"""向量存储模块

这个模块提供了向量存储功能，使用 Langchain 的向量存储。
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import time
import os

from langchain_community.vectorstores import (
    Chroma,
    FAISS,
    Elasticsearch,
    Qdrant
)
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.utils.logging.logger import get_logger
from app.utils.error_handler import VectorStoreError

logger = get_logger("core.vector_stores")

class VectorStoreService:
    """向量存储服务类，提供向量存储功能"""
    
    def __init__(
        self,
        embedding_service,
        store_type: str = "chroma",
        persist_directory: Optional[str] = "./vector_db",
        collection_name: str = "documents",
        **kwargs
    ):
        """初始化向量存储服务
        
        Args:
            embedding_service: 嵌入服务实例
            store_type: 向量存储类型，支持 "chroma", "faiss", "elasticsearch", "qdrant"
            persist_directory: 持久化目录，仅对本地存储有效
            collection_name: 集合名称
            **kwargs: 其他参数
        """
        self.embedding_service = embedding_service
        self.store_type = store_type
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.kwargs = kwargs
        
        # 获取嵌入模型
        self.embeddings = embedding_service.get_embedding_model()
        
        # 初始化向量存储
        self._initialize_vector_store()
        
        logger.info(f"向量存储服务初始化完成，类型: {store_type}, 集合: {collection_name}")
    
    def _initialize_vector_store(self):
        """初始化向量存储"""
        try:
            # 根据存储类型创建向量存储
            if self.store_type.lower() == "chroma":
                # 确保目录存在
                if self.persist_directory:
                    os.makedirs(self.persist_directory, exist_ok=True)
                
                # 创建或加载 Chroma 向量存储
                self.vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory,
                    **self.kwargs
                )
            
            elif self.store_type.lower() == "faiss":
                # 检查是否存在已保存的索引
                index_path = os.path.join(self.persist_directory, f"{self.collection_name}.faiss")
                if os.path.exists(index_path):
                    # 加载已有索引
                    self.vector_store = FAISS.load_local(
                        folder_path=self.persist_directory,
                        embeddings=self.embeddings,
                        index_name=self.collection_name,
                        **self.kwargs
                    )
                else:
                    # 创建新索引
                    self.vector_store = FAISS(
                        embedding_function=self.embeddings,
                        **self.kwargs
                    )
            
            elif self.store_type.lower() == "elasticsearch":
                # 创建 Elasticsearch 向量存储
                self.vector_store = Elasticsearch(
                    embedding=self.embeddings,
                    index_name=self.collection_name,
                    **self.kwargs
                )
            
            elif self.store_type.lower() == "qdrant":
                # 创建 Qdrant 向量存储
                self.vector_store = Qdrant(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    **self.kwargs
                )
            
            else:
                raise ValueError(f"不支持的向量存储类型: {self.store_type}")
            
        except Exception as e:
            error_msg = f"初始化向量存储失败: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    def add_documents(self, documents: List[Document]) -> None:
        """添加文档到向量存储
        
        Args:
            documents: 文档列表
            
        Raises:
            VectorStoreError: 添加文档失败时抛出
        """
        try:
            if not documents:
                logger.warning("没有文档需要添加")
                return
            
            start_time = time.time()
            
            # 添加文档
            self.vector_store.add_documents(documents)
            
            # 如果是本地存储，持久化
            if self.store_type.lower() in ["chroma", "faiss"] and self.persist_directory:
                if hasattr(self.vector_store, "persist"):
                    self.vector_store.persist()
            
            elapsed = time.time() - start_time
            logger.info(f"添加文档完成，文档数量: {len(documents)}, 耗时: {elapsed:.2f}秒")
            
        except Exception as e:
            error_msg = f"添加文档失败: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """添加文本到向量存储
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
            
        Raises:
            VectorStoreError: 添加文本失败时抛出
        """
        try:
            if not texts:
                logger.warning("没有文本需要添加")
                return
            
            start_time = time.time()
            
            # 添加文本
            self.vector_store.add_texts(texts, metadatas=metadatas)
            
            # 如果是本地存储，持久化
            if self.store_type.lower() in ["chroma", "faiss"] and self.persist_directory:
                if hasattr(self.vector_store, "persist"):
                    self.vector_store.persist()
            
            elapsed = time.time() - start_time
            logger.info(f"添加文本完成，文本数量: {len(texts)}, 耗时: {elapsed:.2f}秒")
            
        except Exception as e:
            error_msg = f"添加文本失败: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """搜索相似文档
        
        Args:
            query: 查询文本
            top_k: 返回的最大结果数
            filter_dict: 过滤条件字典
            
        Returns:
            相似文档列表
            
        Raises:
            VectorStoreError: 搜索失败时抛出
        """
        try:
            start_time = time.time()
            
            # 创建查询的向量嵌入
            query_embedding = self.embedding_service.create_embedding(query)
            
            # 执行向量搜索
            search_kwargs = {"k": top_k}
            if filter_dict:
                search_kwargs["filter"] = filter_dict
            
            results = self.vector_store.similarity_search_by_vector(
                embedding=query_embedding,
                **search_kwargs
            )
            
            elapsed = time.time() - start_time
            logger.info(f"搜索完成，查询: '{query}', 找到结果数: {len(results)}, 耗时: {elapsed:.2f}秒")
            
            return results
            
        except Exception as e:
            error_msg = f"搜索失败: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    def similarity_search(
        self, 
        query: str, 
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """相似度搜索
        
        Args:
            query: 查询文本
            top_k: 返回的最大结果数
            filter_dict: 过滤条件字典
            
        Returns:
            相似文档列表
            
        Raises:
            VectorStoreError: 搜索失败时抛出
        """
        try:
            start_time = time.time()
            
            # 执行相似度搜索
            search_kwargs = {"k": top_k}
            if filter_dict:
                search_kwargs["filter"] = filter_dict
            
            results = self.vector_store.similarity_search(
                query=query,
                **search_kwargs
            )
            
            elapsed = time.time() - start_time
            logger.info(f"相似度搜索完成，查询: '{query}', 找到结果数: {len(results)}, 耗时: {elapsed:.2f}秒")
            
            return results
            
        except Exception as e:
            error_msg = f"相似度搜索失败: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    def hybrid_search(
        self, 
        query: str, 
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """混合搜索（向量 + 关键词）
        
        Args:
            query: 查询文本
            top_k: 返回的最大结果数
            filter_dict: 过滤条件字典
            
        Returns:
            相似文档列表
            
        Raises:
            VectorStoreError: 搜索失败时抛出
        """
        try:
            # 检查向量存储是否支持混合搜索
            if hasattr(self.vector_store, "hybrid_search"):
                start_time = time.time()
                
                # 执行混合搜索
                search_kwargs = {"k": top_k}
                if filter_dict:
                    search_kwargs["filter"] = filter_dict
                
                results = self.vector_store.hybrid_search(
                    query=query,
                    **search_kwargs
                )
                
                elapsed = time.time() - start_time
                logger.info(f"混合搜索完成，查询: '{query}', 找到结果数: {len(results)}, 耗时: {elapsed:.2f}秒")
                
                return results
            else:
                # 如果不支持混合搜索，退回到普通相似度搜索
                logger.warning(f"向量存储类型 {self.store_type} 不支持混合搜索，使用普通相似度搜索")
                return self.similarity_search(query, top_k, filter_dict)
            
        except Exception as e:
            error_msg = f"混合搜索失败: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    def clear(self) -> bool:
        """清空向量存储
        
        Returns:
            是否成功清空
            
        Raises:
            VectorStoreError: 清空失败时抛出
        """
        try:
            # 根据存储类型执行清空操作
            if self.store_type.lower() == "chroma":
                self.vector_store.delete_collection()
                self.vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory,
                    **self.kwargs
                )
            
            elif self.store_type.lower() == "faiss":
                # 对于FAISS，重新创建一个空索引
                self.vector_store = FAISS(
                    embedding_function=self.embeddings,
                    **self.kwargs
                )
                
                # 如果有持久化目录，删除旧索引文件
                if self.persist_directory:
                    index_path = os.path.join(self.persist_directory, f"{self.collection_name}.faiss")
                    if os.path.exists(index_path):
                        os.remove(index_path)
            
            elif self.store_type.lower() == "elasticsearch":
                # 对于Elasticsearch，删除并重建索引
                if hasattr(self.vector_store, "_client"):
                    self.vector_store._client.indices.delete(index=self.collection_name)
                self._initialize_vector_store()
            
            elif self.store_type.lower() == "qdrant":
                # 对于Qdrant，删除并重建集合
                if hasattr(self.vector_store, "_client"):
                    self.vector_store._client.delete_collection(collection_name=self.collection_name)
                self._initialize_vector_store()
            
            logger.info(f"清空向量存储完成，类型: {self.store_type}, 集合: {self.collection_name}")
            return True
            
        except Exception as e:
            error_msg = f"清空向量存储失败: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取向量存储统计信息
        
        Returns:
            统计信息字典
            
        Raises:
            VectorStoreError: 获取统计信息失败时抛出
        """
        try:
            stats = {
                "store_type": self.store_type,
                "collection_name": self.collection_name
            }
            
            # 根据存储类型获取统计信息
            if self.store_type.lower() == "chroma":
                if hasattr(self.vector_store, "_collection"):
                    count = self.vector_store._collection.count()
                    stats["document_count"] = count
            
            elif self.store_type.lower() == "faiss":
                if hasattr(self.vector_store, "index"):
                    count = self.vector_store.index.ntotal
                    stats["document_count"] = count
            
            elif self.store_type.lower() == "elasticsearch":
                if hasattr(self.vector_store, "_client"):
                    count_resp = self.vector_store._client.count(index=self.collection_name)
                    stats["document_count"] = count_resp.get("count", 0)
            
            elif self.store_type.lower() == "qdrant":
                if hasattr(self.vector_store, "_client"):
                    collection_info = self.vector_store._client.get_collection(collection_name=self.collection_name)
                    stats["document_count"] = collection_info.get("vectors_count", 0)
            
            logger.info(f"获取向量存储统计信息完成: {stats}")
            return stats
            
        except Exception as e:
            error_msg = f"获取向量存储统计信息失败: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    def as_retriever(
        self, 
        search_type: str = "similarity", 
        search_kwargs: Optional[Dict[str, Any]] = None
    ):
        """将向量存储转换为检索器
        
        Args:
            search_type: 搜索类型，支持 "similarity", "mmr", "similarity_score_threshold"
            search_kwargs: 搜索参数
            
        Returns:
            检索器实例
        """
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def get_vector_store(self) -> VectorStore:
        """获取向量存储实例
        
        Returns:
            向量存储实例
        """
        return self.vector_store 