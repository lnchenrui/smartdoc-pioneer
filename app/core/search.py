"""搜索模块

这个模块提供了文档搜索功能，支持多种搜索引擎。
"""

import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch, helpers

from app.utils.logging.logger import get_logger
from app.utils.config.loader import get_config

logger = get_logger("core.search")


class SearchClient(ABC):
    """搜索客户端抽象基类"""
    
    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 5) -> Dict[str, Any]:
        """向量相似度搜索
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            
        Returns:
            搜索结果
        """
        pass
    
    @abstractmethod
    def index_document(self, document: Dict[str, Any]) -> bool:
        """索引文档
        
        Args:
            document: 文档数据
            
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    def bulk_index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量索引文档
        
        Args:
            documents: 文档数据列表
            
        Returns:
            索引结果
        """
        pass


class ElasticsearchClient(SearchClient):
    """Elasticsearch客户端"""
    
    def __init__(self):
        """初始化Elasticsearch客户端"""
        # 从配置中获取连接信息
        hosts = get_config().get('elasticsearch.hosts', ["http://localhost:9200"])
        username = get_config().get('elasticsearch.username', '')
        password = get_config().get('elasticsearch.password', '')
        self.index_name = get_config().get('elasticsearch.index_name', 'wx_dev_doc_v4')
        self.embedding_dim = get_config().get('elasticsearch.embedding_dim', 1536)
        self.similarity_threshold = get_config().get('elasticsearch.similarity_threshold', 0.7)
        self.max_results = get_config().get('elasticsearch.max_results', 5)
        
        # 创建ES客户端
        if username and password:
            self.es = Elasticsearch(hosts, basic_auth=(username, password))
        else:
            self.es = Elasticsearch(hosts)
        
        logger.info(f"Elasticsearch客户端初始化完成，索引: {self.index_name}")
        
        # 确保索引存在
        self._ensure_index()
    
    def _ensure_index(self):
        """确保索引存在，如果不存在则创建"""
        try:
            if not self.es.indices.exists(index=self.index_name):
                logger.info(f"创建索引: {self.index_name}")
                
                # 定义索引映射
                mappings = {
                    "properties": {
                        "content": {"type": "text", "analyzer": "standard"},
                        "title": {"type": "text", "analyzer": "standard"},
                        "source": {"type": "keyword"},
                        "url": {"type": "keyword"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": self.embedding_dim,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "created_at": {"type": "date"},
                        "updated_at": {"type": "date"}
                    }
                }
                
                # 创建索引
                self.es.indices.create(
                    index=self.index_name,
                    mappings=mappings,
                    settings={
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    }
                )
                logger.info(f"索引 {self.index_name} 创建成功")
            else:
                logger.info(f"索引 {self.index_name} 已存在")
                
        except Exception as e:
            logger.error(f"确保索引存在时出错: {str(e)}")
    
    def search(self, query_vector: List[float], top_k: int = 5) -> Dict[str, Any]:
        """搜索相似文档
        
        Args:
            query_vector: 查询向量
            top_k: 返回的最相似文档数量
            
        Returns:
            搜索结果字典
        """
        try:
            # 构建KNN查询
            knn_query = {
                "knn": {
                    "field": "content_vector",
                    "query_vector": query_vector,
                    "k": top_k,
                    "num_candidates": top_k * 2
                }
            }
            
            # 执行查询
            response = self.es.search(
                index=self.index_name,
                query=knn_query,
                size=top_k,
                _source=["content", "title", "source", "url", "score"]
            )
            
            # 处理结果
            hits = response["hits"]["hits"]
            results = []
            
            for hit in hits:
                score = hit["_score"]
                source = hit["_source"]
                
                result = {
                    "content": source["content"],
                    "title": source["title"],
                    "source": source["source"],
                    "url": source["url"],
                    "score": score
                }
                results.append(result)
            
            logger.info(f"KNN搜索完成，找到 {len(results)} 个相似文档")
            return {
                "total": len(results),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"KNN搜索出错: {str(e)}")
            return {
                "total": 0,
                "results": []
            }
    
    def index_document(self, document: Dict[str, Any]) -> bool:
        """索引单个文档
        
        Args:
            document: 文档数据
            
        Returns:
            是否成功
        """
        if not document:
            logger.error("文档为空")
            return False
        
        logger.info(f"索引文档: {document.get('title', '未知标题')}")
        
        try:
            # 添加时间戳
            if 'created_at' not in document:
                document['created_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            document['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            
            # 索引文档
            result = self.es.index(index=self.index_name, document=document)
            
            if result.get('result') in ['created', 'updated']:
                logger.info(f"文档索引成功，ID: {result.get('_id')}")
                return True
            else:
                logger.error(f"文档索引失败: {result}")
                return False
                
        except Exception as e:
            logger.error(f"索引文档时出错: {str(e)}")
            return False
    
    def bulk_index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量索引文档
        
        Args:
            documents: 文档数据列表
            
        Returns:
            索引结果
        """
        if not documents:
            logger.error("文档列表为空")
            return {"errors": True, "items": []}
        
        logger.info(f"批量索引文档，数量: {len(documents)}")
        start_time = time.time()
        
        try:
            # 准备批量索引数据
            actions = []
            timestamp = time.strftime('%Y-%m-%dT%H:%M:%S')
            
            for doc in documents:
                # 添加时间戳
                if 'created_at' not in doc:
                    doc['created_at'] = timestamp
                doc['updated_at'] = timestamp
                
                # 添加索引操作
                action = {
                    "_index": self.index_name,
                    "_source": doc
                }
                actions.append(action)
            
            # 执行批量索引
            result = helpers.bulk(self.es, actions)
            
            elapsed = time.time() - start_time
            logger.info(f"批量索引完成，成功: {result[0]}，失败: {len(result[1]) if len(result) > 1 else 0}，耗时: {elapsed:.2f}秒")
            
            return {"success": result[0], "errors": len(result[1]) if len(result) > 1 else 0}
            
        except Exception as e:
            logger.error(f"批量索引文档时出错: {str(e)}")
            return {"errors": True, "items": []}
    
    def delete_document(self, doc_id: str) -> bool:
        """删除文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            是否成功
        """
        logger.info(f"删除文档，ID: {doc_id}")
        
        try:
            result = self.es.delete(index=self.index_name, id=doc_id)
            
            if result.get('result') == 'deleted':
                logger.info(f"文档删除成功，ID: {doc_id}")
                return True
            else:
                logger.error(f"文档删除失败: {result}")
                return False
                
        except Exception as e:
            logger.error(f"删除文档时出错: {str(e)}")
            return False
    
    def delete_all_documents(self) -> bool:
        """删除所有文档
        
        Returns:
            是否成功
        """
        logger.info(f"删除索引 {self.index_name} 中的所有文档")
        
        try:
            result = self.es.delete_by_query(
                index=self.index_name,
                body={"query": {"match_all": {}}}
            )
            
            deleted = result.get('deleted', 0)
            logger.info(f"删除了 {deleted} 个文档")
            return True
                
        except Exception as e:
            logger.error(f"删除所有文档时出错: {str(e)}")
            return False


class DummySearchClient(SearchClient):
    """虚拟搜索客户端，用于在Elasticsearch不可用时提供基本功能"""
    
    def __init__(self):
        """初始化虚拟搜索客户端"""
        self.index_name = "dummy_index"
        self.documents = []
        logger.info("虚拟搜索客户端初始化完成")
    
    def search(self, query_vector: List[float], top_k: int = 5) -> Dict[str, Any]:
        """虚拟搜索，返回空结果
        
        Args:
            query_vector: 查询向量
            top_k: 返回的最相似文档数量
            
        Returns:
            空搜索结果
        """
        logger.info(f"执行虚拟搜索，查询向量长度: {len(query_vector)}, top_k: {top_k}")
        return {
            "total": 0,
            "results": []
        }
    
    def index_document(self, document: Dict[str, Any]) -> bool:
        """虚拟索引单个文档
        
        Args:
            document: 文档数据
            
        Returns:
            总是返回成功
        """
        logger.info(f"虚拟索引文档: {document.get('title', '未知标题')}")
        self.documents.append(document)
        return True
    
    def bulk_index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """虚拟批量索引文档
        
        Args:
            documents: 文档数据列表
            
        Returns:
            虚拟索引结果
        """
        logger.info(f"虚拟批量索引文档，数量: {len(documents)}")
        self.documents.extend(documents)
        return {"success": len(documents), "errors": 0}
    
    def delete_document(self, doc_id: str) -> bool:
        """虚拟删除文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            总是返回成功
        """
        logger.info(f"虚拟删除文档，ID: {doc_id}")
        return True
    
    def delete_all_documents(self) -> bool:
        """虚拟删除所有文档
        
        Returns:
            总是返回成功
        """
        logger.info("虚拟删除所有文档")
        self.documents = []
        return True


# 创建默认搜索客户端实例，方便直接导入使用
try:
    elasticsearch_client = ElasticsearchClient()
except Exception as e:
    logger.warning(f"Elasticsearch客户端初始化失败: {str(e)}，使用虚拟搜索客户端")
    elasticsearch_client = DummySearchClient() 