"""RAG服务模块

这个模块提供了RAG（检索增强生成）服务的主要功能。
"""

import time
from typing import List, Dict, Any, Optional, Generator, Union

from app.utils.logging.logger import get_logger
from app.utils.exceptions import ServiceError
from app.core.document.processor import DocumentProcessor
from app.core.llm.client import LLMClient
from app.core.llm.response import LLMResponseHandler
from app.core.search import ElasticsearchClient
from app.core.embedding import OpenAIEmbeddingClient, batch_embedding

logger = get_logger("services.rag.service")

class RAGService:
    """RAG服务类，整合检索和生成功能"""
    
    def __init__(
        self,
        document_processor: DocumentProcessor,
        llm_client: LLMClient,
        llm_response_handler: Optional[LLMResponseHandler] = None,
        system_message: str = "",
        search_client = None,
        embedding_client = None
    ):
        """初始化RAG服务
        
        Args:
            document_processor: 文档处理器
            llm_client: LLM客户端
            llm_response_handler: LLM响应处理器，如果为None则创建新实例
            system_message: 系统消息
            search_client: 搜索客户端，如果为None则创建新实例
            embedding_client: 嵌入客户端，如果为None则创建新实例
        """
        self.document_processor = document_processor
        self.llm_client = llm_client
        self.llm_response_handler = llm_response_handler or LLMResponseHandler()
        self.system_message = system_message or "你是一个智能助手，可以回答用户关于文档内容的问题。请基于提供的文档内容回答问题，如果无法从文档中找到答案，请明确告知用户。"
        
        # 初始化搜索和嵌入客户端
        try:
            self.search_client = search_client or ElasticsearchClient()
            self.embedding_client = embedding_client or OpenAIEmbeddingClient()
            self.use_vector_search = True
            logger.info("向量搜索功能已启用")
        except Exception as e:
            logger.warning(f"初始化向量搜索功能失败: {str(e)}，将使用简单搜索")
            self.use_vector_search = False
        
        logger.info("RAG服务初始化完成")
    
    def create_prompt(self, query: str, search_results: str) -> List[Dict[str, str]]:
        """创建提示
        
        Args:
            query: 用户查询
            search_results: 搜索结果
            
        Returns:
            消息列表
        """
        system_message = {
            "role": "system",
            "content": self.system_message
        }
        
        # 将检索结果格式化为上下文
        context = search_results if search_results != "未找到相关内容" else "没有找到与问题相关的文档内容。"
        
        user_message = {
            "role": "user",
            "content": f"基于以下文档内容，回答我的问题。\n\n文档内容：\n{context}\n\n问题：{query}"
        }
        
        logger.debug(f"创建提示模板，用户查询: {query}, 检索结果长度: {len(search_results)}字符")
        return [system_message, user_message]
    
    def search(self, query: str) -> str:
        """搜索相关文档
        
        Args:
            query: 用户查询
            
        Returns:
            格式化的搜索结果
            
        Raises:
            ServiceError: 搜索失败时抛出
        """
        logger.info(f"开始处理搜索请求: {query}")
        start_time = time.time()
        
        try:
            # 如果启用了向量搜索
            if self.use_vector_search:
                try:
                    # 获取查询的向量嵌入
                    query_embedding = self.embedding_client.get_embedding([query])[0]
                    
                    # 执行向量搜索
                    search_results = self.search_client.search(query_vector=query_embedding, top_k=5)
                    
                    # 提取搜索结果中的文档内容
                    hits = search_results.get('hits', {}).get('hits', [])
                    if hits:
                        # 格式化搜索结果
                        formatted_results = []
                        for hit in hits:
                            source = hit.get('_source', {})
                            content = source.get('content', '')
                            metadata = source.get('metadata', {})
                            file_name = metadata.get('file_name', '未知文件')
                            score = hit.get('_score', 0)
                            
                            formatted_results.append(f"文件: {file_name} (相关度: {score:.2f})\n{content}")
                        
                        result = "\n\n".join(formatted_results)
                        logger.info(f"向量搜索完成，找到 {len(hits)} 个相关文档")
                        
                        elapsed = time.time() - start_time
                        logger.info(f"搜索处理完成，耗时: {elapsed:.2f}秒")
                        return result
                    else:
                        logger.warning("向量搜索未找到相关文档，将使用简单搜索")
                except Exception as e:
                    logger.warning(f"向量搜索失败: {str(e)}，将使用简单搜索")
            
            # 如果向量搜索失败或未启用，使用简单搜索
            logger.info("使用简单搜索")
            content = self.document_processor.read_all_documents()
            
            elapsed = time.time() - start_time
            logger.info(f"简单搜索处理完成，耗时: {elapsed:.2f}秒")
            return content
            
        except Exception as e:
            error_msg = f"搜索处理异常: {str(e)}"
            logger.error(error_msg)
            raise ServiceError(error_msg)
    
    def process(self, query: str) -> str:
        """处理用户查询，返回回复
        
        Args:
            query: 用户查询
            
        Returns:
            生成的回复
            
        Raises:
            ServiceError: 处理失败时抛出
        """
        logger.info(f"开始处理用户查询: {query}")
        start_time = time.time()
        
        try:
            # 搜索相关文档
            search_results = self.search(query)
            
            # 如果没有找到相关内容，直接返回提示信息
            if search_results == "未找到相关内容":
                logger.info("未找到相关内容，返回默认提示信息")
                return "抱歉，我没有找到与您问题相关的信息。请尝试使用不同的关键词或更具体的问题。"
            
            # 创建提示
            messages = self.create_prompt(query, search_results)
            
            # 调用LLM生成回复
            logger.info("调用LLM生成非流式回复")
            response = self.llm_client.generate_response(messages, stream=False)
            response_json = self.llm_response_handler.handle_normal_response(response)
            content = self.llm_response_handler.extract_content_from_response(response_json)
            
            elapsed = time.time() - start_time
            logger.info(f"查询处理完成，耗时: {elapsed:.2f}秒")
            return content
            
        except Exception as e:
            error_msg = f"处理用户查询异常: {str(e)}"
            logger.error(error_msg)
            raise ServiceError(error_msg)
    
    def process_stream(self, query: str) -> Generator[Dict[str, Any], None, None]:
        """流式处理用户查询，返回流式回复
        
        Args:
            query: 用户查询
            
        Returns:
            流式回复生成器
            
        Raises:
            ServiceError: 处理失败时抛出
        """
        logger.info(f"开始流式处理用户查询: {query}")
        start_time = time.time()
        
        try:
            # 搜索相关文档
            search_results = self.search(query)
            
            # 如果没有找到相关内容，直接返回提示信息
            if search_results == "未找到相关内容":
                logger.info("未找到相关内容，返回默认提示信息")
                yield {'content': "抱歉，我没有找到与您问题相关的信息。请尝试使用不同的关键词或更具体的问题。"}
                return
            
            # 创建提示
            messages = self.create_prompt(query, search_results)
            
            # 调用LLM生成流式回复
            logger.info("调用LLM生成流式回复")
            response = self.llm_client.generate_response(messages, stream=True)
            for chunk in self.llm_response_handler.handle_stream_response(response):
                yield chunk
            
            elapsed = time.time() - start_time
            logger.info(f"流式处理完成，耗时: {elapsed:.2f}秒")
            
        except Exception as e:
            error_msg = f"流式处理用户查询异常: {str(e)}"
            logger.error(error_msg)
            yield {'error': str(e)}
            raise ServiceError(error_msg)
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """索引文档
        
        Args:
            documents: 文档数据列表
            
        Returns:
            索引结果
            
        Raises:
            ServiceError: 索引失败时抛出
        """
        logger.info(f"开始索引文档，数量: {len(documents)}")
        start_time = time.time()
        
        try:
            success_count = 0
            error_count = 0
            
            # 如果启用了向量搜索
            if self.use_vector_search:
                try:
                    # 处理每个文档
                    processed_documents = []
                    for doc in documents:
                        try:
                            # 获取文档内容和元数据
                            content = doc.get('content', '')
                            metadata = doc.get('metadata', {})
                            
                            # 获取文档的向量嵌入
                            embedding = self.embedding_client.get_embedding([content])[0]
                            
                            # 准备索引文档
                            processed_doc = {
                                'content': content,
                                'metadata': metadata,
                                'embedding': embedding
                            }
                            
                            processed_documents.append(processed_doc)
                            success_count += 1
                        except Exception as e:
                            logger.error(f"处理文档失败: {str(e)}")
                            error_count += 1
                    
                    # 批量索引文档
                    if processed_documents:
                        result = self.search_client.bulk_index_documents(processed_documents)
                        logger.info(f"批量索引完成，结果: {result}")
                except Exception as e:
                    logger.error(f"向量索引失败: {str(e)}")
                    error_count += len(documents)
            else:
                # 简单实现，仅返回成功信息
                logger.info("向量搜索未启用，跳过索引")
                success_count = len(documents)
            
            elapsed = time.time() - start_time
            logger.info(f"文档索引完成，成功: {success_count}，失败: {error_count}，耗时: {elapsed:.2f}秒")
            return {"success": success_count, "errors": error_count}
            
        except Exception as e:
            error_msg = f"索引文档异常: {str(e)}"
            logger.error(error_msg)
            raise ServiceError(error_msg) 