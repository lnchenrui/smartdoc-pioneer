"""RAG 服务模块

这个模块提供了检索增强生成 (RAG) 相关的服务功能，使用 Langchain 的 LCEL 和 RAG 功能。
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Generator
import time

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.utils.logging.logger import get_logger
from app.utils.error_handler import ServiceError, ValidationError

logger = get_logger("services.rag_service")

class RAGService:
    """RAG 服务类，提供检索增强生成功能"""
    
    def __init__(
        self,
        llm_service,
        vector_store_service,
        prompt_service,
        system_message: Optional[str] = None,
        top_k: int = 5,
        search_type: str = "similarity"
    ):
        """初始化 RAG 服务
        
        Args:
            llm_service: LLM 服务实例
            vector_store_service: 向量存储服务实例
            prompt_service: 提示模板服务实例
            system_message: 系统消息
            top_k: 检索的文档数量
            search_type: 搜索类型，支持 "similarity", "mmr"
        """
        self.llm_service = llm_service
        self.vector_store_service = vector_store_service
        self.prompt_service = prompt_service
        self.system_message = system_message
        self.top_k = top_k
        self.search_type = search_type
        
        # 获取 LLM 模型
        self.llm = llm_service.get_llm()
        
        # 创建检索器
        self.retriever = vector_store_service.as_retriever(
            search_type=search_type,
            search_kwargs={"k": top_k}
        )
        
        # 创建 RAG 提示模板
        self.rag_prompt = prompt_service.create_rag_prompt()
        
        # 创建 RAG 链
        self._create_rag_chain()
        
        logger.info("RAG 服务初始化完成")
    
    def _create_rag_chain(self):
        """创建 RAG 链"""
        # 创建文档格式化函数
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        # 创建 RAG 链
        self.rag_chain = (
            {
                "context": self.retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )
        
        logger.debug("RAG 链创建完成")
    
    def retrieve(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回的最大结果数，如果为 None 则使用默认值
            filter_dict: 过滤条件字典
            
        Returns:
            检索到的文档列表
        """
        try:
            if not query:
                raise ValidationError("查询不能为空")
            
            # 使用配置中的 top_k 如果未指定
            if top_k is None:
                top_k = self.top_k
            
            logger.info(f"RAG 检索: query='{query}', top_k={top_k}, search_type={self.search_type}")
            
            # 创建检索参数
            search_kwargs = {"k": top_k}
            if filter_dict:
                search_kwargs["filter"] = filter_dict
            
            # 创建检索器
            retriever = self.vector_store_service.as_retriever(
                search_type=self.search_type,
                search_kwargs=search_kwargs
            )
            
            # 执行检索
            start_time = time.time()
            results = retriever.invoke(query)
            elapsed = time.time() - start_time
            
            # 记录检索结果
            logger.info(f"RAG 检索完成，找到 {len(results)} 个结果，耗时: {elapsed:.2f}秒")
            
            return results
        except Exception as e:
            error_msg = f"RAG 检索失败: {str(e)}"
            logger.error(error_msg)
            raise ServiceError(error_msg)
    
    def generate(self, query: str, context: Union[str, List[Document]]) -> str:
        """生成回答
        
        Args:
            query: 查询文本
            context: 上下文文本或文档列表
            
        Returns:
            生成的回答
        """
        try:
            if not query:
                raise ValidationError("查询不能为空")
            
            logger.info(f"RAG 生成: query='{query}'")
            
            # 格式化上下文
            if isinstance(context, list):
                context_text = "\n\n".join([doc.page_content for doc in context])
            else:
                context_text = context
            
            # 创建输入
            inputs = {
                "context": context_text,
                "question": query
            }
            
            # 执行生成
            start_time = time.time()
            result = self.rag_prompt.invoke(inputs) | self.llm | StrOutputParser()
            elapsed = time.time() - start_time
            
            logger.info(f"RAG 生成完成，耗时: {elapsed:.2f}秒")
            
            return result
        except Exception as e:
            error_msg = f"RAG 生成失败: {str(e)}"
            logger.error(error_msg)
            raise ServiceError(error_msg)
    
    def process(self, query: str, filter_dict: Optional[Dict[str, Any]] = None) -> str:
        """处理 RAG 查询
        
        Args:
            query: 查询文本
            filter_dict: 过滤条件字典
            
        Returns:
            生成的回答
        """
        try:
            if not query:
                raise ValidationError("查询不能为空")
            
            logger.info(f"处理 RAG 查询: query='{query}'")
            
            # 创建检索参数
            search_kwargs = {"k": self.top_k}
            if filter_dict:
                search_kwargs["filter"] = filter_dict
                
                # 更新检索器
                self.retriever = self.vector_store_service.as_retriever(
                    search_type=self.search_type,
                    search_kwargs=search_kwargs
                )
                
                # 重新创建 RAG 链
                self._create_rag_chain()
            
            # 执行 RAG 链
            start_time = time.time()
            result = self.rag_chain.invoke(query)
            elapsed = time.time() - start_time
            
            logger.info(f"RAG 查询处理完成，耗时: {elapsed:.2f}秒")
            
            return result
        except Exception as e:
            error_msg = f"处理 RAG 查询失败: {str(e)}"
            logger.error(error_msg)
            raise ServiceError(error_msg)
    
    def process_with_sources(
        self, 
        query: str, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """处理 RAG 查询并返回来源
        
        Args:
            query: 查询文本
            filter_dict: 过滤条件字典
            
        Returns:
            (生成的回答, 来源列表)
        """
        try:
            if not query:
                raise ValidationError("查询不能为空")
            
            logger.info(f"处理 RAG 查询并返回来源: query='{query}'")
            
            # 检索文档
            docs = self.retrieve(query, filter_dict=filter_dict)
            
            if not docs:
                logger.warning("未检索到相关文档")
                return "我无法找到与您问题相关的信息。", []
            
            # 生成回答
            answer = self.generate(query, docs)
            
            # 提取来源信息
            sources = []
            for doc in docs:
                source = {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source)
            
            return answer, sources
        except Exception as e:
            error_msg = f"处理 RAG 查询并返回来源失败: {str(e)}"
            logger.error(error_msg)
            raise ServiceError(error_msg)
    
    def process_stream(
        self, 
        query: str, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Generator[str, None, None]:
        """流式处理 RAG 查询
        
        Args:
            query: 查询文本
            filter_dict: 过滤条件字典
            
        Returns:
            生成的回答的生成器
        """
        try:
            if not query:
                raise ValidationError("查询不能为空")
            
            logger.info(f"流式处理 RAG 查询: query='{query}'")
            
            # 检索文档
            docs = self.retrieve(query, filter_dict=filter_dict)
            
            if not docs:
                logger.warning("未检索到相关文档")
                yield "我无法找到与您问题相关的信息。"
                return
            
            # 格式化上下文
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # 创建输入
            inputs = {
                "context": context,
                "question": query
            }
            
            # 创建消息
            messages = [
                SystemMessage(content=f"你是一个智能助手，可以回答用户关于文档内容的问题。请基于提供的文档内容回答问题。\n\n文档内容:\n{context}"),
                HumanMessage(content=query)
            ]
            
            # 流式生成
            for chunk in self.llm_service.stream_chat_response([
                {"role": "system", "content": f"你是一个智能助手，可以回答用户关于文档内容的问题。请基于提供的文档内容回答问题。\n\n文档内容:\n{context}"},
                {"role": "user", "content": query}
            ]):
                yield chunk
                
        except Exception as e:
            error_msg = f"流式处理 RAG 查询失败: {str(e)}"
            logger.error(error_msg)
            raise ServiceError(error_msg) 