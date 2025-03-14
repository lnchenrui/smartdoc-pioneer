"""索引服务模块

这个模块提供了文档索引相关的服务功能，使用 Langchain 的最新索引功能。
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import os
import time
import uuid
from datetime import datetime

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.utils.logging.logger import get_logger
from app.utils.error_handler import ServiceError, ValidationError

logger = get_logger("services.index")

class IndexService:
    """索引服务类，提供文档索引功能"""
    
    def __init__(
        self,
        vector_store_service,
        embedding_service,
        document_loader_service,
        config: Dict[str, Any]
    ):
        """初始化索引服务
        
        Args:
            vector_store_service: 向量存储服务实例
            embedding_service: 嵌入服务实例
            document_loader_service: 文档加载服务实例
            config: 配置字典
        """
        self.vector_store_service = vector_store_service
        self.embedding_service = embedding_service
        self.document_loader_service = document_loader_service
        self.config = config
        
        # 获取索引配置
        self.chunk_size = config.get('index', {}).get('chunk_size', 1000)
        self.chunk_overlap = config.get('index', {}).get('chunk_overlap', 200)
        self.upload_dir = config.get('index', {}).get('upload_dir', 'uploads')
        
        # 确保上传目录存在
        os.makedirs(self.upload_dir, exist_ok=True)
        
        logger.debug("索引服务初始化完成")
    
    def create_text_splitter(
        self, 
        chunk_size: Optional[int] = None, 
        chunk_overlap: Optional[int] = None
    ) -> RecursiveCharacterTextSplitter:
        """创建文本分割器
        
        Args:
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            
        Returns:
            文本分割器实例
        """
        # 使用默认值如果未指定
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        if chunk_overlap is None:
            chunk_overlap = self.chunk_overlap
        
        logger.debug(f"创建文本分割器: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
    
    def process_document(
        self, 
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Document]:
        """处理文档
        
        加载文档并分割成块
        
        Args:
            file_path: 文档路径
            metadata: 额外的元数据
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            
        Returns:
            处理后的文档块列表
            
        Raises:
            ServiceError: 处理失败时抛出
        """
        try:
            if not os.path.exists(file_path):
                raise ValidationError(f"文件不存在: {file_path}")
            
            logger.info(f"处理文档: {file_path}")
            
            # 加载文档
            start_time = time.time()
            documents = self.document_loader_service.load_document(file_path)
            load_time = time.time() - start_time
            
            logger.info(f"文档加载完成，加载了 {len(documents)} 个文档，耗时: {load_time:.2f}秒")
            
            # 创建文本分割器
            text_splitter = self.create_text_splitter(chunk_size, chunk_overlap)
            
            # 分割文档
            start_time = time.time()
            chunks = text_splitter.split_documents(documents)
            split_time = time.time() - start_time
            
            logger.info(f"文档分割完成，生成了 {len(chunks)} 个文本块，耗时: {split_time:.2f}秒")
            
            # 添加额外元数据
            if metadata:
                for chunk in chunks:
                    chunk.metadata.update(metadata)
            
            # 添加处理时间元数据
            for chunk in chunks:
                chunk.metadata.update({
                    "processed_at": datetime.now().isoformat(),
                    "chunk_id": str(uuid.uuid4())
                })
            
            return chunks
            
        except Exception as e:
            error_msg = f"处理文档失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def index_document(
        self, 
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> Tuple[int, float]:
        """索引文档
        
        处理文档并添加到向量存储
        
        Args:
            file_path: 文档路径
            metadata: 额外的元数据
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            
        Returns:
            元组 (索引的文档块数量, 总耗时)
            
        Raises:
            ServiceError: 索引失败时抛出
        """
        try:
            start_time = time.time()
            
            # 处理文档
            chunks = self.process_document(
                file_path=file_path,
                metadata=metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # 添加到向量存储
            self.vector_store_service.add_documents(chunks)
            
            total_time = time.time() - start_time
            logger.info(f"文档索引完成，索引了 {len(chunks)} 个文本块，总耗时: {total_time:.2f}秒")
            
            return len(chunks), total_time
            
        except Exception as e:
            error_msg = f"索引文档失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def index_text(
        self, 
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> Tuple[int, float]:
        """索引文本
        
        处理文本并添加到向量存储
        
        Args:
            text: 文本内容
            metadata: 元数据
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            
        Returns:
            元组 (索引的文档块数量, 总耗时)
            
        Raises:
            ServiceError: 索引失败时抛出
        """
        try:
            if not text:
                raise ValidationError("文本内容不能为空")
            
            start_time = time.time()
            
            # 创建文本分割器
            text_splitter = self.create_text_splitter(chunk_size, chunk_overlap)
            
            # 分割文本
            chunks = text_splitter.create_documents([text], [metadata or {}])
            
            # 添加处理时间元数据
            for chunk in chunks:
                chunk.metadata.update({
                    "processed_at": datetime.now().isoformat(),
                    "chunk_id": str(uuid.uuid4())
                })
            
            # 添加到向量存储
            self.vector_store_service.add_documents(chunks)
            
            total_time = time.time() - start_time
            logger.info(f"文本索引完成，索引了 {len(chunks)} 个文本块，总耗时: {total_time:.2f}秒")
            
            return len(chunks), total_time
            
        except Exception as e:
            error_msg = f"索引文本失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def delete_documents(
        self, 
        filter_dict: Dict[str, Any]
    ) -> int:
        """删除文档
        
        根据过滤条件删除向量存储中的文档
        
        Args:
            filter_dict: 过滤条件字典
            
        Returns:
            删除的文档数量
            
        Raises:
            ServiceError: 删除失败时抛出
        """
        try:
            if not filter_dict:
                raise ValidationError("过滤条件不能为空")
            
            logger.info(f"删除文档: filter={filter_dict}")
            
            # 删除文档
            start_time = time.time()
            deleted_count = self.vector_store_service.delete(filter_dict)
            elapsed = time.time() - start_time
            
            logger.info(f"文档删除完成，删除了 {deleted_count} 个文档，耗时: {elapsed:.2f}秒")
            
            return deleted_count
            
        except Exception as e:
            error_msg = f"删除文档失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def get_document_count(
        self, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> int:
        """获取文档数量
        
        Args:
            filter_dict: 过滤条件字典
            
        Returns:
            文档数量
            
        Raises:
            ServiceError: 获取失败时抛出
        """
        try:
            logger.info(f"获取文档数量: filter={filter_dict}")
            
            # 获取文档数量
            count = self.vector_store_service.get_document_count(filter_dict)
            
            logger.info(f"获取文档数量完成: {count}")
            
            return count
            
        except Exception as e:
            error_msg = f"获取文档数量失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg) 