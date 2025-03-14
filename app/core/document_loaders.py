"""文档加载器模块

这个模块提供了文档加载和处理功能，使用 Langchain 的文档加载器和文本分割器。
"""

import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader, 
    TextLoader, 
    PyPDFLoader,
    CSVLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
from langchain_core.documents import Document

from app.utils.logging.logger import get_logger
from app.utils.error_handler import DocumentError

logger = get_logger("core.document_loaders")

class DocumentLoader:
    """文档加载器，用于加载和处理文档"""
    
    def __init__(self, datasets_dir: str = 'datasets'):
        """初始化文档加载器
        
        Args:
            datasets_dir: 数据集目录路径
        """
        self.datasets_dir = datasets_dir
        logger.info(f"文档加载器初始化完成，数据集目录: {datasets_dir}")
    
    def load_documents(
        self, 
        path: Optional[str] = None, 
        recursive: bool = True,
        file_types: Optional[List[str]] = None
    ) -> List[Document]:
        """加载文档
        
        Args:
            path: 文档路径，如果为None则使用默认路径
            recursive: 是否递归加载子目录
            file_types: 要加载的文件类型列表，如[".txt", ".pdf", ".md"]
            
        Returns:
            文档列表
            
        Raises:
            DocumentError: 加载文档失败时抛出
        """
        try:
            # 如果未指定路径，使用默认路径
            if path is None:
                path = self.datasets_dir
            
            # 确保路径存在
            if not os.path.exists(path):
                error_msg = f"文档路径不存在: {path}"
                logger.error(error_msg)
                raise DocumentError(error_msg)
            
            # 如果未指定文件类型，使用默认文件类型
            if file_types is None:
                file_types = [".txt", ".pdf", ".md", ".csv"]
            
            # 创建加载器映射
            loader_mapping = {
                ".txt": TextLoader,
                ".pdf": PyPDFLoader,
                ".md": UnstructuredMarkdownLoader,
                ".csv": CSVLoader
            }
            
            # 过滤支持的文件类型
            supported_types = [ext for ext in file_types if ext in loader_mapping]
            if not supported_types:
                error_msg = f"未指定支持的文件类型，支持的类型: {list(loader_mapping.keys())}"
                logger.error(error_msg)
                raise DocumentError(error_msg)
            
            # 创建加载器
            loaders = []
            for file_type in supported_types:
                glob_pattern = f"**/*{file_type}" if recursive else f"*{file_type}"
                loader_class = loader_mapping[file_type]
                loader = DirectoryLoader(
                    path, 
                    glob=glob_pattern, 
                    loader_cls=loader_class,
                    show_progress=True
                )
                loaders.append(loader)
            
            # 加载文档
            documents = []
            for loader in loaders:
                try:
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info(f"成功加载 {len(docs)} 个文档")
                except Exception as e:
                    logger.warning(f"加载器 {loader.__class__.__name__} 加载失败: {str(e)}")
            
            if not documents:
                logger.warning(f"未找到任何文档，路径: {path}, 文件类型: {supported_types}")
                return []
            
            logger.info(f"成功加载所有文档，总数: {len(documents)}")
            return documents
            
        except DocumentError:
            # 重新抛出DocumentError
            raise
        except Exception as e:
            error_msg = f"加载文档时发生错误: {str(e)}"
            logger.error(error_msg)
            raise DocumentError(error_msg)
    
    def split_documents(
        self, 
        documents: List[Document], 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> List[Document]:
        """将文档分割成小块
        
        Args:
            documents: 文档列表
            chunk_size: 每块的最大字符数
            chunk_overlap: 相邻块之间的重叠字符数
            
        Returns:
            分割后的文档块列表
        """
        try:
            if not documents:
                logger.warning("没有文档需要分割")
                return []
            
            # 创建文本分割器
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False
            )
            
            # 分割文档
            chunks = text_splitter.split_documents(documents)
            
            logger.info(f"文档分割完成，原始文档数: {len(documents)}, 分割后块数: {len(chunks)}")
            return chunks
            
        except Exception as e:
            error_msg = f"分割文档时发生错误: {str(e)}"
            logger.error(error_msg)
            raise DocumentError(error_msg)
    
    def load_and_split(
        self, 
        path: Optional[str] = None, 
        recursive: bool = True,
        file_types: Optional[List[str]] = None,
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> List[Document]:
        """加载并分割文档
        
        Args:
            path: 文档路径，如果为None则使用默认路径
            recursive: 是否递归加载子目录
            file_types: 要加载的文件类型列表
            chunk_size: 每块的最大字符数
            chunk_overlap: 相邻块之间的重叠字符数
            
        Returns:
            分割后的文档块列表
        """
        # 加载文档
        documents = self.load_documents(path, recursive, file_types)
        
        # 分割文档
        if documents:
            return self.split_documents(documents, chunk_size, chunk_overlap)
        
        return []
    
    def read_all_documents(self) -> str:
        """读取所有文档内容并合并
        
        Returns:
            所有文档内容的组合
        """
        try:
            # 加载文档
            documents = self.load_documents(self.datasets_dir, recursive=True)
            
            if not documents:
                logger.warning("未找到任何文档")
                return "未找到任何文档内容"
            
            # 合并所有文档内容
            combined_content = "\n\n".join([doc.page_content for doc in documents])
            
            logger.info(f"成功读取所有文档，总字符数: {len(combined_content)}")
            return combined_content
            
        except Exception as e:
            error_msg = f"读取文档时发生错误: {str(e)}"
            logger.error(error_msg)
            raise DocumentError(error_msg) 