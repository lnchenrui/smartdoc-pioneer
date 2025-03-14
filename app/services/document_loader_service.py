"""文档加载服务模块

这个模块提供了文档加载相关的服务功能，使用 Langchain 的最新文档加载功能。
"""

from typing import List, Dict, Any, Optional, Union
import os
import time
import mimetypes

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredFileLoader,
    DirectoryLoader
)

from app.utils.logging.logger import get_logger
from app.utils.error_handler import ServiceError, ValidationError

logger = get_logger("services.document_loader")

class DocumentLoaderService:
    """文档加载服务类，提供文档加载功能"""
    
    def __init__(
        self,
        config: Dict[str, Any]
    ):
        """初始化文档加载服务
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 获取文档加载配置
        self.supported_extensions = config.get('document_loader', {}).get('supported_extensions', [
            '.txt', '.pdf', '.csv', '.md', '.html', '.htm', '.doc', '.docx', 
            '.ppt', '.pptx', '.xls', '.xlsx', '.json', '.xml'
        ])
        
        # 初始化MIME类型
        mimetypes.init()
        
        logger.debug("文档加载服务初始化完成")
    
    def is_supported_file(self, file_path: str) -> bool:
        """检查文件是否支持
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否支持
        """
        _, ext = os.path.splitext(file_path.lower())
        return ext in self.supported_extensions
    
    def get_loader_for_file(self, file_path: str):
        """获取文件的加载器
        
        Args:
            file_path: 文件路径
            
        Returns:
            加载器实例
            
        Raises:
            ValidationError: 文件不支持时抛出
        """
        if not os.path.exists(file_path):
            raise ValidationError(f"文件不存在: {file_path}")
        
        _, ext = os.path.splitext(file_path.lower())
        
        # 根据文件扩展名选择加载器
        if ext == '.txt':
            return TextLoader(file_path, encoding='utf-8')
        elif ext == '.pdf':
            return PyPDFLoader(file_path)
        elif ext == '.csv':
            return CSVLoader(file_path)
        elif ext == '.md':
            return UnstructuredMarkdownLoader(file_path)
        elif ext in ['.html', '.htm']:
            return UnstructuredHTMLLoader(file_path)
        elif ext in ['.doc', '.docx']:
            return UnstructuredWordDocumentLoader(file_path)
        elif ext in ['.ppt', '.pptx']:
            return UnstructuredPowerPointLoader(file_path)
        elif ext in ['.xls', '.xlsx']:
            return UnstructuredExcelLoader(file_path)
        elif ext in self.supported_extensions:
            # 尝试使用通用加载器
            return UnstructuredFileLoader(file_path)
        else:
            raise ValidationError(f"不支持的文件类型: {ext}")
    
    def load_document(self, file_path: str) -> List[Document]:
        """加载文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档列表
            
        Raises:
            ServiceError: 加载失败时抛出
        """
        try:
            if not os.path.exists(file_path):
                raise ValidationError(f"文件不存在: {file_path}")
            
            if os.path.isdir(file_path):
                # 加载目录
                return self.load_directory(file_path)
            
            if not self.is_supported_file(file_path):
                raise ValidationError(f"不支持的文件类型: {file_path}")
            
            logger.info(f"加载文档: {file_path}")
            
            # 获取加载器
            loader = self.get_loader_for_file(file_path)
            
            # 加载文档
            start_time = time.time()
            documents = loader.load()
            elapsed = time.time() - start_time
            
            # 添加文件路径元数据
            for doc in documents:
                if "source" not in doc.metadata:
                    doc.metadata["source"] = file_path
            
            logger.info(f"文档加载完成，加载了 {len(documents)} 个文档，耗时: {elapsed:.2f}秒")
            
            return documents
            
        except Exception as e:
            error_msg = f"加载文档失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """加载目录
        
        Args:
            directory_path: 目录路径
            
        Returns:
            文档列表
            
        Raises:
            ServiceError: 加载失败时抛出
        """
        try:
            if not os.path.exists(directory_path):
                raise ValidationError(f"目录不存在: {directory_path}")
            
            if not os.path.isdir(directory_path):
                raise ValidationError(f"不是目录: {directory_path}")
            
            logger.info(f"加载目录: {directory_path}")
            
            # 创建目录加载器
            glob_pattern = "**/*"  # 递归加载所有文件
            
            # 创建加载器映射
            loader_mapping = {
                ".txt": TextLoader,
                ".pdf": PyPDFLoader,
                ".csv": CSVLoader,
                ".md": UnstructuredMarkdownLoader,
                ".html": UnstructuredHTMLLoader,
                ".htm": UnstructuredHTMLLoader,
                ".doc": UnstructuredWordDocumentLoader,
                ".docx": UnstructuredWordDocumentLoader,
                ".ppt": UnstructuredPowerPointLoader,
                ".pptx": UnstructuredPowerPointLoader,
                ".xls": UnstructuredExcelLoader,
                ".xlsx": UnstructuredExcelLoader
            }
            
            # 创建目录加载器
            loader = DirectoryLoader(
                directory_path,
                glob=glob_pattern,
                loader_cls=UnstructuredFileLoader,  # 默认加载器
                loader_kwargs={"mode": "elements"},
                use_multithreading=True,
                show_progress=True,
                load_hidden=False,
                loader_mapping=loader_mapping
            )
            
            # 加载文档
            start_time = time.time()
            documents = loader.load()
            elapsed = time.time() - start_time
            
            logger.info(f"目录加载完成，加载了 {len(documents)} 个文档，耗时: {elapsed:.2f}秒")
            
            return documents
            
        except Exception as e:
            error_msg = f"加载目录失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)
    
    def load_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """加载文本
        
        Args:
            text: 文本内容
            metadata: 元数据
            
        Returns:
            文档列表
        """
        if not text:
            logger.warning("文本内容为空")
            return []
        
        logger.debug(f"加载文本: 长度={len(text)}")
        
        # 创建文档
        document = Document(
            page_content=text,
            metadata=metadata or {}
        )
        
        return [document]
    
    def load_texts(
        self, 
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[Document]:
        """加载多个文本
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
            
        Returns:
            文档列表
        """
        if not texts:
            logger.warning("文本列表为空")
            return []
        
        logger.info(f"加载 {len(texts)} 个文本")
        
        # 创建文档列表
        documents = []
        
        for i, text in enumerate(texts):
            # 获取元数据
            metadata = {}
            if metadatas and i < len(metadatas):
                metadata = metadatas[i]
            
            # 创建文档
            document = Document(
                page_content=text,
                metadata=metadata
            )
            
            documents.append(document)
        
        return documents 