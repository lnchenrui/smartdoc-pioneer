"""文档处理模块

这个模块提供了文档处理功能，包括读取和处理文档内容。
"""

import os
from typing import List, Dict, Any, Optional

from app.utils.logging.logger import get_logger
from app.utils.exceptions import DocumentError

logger = get_logger("document.processor")


class DocumentProcessor:
    """文档处理器，用于读取和处理文档内容"""
    
    def __init__(self, datasets_dir: str = 'datasets'):
        """初始化文档处理器
        
        Args:
            datasets_dir: 数据集目录路径
        """
        self.datasets_dir = datasets_dir
        logger.info(f"文档处理器初始化完成，数据集目录: {datasets_dir}")
    
    def read_document(self, file_path: str) -> str:
        """读取单个文档内容
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            文档内容
            
        Raises:
            DocumentError: 读取文档失败时抛出
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"成功读取文件: {file_path}")
            return content
        except Exception as e:
            error_msg = f"读取文件 {file_path} 时出错: {str(e)}"
            logger.error(error_msg)
            raise DocumentError(error_msg)
    
    def read_all_documents(self) -> str:
        """读取datasets目录下所有文档的内容
        
        Returns:
            所有文档内容的组合
            
        Raises:
            DocumentError: 读取文档失败时抛出
        """
        all_content = []
        try:
            # 确保目录存在
            if not os.path.exists(self.datasets_dir):
                error_msg = f"数据集目录不存在: {self.datasets_dir}"
                logger.error(error_msg)
                raise DocumentError(error_msg)
            
            # 读取所有.txt文件
            for filename in os.listdir(self.datasets_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(self.datasets_dir, filename)
                    content = self.read_document(file_path)
                    if content:
                        all_content.append(f"文件 {filename} 的内容:\n{content}")
            
            if not all_content:
                logger.warning("未找到任何.txt文件")
                return "未找到任何文档内容"
            
            # 合并所有文档内容
            combined_content = "\n\n".join(all_content)
            logger.info(f"成功读取所有文档，总字符数: {len(combined_content)}")
            return combined_content
            
        except DocumentError:
            # 重新抛出DocumentError
            raise
        except Exception as e:
            error_msg = f"读取文档时发生错误: {str(e)}"
            logger.error(error_msg)
            raise DocumentError(error_msg)
    
    def split_document(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """将文档内容分割成小块
        
        Args:
            content: 文档内容
            chunk_size: 每块的最大字符数
            overlap: 相邻块之间的重叠字符数
            
        Returns:
            分割后的文档块列表
        """
        if not content:
            return []
        
        chunks = []
        start = 0
        content_len = len(content)
        
        while start < content_len:
            # 计算当前块的结束位置
            end = min(start + chunk_size, content_len)
            
            # 如果不是最后一块，尝试在句子边界分割
            if end < content_len:
                # 在chunk_size范围内寻找句号、问号或感叹号
                for i in range(end - 1, max(start, end - 100), -1):
                    if content[i] in ['。', '.', '?', '!', '？', '！']:
                        end = i + 1
                        break
            
            # 添加当前块
            chunks.append(content[start:end])
            
            # 更新下一块的起始位置，考虑重叠
            start = end - overlap
            
            # 确保起始位置不会倒退
            start = max(start, 0)
        
        logger.info(f"文档分割完成，共 {len(chunks)} 个块")
        return chunks
    
    def process_document(self, file_path: Optional[str] = None, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """处理文档，包括读取和分割
        
        Args:
            file_path: 文档文件路径，如果为None则读取所有文档
            chunk_size: 每块的最大字符数
            overlap: 相邻块之间的重叠字符数
            
        Returns:
            处理后的文档块列表
            
        Raises:
            DocumentError: 处理文档失败时抛出
        """
        try:
            # 读取文档内容
            if file_path:
                content = self.read_document(file_path)
            else:
                content = self.read_all_documents()
            
            # 分割文档
            chunks = self.split_document(content, chunk_size, overlap)
            
            return chunks
        except DocumentError:
            # 重新抛出DocumentError
            raise
        except Exception as e:
            error_msg = f"处理文档时发生错误: {str(e)}"
            logger.error(error_msg)
            raise DocumentError(error_msg) 