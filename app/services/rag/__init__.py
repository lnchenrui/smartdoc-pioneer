"""RAG服务包

这个包提供了RAG（检索增强生成）服务的功能。
"""

from app.utils.logging.logger import get_logger

logger = get_logger("services.rag")

logger.info("RAG服务模块初始化")

# 导出常用类
from app.services.rag.service import RAGService 