"""API路由包

这个包包含了应用的API路由模块。
"""

from app.utils.logging.logger import get_logger

logger = get_logger("api.routes")

logger.info("API路由模块初始化")

# 导入路由模块
from app.api.routes import chat, search, index 