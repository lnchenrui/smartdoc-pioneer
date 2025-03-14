"""SmartDoc Pioneer 应用

这个模块是应用的入口点，负责初始化应用。
"""

from app.utils.logging.logger import get_logger

logger = get_logger("app")

# 版本信息
__version__ = '1.0.0'

logger.info(f"SmartDoc Pioneer 应用初始化，版本: {__version__}")

# 导入依赖注入容器
from app.di.container import get_container

# 初始化容器
container = get_container()

# 导入路由
from app.api.routes import chat, search, index

# 导入API文档
from app.api.docs import docs_bp, swagger_bp 