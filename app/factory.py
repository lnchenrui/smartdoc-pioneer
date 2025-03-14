"""应用程序工厂模块

这个模块提供了创建Flask应用程序实例的工厂函数。
"""

import os
from typing import Dict, Any, Optional

from flask import Flask, jsonify
from flask_cors import CORS

from app.config import load_config
from app.di.container import get_container, initialize_container
from app.utils.error_handler import handle_api_error, APIError, ValidationError, ServiceError
from app.utils.logging.logger import setup_logging, get_logger

logger = get_logger("app.factory")

def create_app(config_path: Optional[str] = None) -> Flask:
    """创建Flask应用程序实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Flask应用程序实例
    """
    # 创建应用程序实例
    app = Flask(__name__)
    
    # 配置应用程序
    configure_app(app, config_path)
    
    # 设置CORS
    CORS(app)
    
    # 初始化服务
    initialize_services(app)
    
    # 注册蓝图
    register_blueprints(app)
    
    # 注册错误处理器
    register_error_handlers(app)
    
    logger.info("应用程序初始化完成")
    
    return app

def configure_app(app: Flask, config_path: Optional[str] = None):
    """配置应用程序
    
    Args:
        app: Flask应用程序实例
        config_path: 配置文件路径
    """
    # 加载配置
    config = load_config(config_path)
    
    # 设置Flask配置
    app.config.update(config.get('flask', {}))
    
    # 设置日志
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    log_file = log_config.get('file', 'app.log')
    setup_logging(log_level, log_file)
    
    # 初始化容器并设置配置
    initialize_container(config)
    app.container = get_container()
    
    logger.info(f"应用程序配置完成，环境: {app.config.get('ENV', 'development')}")

def initialize_services(app: Flask):
    """初始化服务
    
    Args:
        app: Flask应用程序实例
    """
    # 服务已在容器初始化时创建
    logger.info("服务初始化完成")

def register_blueprints(app: Flask):
    """注册蓝图
    
    Args:
        app: Flask应用程序实例
    """
    # 导入蓝图
    from app.api.chat import chat_bp
    from app.api.search import search_bp
    from app.api.index import index_bp
    
    # 注册蓝图
    app.register_blueprint(chat_bp, url_prefix='/api/chat')
    app.register_blueprint(search_bp, url_prefix='/api/search')
    app.register_blueprint(index_bp, url_prefix='/api/index')
    
    logger.info("蓝图注册完成")

def register_error_handlers(app: Flask):
    """注册错误处理器
    
    Args:
        app: Flask应用程序实例
    """
    # 注册API错误处理器
    @app.errorhandler(APIError)
    def handle_api_exception(error):
        return handle_api_error(error)
    
    # 注册验证错误处理器
    @app.errorhandler(ValidationError)
    def handle_validation_exception(error):
        return handle_api_error(error)
    
    # 注册服务错误处理器
    @app.errorhandler(ServiceError)
    def handle_service_exception(error):
        return handle_api_error(error)
    
    # 注册404错误处理器
    @app.errorhandler(404)
    def handle_not_found(error):
        return jsonify({
            "status": "error",
            "code": 404,
            "message": "资源不存在"
        }), 404
    
    # 注册500错误处理器
    @app.errorhandler(500)
    def handle_server_error(error):
        logger.error(f"服务器错误: {str(error)}", exc_info=True)
        return jsonify({
            "status": "error",
            "code": 500,
            "message": "服务器内部错误"
        }), 500
    
    logger.info("错误处理器注册完成") 