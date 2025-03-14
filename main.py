"""SmartDoc Pioneer 应用入口

这个模块是应用的入口点，负责创建和运行Flask应用。
"""

import os
from flask import Flask

from app.utils.logging.logger import get_logger
from app.utils.config.loader import ConfigLoader
from app.di.container import get_container

logger = get_logger("main")

def create_app():
    """创建Flask应用
    
    Returns:
        Flask应用实例
    """
    # 获取容器
    container = get_container()
    
    # 加载配置
    config_path = os.environ.get('CONFIG_PATH', 'config/settings.yaml')
    config = ConfigLoader.load_default()
    
    # 更新容器配置
    container.config.from_dict(config)
    
    # 创建Flask应用
    app = Flask(__name__)
    
    # 配置应用
    app.config['JSON_AS_ASCII'] = False
    app.config['DEBUG'] = config.get('api', {}).get('debug', False)
    
    # 注册蓝图
    from app.api.routes.chat import chat_bp
    from app.api.routes.search import search_bp
    from app.api.routes.index import index_bp
    
    app.register_blueprint(chat_bp, url_prefix='/api/chat')
    app.register_blueprint(search_bp, url_prefix='/api/search')
    app.register_blueprint(index_bp, url_prefix='/api/index')
    
    # 设置容器
    app.container = container
    
    logger.info("Flask应用创建完成")
    return app


if __name__ == '__main__':
    # 创建应用
    app = create_app()
    
    # 获取配置
    config = ConfigLoader.load_default()
    host = config.get('api', {}).get('host', '0.0.0.0')
    port = config.get('api', {}).get('port', 5000)
    debug = config.get('api', {}).get('debug', False)
    
    # 运行应用
    logger.info(f"启动应用服务器，监听: {host}:{port}, 调试模式: {debug}")
    app.run(host=host, port=port, debug=debug) 