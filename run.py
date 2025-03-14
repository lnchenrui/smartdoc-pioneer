"""应用程序入口文件

这个文件是应用程序的入口点，用于启动Flask应用程序。
"""

import os
import argparse
from app.factory import create_app
from app.utils.logging.logger import get_logger

logger = get_logger("run")

def parse_args():
    """解析命令行参数
    
    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(description='启动SmartDoc Pioneer应用')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--host', type=str, help='主机地址')
    parser.add_argument('--port', type=int, help='端口号')
    parser.add_argument('--debug', action='store_true', help='是否启用调试模式')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建应用程序
    app = create_app(args.config)
    
    # 获取运行参数
    host = args.host or app.config.get('HOST', '0.0.0.0')
    port = args.port or app.config.get('PORT', 5000)
    debug = args.debug or app.config.get('DEBUG', False)
    
    # 启动应用程序
    logger.info(f"启动应用程序，主机: {host}, 端口: {port}, 调试模式: {debug}")
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    main() 