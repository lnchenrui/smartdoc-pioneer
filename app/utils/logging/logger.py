import logging
import os
from typing import Optional

# 配置日志系统
def configure_logging(log_level: str = 'INFO'):
    """配置日志系统
    
    Args:
        log_level: 日志级别，默认为INFO
    """
    # 转换日志级别
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    level = level_map.get(log_level, logging.INFO)
    
    # 配置基本日志
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )


# 初始化日志配置
configure_logging()


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """设置日志系统
    
    Args:
        log_level: 日志级别，默认为INFO
        log_file: 日志文件路径，如果为None则只输出到控制台
    """
    # 转换日志级别
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    level = level_map.get(log_level, logging.INFO)
    
    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 记录日志设置
    logging.info(f"日志系统已设置，级别: {log_level}, 文件: {log_file or '无'}")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取日志器
    
    Args:
        name: 日志器名称，如果为None则返回根日志器
        
    Returns:
        日志器实例
    """
    return logging.getLogger(name)


# 创建默认日志器，方便直接导入使用
logger = get_logger('smartdoc')