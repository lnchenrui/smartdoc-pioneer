"""LLM客户端工厂模块

这个模块提供了创建LLM客户端的工厂功能。
"""

from typing import Dict, Any

from app.utils.logging.logger import get_logger
from app.utils.exceptions import ConfigError
from app.core.llm.client import LLMClient

logger = get_logger("llm.factory")

class LLMClientFactory:
    """LLM客户端工厂，用于创建LLM客户端实例"""
    
    @staticmethod
    def create_client(config: Dict[str, Any] = None) -> LLMClient:
        """创建LLM客户端实例
        
        Args:
            config: 配置字典，如果为None则使用默认配置
            
        Returns:
            LLM客户端实例
            
        Raises:
            ConfigError: 配置错误时抛出
        """
        if config is None:
            from app.utils.config.loader import ConfigLoader
            config = ConfigLoader.load_default().get('llm', {})
        
        # 验证必要的配置项
        required_fields = ['model', 'api_key', 'endpoint']
        for field in required_fields:
            if field not in config or not config[field]:
                error_msg = f"LLM配置缺少必要的字段: {field}"
                logger.error(error_msg)
                raise ConfigError(error_msg)
        
        # 创建LLM客户端
        client = LLMClient(
            model=config.get('model', ''),
            api_key=config.get('api_key', ''),
            endpoint=config.get('endpoint', ''),
            temperature=config.get('temperature', 0.7),
            max_tokens=config.get('max_tokens', 10240),
            top_p=config.get('top_p', 0.95)
        )
        
        logger.info(f"创建LLM客户端成功，模型: {config.get('model')}")
        return client 