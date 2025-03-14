import os
import yaml
from typing import Dict, Any, Optional

# 全局配置缓存
_config_cache = None

def get_config() -> Dict[str, Any]:
    """获取配置
    
    如果配置已加载，则返回缓存的配置，否则加载默认配置
    
    Returns:
        配置字典
    """
    global _config_cache
    if _config_cache is None:
        _config_cache = ConfigLoader.load_default()
    return _config_cache

class ConfigLoader:
    """配置加载器，负责加载配置文件"""
    
    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """加载YAML配置文件
        
        Args:
            file_path: 配置文件路径
            
        Returns:
            配置字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")
            return {}
    
    @staticmethod
    def load_default() -> Dict[str, Any]:
        """加载默认配置
        
        Returns:
            配置字典
        """
        # 默认配置文件路径
        config_path = os.environ.get('CONFIG_PATH', 'config/settings.yaml')
        
        # 加载配置文件
        config = ConfigLoader.load_yaml(config_path)
        
        # 处理环境变量替换
        config = ConfigLoader._process_env_vars(config)
        
        return config
    
    @staticmethod
    def _process_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
        """处理配置中的环境变量引用
        
        Args:
            config: 配置字典
            
        Returns:
            处理后的配置字典
        """
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, dict):
                    config[key] = ConfigLoader._process_env_vars(value)
                elif isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    # 处理 ${ENV_VAR} 格式的环境变量
                    env_var = value[2:-1]
                    if ':' in env_var:
                        # 处理 ${ENV_VAR:default} 格式
                        env_var, default = env_var.split(':', 1)
                        config[key] = os.environ.get(env_var, default)
                    else:
                        # 处理 ${ENV_VAR} 格式
                        config[key] = os.environ.get(env_var, '')
        return config

def reload_config() -> Dict[str, Any]:
    """重新加载配置
    
    Returns:
        配置字典
    """
    global _config_cache
    _config_cache = ConfigLoader.load_default()
    return _config_cache