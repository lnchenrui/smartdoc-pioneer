"""配置模块

这个模块提供了配置加载和管理功能。
"""

import os
import yaml
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

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

def process_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """处理配置中的环境变量引用
    
    Args:
        config: 配置字典
        
    Returns:
        处理后的配置字典
    """
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, dict):
                config[key] = process_env_vars(value)
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

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载配置
    
    Args:
        config_path: 配置文件路径，如果为None则使用环境变量或默认路径
        
    Returns:
        配置字典
    """
    # 确定配置文件路径
    if config_path is None:
        config_path = os.environ.get('CONFIG_PATH', 'config/settings.yaml')
    
    # 加载配置文件
    config = load_yaml(config_path)
    
    # 处理环境变量替换
    config = process_env_vars(config)
    
    # 添加默认配置
    config = add_default_config(config)
    
    return config

def add_default_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """添加默认配置
    
    Args:
        config: 配置字典
        
    Returns:
        添加默认配置后的配置字典
    """
    # 确保配置字典中包含必要的部分
    if 'flask' not in config:
        config['flask'] = {}
    
    # 设置默认的Flask配置
    flask_config = config['flask']
    flask_config.setdefault('ENV', os.environ.get('FLASK_ENV', 'development'))
    flask_config.setdefault('DEBUG', flask_config['ENV'] == 'development')
    flask_config.setdefault('SECRET_KEY', os.environ.get('SECRET_KEY', 'dev-key'))
    
    # 设置日志配置
    if 'logging' not in config:
        config['logging'] = {}
    
    logging_config = config['logging']
    logging_config.setdefault('level', os.environ.get('LOG_LEVEL', 'INFO'))
    logging_config.setdefault('file', os.environ.get('LOG_FILE', 'app.log'))
    
    # 设置LLM配置
    if 'llm' not in config:
        config['llm'] = {}
    
    llm_config = config['llm']
    llm_config.setdefault('model_name', os.environ.get('LLM_MODEL', 'gpt-3.5-turbo'))
    llm_config.setdefault('temperature', float(os.environ.get('LLM_TEMPERATURE', '0.7')))
    llm_config.setdefault('max_tokens', int(os.environ.get('LLM_MAX_TOKENS', '1024')))
    llm_config.setdefault('streaming', os.environ.get('LLM_STREAMING', 'true').lower() == 'true')
    llm_config.setdefault('system_message', "你是一个有用的AI助手。")
    
    # 设置嵌入配置
    if 'embedding' not in config:
        config['embedding'] = {}
    
    embedding_config = config['embedding']
    embedding_config.setdefault('model_name', os.environ.get('EMBEDDING_MODEL', 'text-embedding-ada-002'))
    embedding_config.setdefault('batch_size', int(os.environ.get('EMBEDDING_BATCH_SIZE', '32')))
    
    # 设置向量存储配置
    if 'vector_store' not in config:
        config['vector_store'] = {}
    
    vector_store_config = config['vector_store']
    vector_store_config.setdefault('persist_directory', os.environ.get('VECTOR_STORE_DIR', 'chroma_db'))
    
    # 设置搜索配置
    if 'search' not in config:
        config['search'] = {}
    
    search_config = config['search']
    search_config.setdefault('default_top_k', int(os.environ.get('SEARCH_TOP_K', '5')))
    search_config.setdefault('default_search_type', os.environ.get('SEARCH_TYPE', 'similarity'))
    
    # 设置索引配置
    if 'index' not in config:
        config['index'] = {}
    
    index_config = config['index']
    index_config.setdefault('chunk_size', int(os.environ.get('INDEX_CHUNK_SIZE', '1000')))
    index_config.setdefault('chunk_overlap', int(os.environ.get('INDEX_CHUNK_OVERLAP', '200')))
    index_config.setdefault('upload_dir', os.environ.get('UPLOAD_DIR', 'uploads'))
    
    # 设置RAG配置
    if 'rag' not in config:
        config['rag'] = {}
    
    rag_config = config['rag']
    rag_config.setdefault('top_k', int(os.environ.get('RAG_TOP_K', '5')))
    rag_config.setdefault('search_type', os.environ.get('RAG_SEARCH_TYPE', 'similarity'))
    rag_config.setdefault('system_message', "你是一个有用的AI助手，能够根据提供的上下文回答问题。")
    
    # 设置提示配置
    if 'prompt' not in config:
        config['prompt'] = {}
    
    prompt_config = config['prompt']
    prompt_config.setdefault('templates_dir', os.environ.get('PROMPT_TEMPLATES_DIR', 'prompts'))
    
    # 设置文档加载配置
    if 'document_loader' not in config:
        config['document_loader'] = {}
    
    document_loader_config = config['document_loader']
    document_loader_config.setdefault('supported_extensions', [
        '.txt', '.pdf', '.csv', '.md', '.html', '.htm', '.doc', '.docx', 
        '.ppt', '.pptx', '.xls', '.xlsx', '.json', '.xml'
    ])
    
    return config 