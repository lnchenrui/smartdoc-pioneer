"""索引API路由模块

这个模块提供了索引相关的API路由。
"""

from flask import Blueprint, request, jsonify
from app.di.container import get_container
from app.utils.error_handler import handle_api_error, ValidationError
from app.utils.logging.logger import get_logger

logger = get_logger("app.api.routes.index")
index_bp = Blueprint('index', __name__)

@index_bp.route('/index', methods=['POST'])
def index_documents():
    """索引文档API
    
    索引指定路径的文档。
    
    请求格式:
    {
        "path": "文档路径",  // 可选，默认使用配置中的路径
        "recursive": true,   // 可选，是否递归索引子目录，默认为true
        "file_types": [".txt", ".pdf", ".docx"]  // 可选，要索引的文件类型
    }
    
    响应格式:
    {
        "indexed_count": 10,
        "failed_count": 0,
        "message": "索引完成"
    }
    
    Returns:
        索引结果的JSON响应
    """
    try:
        # 解析请求
        data = request.json or {}
        path = data.get('path')
        recursive = data.get('recursive', True)
        file_types = data.get('file_types')
        
        # 记录请求
        logger.info(f"收到索引请求: path='{path}', recursive={recursive}")
        
        # 获取索引服务
        index_service = get_container().get("index_service")
        
        # 执行索引
        indexed_count, failed_count = index_service.index_documents(
            path=path, 
            recursive=recursive,
            file_types=file_types
        )
        
        # 返回结果
        return jsonify({
            "indexed_count": indexed_count,
            "failed_count": failed_count,
            "message": "索引完成"
        })
    
    except Exception as e:
        return handle_api_error(e) 