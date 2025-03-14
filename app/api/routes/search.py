"""搜索API路由模块

这个模块提供了搜索相关的API路由。
"""

from flask import Blueprint, request, jsonify
from app.di.container import get_container
from app.utils.error_handler import handle_api_error, ValidationError
from app.utils.logging.logger import get_logger

logger = get_logger("app.api.routes.search")
search_bp = Blueprint('search', __name__)

@search_bp.route('', methods=['POST'])
def search():
    """搜索API
    
    搜索文档库中的内容。
    
    请求格式:
    {
        "query": "搜索查询",
        "top_k": 5,  // 可选，默认为5
        "filter": {  // 可选
            "source": "文档来源",
            "type": "文档类型"
        }
    }
    
    响应格式:
    {
        "results": [
            {
                "content": "文档内容",
                "metadata": {
                    "title": "文档标题",
                    "source": "文档来源",
                    "relevance": 0.95
                }
            }
        ]
    }
    
    Returns:
        搜索结果的JSON响应
    """
    try:
        # 解析请求
        data = request.json
        if not data:
            raise ValidationError("请求体不能为空")
        
        query = data.get('query')
        top_k = data.get('top_k', 5)
        filter_dict = data.get('filter', {})
        
        if not query:
            raise ValidationError("搜索查询不能为空")
        
        # 记录请求
        logger.info(f"收到搜索请求: query='{query}', top_k={top_k}")
        
        # 获取搜索服务
        search_service = get_container().get("search_service")
        
        # 执行搜索
        results = search_service.search(query, top_k=top_k, filter_dict=filter_dict)
        
        # 格式化结果
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.content,
                "metadata": doc.metadata
            })
        
        return jsonify({"results": formatted_results})
    
    except Exception as e:
        return handle_api_error(e) 