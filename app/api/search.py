"""搜索API路由模块

这个模块提供了搜索相关的API路由。
"""

from typing import Dict, Any, List, Optional
import time

from flask import Blueprint, request, jsonify, current_app

from app.di.container import container
from app.utils.error_handler import ValidationError
from app.utils.logging.logger import get_logger

logger = get_logger("api.search")

# 创建蓝图
search_bp = Blueprint('search', __name__)

@search_bp.route('/query', methods=['POST'])
def search_query():
    """搜索查询API
    
    处理搜索请求并返回结果。
    
    Returns:
        JSON响应
    """
    try:
        # 获取请求数据
        data = request.get_json()
        if not data:
            raise ValidationError("请求体不能为空")
        
        # 验证必要字段
        if 'query' not in data:
            raise ValidationError("缺少必要字段: query")
        
        query = data.get('query')
        if not query or not isinstance(query, str):
            raise ValidationError("query必须是非空字符串")
        
        # 获取可选参数
        top_k = data.get('top_k')
        filter_dict = data.get('filter')
        search_type = data.get('search_type')
        
        # 获取搜索服务
        search_service = container.get("search_service")
        
        # 执行搜索
        start_time = time.time()
        results = search_service.search(
            query=query,
            top_k=top_k,
            filter_dict=filter_dict,
            search_type=search_type
        )
        elapsed = time.time() - start_time
        
        # 格式化结果
        formatted_results = search_service.format_search_results(results)
        
        # 构建响应
        result = {
            "status": "success",
            "data": {
                "query": query,
                "results": formatted_results,
                "total": len(formatted_results)
            },
            "meta": {
                "elapsed_time": elapsed,
                "search_type": search_type or "similarity"
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"搜索查询API错误: {str(e)}", exc_info=True)
        raise

@search_bp.route('/hybrid', methods=['POST'])
def hybrid_search():
    """混合搜索API
    
    处理混合搜索请求并返回结果。
    
    Returns:
        JSON响应
    """
    try:
        # 获取请求数据
        data = request.get_json()
        if not data:
            raise ValidationError("请求体不能为空")
        
        # 验证必要字段
        if 'query' not in data:
            raise ValidationError("缺少必要字段: query")
        
        query = data.get('query')
        if not query or not isinstance(query, str):
            raise ValidationError("query必须是非空字符串")
        
        # 获取可选参数
        top_k = data.get('top_k')
        filter_dict = data.get('filter')
        
        # 获取搜索服务
        search_service = container.get("search_service")
        
        # 执行混合搜索
        start_time = time.time()
        results = search_service.hybrid_search(
            query=query,
            top_k=top_k,
            filter_dict=filter_dict
        )
        elapsed = time.time() - start_time
        
        # 格式化结果
        formatted_results = search_service.format_search_results(results)
        
        # 构建响应
        result = {
            "status": "success",
            "data": {
                "query": query,
                "results": formatted_results,
                "total": len(formatted_results)
            },
            "meta": {
                "elapsed_time": elapsed,
                "search_type": "hybrid"
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"混合搜索API错误: {str(e)}", exc_info=True)
        raise

@search_bp.route('/keyword', methods=['POST'])
def keyword_search():
    """关键词搜索API
    
    处理关键词搜索请求并返回结果。
    
    Returns:
        JSON响应
    """
    try:
        # 获取请求数据
        data = request.get_json()
        if not data:
            raise ValidationError("请求体不能为空")
        
        # 验证必要字段
        if 'query' not in data:
            raise ValidationError("缺少必要字段: query")
        
        query = data.get('query')
        if not query or not isinstance(query, str):
            raise ValidationError("query必须是非空字符串")
        
        # 获取可选参数
        top_k = data.get('top_k')
        filter_dict = data.get('filter')
        
        # 获取搜索服务
        search_service = container.get("search_service")
        
        # 执行关键词搜索
        start_time = time.time()
        results = search_service.keyword_search(
            query=query,
            top_k=top_k,
            filter_dict=filter_dict
        )
        elapsed = time.time() - start_time
        
        # 格式化结果
        formatted_results = search_service.format_search_results(results)
        
        # 构建响应
        result = {
            "status": "success",
            "data": {
                "query": query,
                "results": formatted_results,
                "total": len(formatted_results)
            },
            "meta": {
                "elapsed_time": elapsed,
                "search_type": "keyword"
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"关键词搜索API错误: {str(e)}", exc_info=True)
        raise 