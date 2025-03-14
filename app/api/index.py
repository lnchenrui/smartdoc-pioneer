"""索引API路由模块

这个模块提供了索引相关的API路由。
"""

from typing import Dict, Any, List, Optional
import os
import time
import uuid

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from app.di.container import container
from app.utils.error_handler import ValidationError
from app.utils.logging.logger import get_logger

logger = get_logger("api.index")

# 创建蓝图
index_bp = Blueprint('index', __name__)

@index_bp.route('/document', methods=['POST'])
def index_document():
    """索引文档API
    
    处理文档索引请求并返回结果。
    
    Returns:
        JSON响应
    """
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            raise ValidationError("没有上传文件")
        
        file = request.files['file']
        if file.filename == '':
            raise ValidationError("文件名为空")
        
        # 获取元数据
        metadata = {}
        if 'metadata' in request.form:
            metadata_str = request.form['metadata']
            try:
                import json
                metadata = json.loads(metadata_str)
            except Exception as e:
                logger.warning(f"解析元数据失败: {str(e)}")
        
        # 获取可选参数
        chunk_size = request.form.get('chunk_size')
        if chunk_size:
            try:
                chunk_size = int(chunk_size)
            except ValueError:
                raise ValidationError("chunk_size必须是整数")
        
        chunk_overlap = request.form.get('chunk_overlap')
        if chunk_overlap:
            try:
                chunk_overlap = int(chunk_overlap)
            except ValueError:
                raise ValidationError("chunk_overlap必须是整数")
        
        # 获取索引服务
        index_service = container.get("index_service")
        
        # 保存文件
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        upload_dir = index_service.upload_dir
        file_path = os.path.join(upload_dir, unique_filename)
        file.save(file_path)
        
        # 添加文件元数据
        if not metadata:
            metadata = {}
        metadata.update({
            "filename": filename,
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_path": file_path
        })
        
        # 索引文档
        start_time = time.time()
        chunk_count, processing_time = index_service.index_document(
            file_path=file_path,
            metadata=metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        elapsed = time.time() - start_time
        
        # 构建响应
        result = {
            "status": "success",
            "data": {
                "filename": filename,
                "chunks": chunk_count,
                "metadata": metadata
            },
            "meta": {
                "elapsed_time": elapsed,
                "processing_time": processing_time
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"索引文档API错误: {str(e)}", exc_info=True)
        raise

@index_bp.route('/text', methods=['POST'])
def index_text():
    """索引文本API
    
    处理文本索引请求并返回结果。
    
    Returns:
        JSON响应
    """
    try:
        # 获取请求数据
        data = request.get_json()
        if not data:
            raise ValidationError("请求体不能为空")
        
        # 验证必要字段
        if 'text' not in data:
            raise ValidationError("缺少必要字段: text")
        
        text = data.get('text')
        if not text or not isinstance(text, str):
            raise ValidationError("text必须是非空字符串")
        
        # 获取可选参数
        metadata = data.get('metadata', {})
        chunk_size = data.get('chunk_size')
        chunk_overlap = data.get('chunk_overlap')
        
        # 获取索引服务
        index_service = container.get("index_service")
        
        # 索引文本
        start_time = time.time()
        chunk_count, processing_time = index_service.index_text(
            text=text,
            metadata=metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        elapsed = time.time() - start_time
        
        # 构建响应
        result = {
            "status": "success",
            "data": {
                "chunks": chunk_count,
                "metadata": metadata
            },
            "meta": {
                "elapsed_time": elapsed,
                "processing_time": processing_time
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"索引文本API错误: {str(e)}", exc_info=True)
        raise

@index_bp.route('/delete', methods=['POST'])
def delete_documents():
    """删除文档API
    
    处理文档删除请求并返回结果。
    
    Returns:
        JSON响应
    """
    try:
        # 获取请求数据
        data = request.get_json()
        if not data:
            raise ValidationError("请求体不能为空")
        
        # 验证必要字段
        if 'filter' not in data:
            raise ValidationError("缺少必要字段: filter")
        
        filter_dict = data.get('filter')
        if not filter_dict or not isinstance(filter_dict, dict):
            raise ValidationError("filter必须是非空字典")
        
        # 获取索引服务
        index_service = container.get("index_service")
        
        # 删除文档
        start_time = time.time()
        deleted_count = index_service.delete_documents(filter_dict)
        elapsed = time.time() - start_time
        
        # 构建响应
        result = {
            "status": "success",
            "data": {
                "deleted_count": deleted_count,
                "filter": filter_dict
            },
            "meta": {
                "elapsed_time": elapsed
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"删除文档API错误: {str(e)}", exc_info=True)
        raise

@index_bp.route('/count', methods=['POST'])
def get_document_count():
    """获取文档数量API
    
    处理获取文档数量请求并返回结果。
    
    Returns:
        JSON响应
    """
    try:
        # 获取请求数据
        data = request.get_json()
        filter_dict = None
        
        if data and 'filter' in data:
            filter_dict = data.get('filter')
            if not isinstance(filter_dict, dict):
                raise ValidationError("filter必须是字典")
        
        # 获取索引服务
        index_service = container.get("index_service")
        
        # 获取文档数量
        start_time = time.time()
        count = index_service.get_document_count(filter_dict)
        elapsed = time.time() - start_time
        
        # 构建响应
        result = {
            "status": "success",
            "data": {
                "count": count,
                "filter": filter_dict
            },
            "meta": {
                "elapsed_time": elapsed
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"获取文档数量API错误: {str(e)}", exc_info=True)
        raise 