"""API文档模块

这个模块提供了API文档功能，使用Swagger生成API文档。
"""

from flask import Blueprint, jsonify
from flask_swagger_ui import get_swaggerui_blueprint

from app.utils.logging.logger import get_logger

logger = get_logger("api.docs")

# 创建API文档蓝图
docs_bp = Blueprint('docs', __name__)

# 创建Swagger UI蓝图
swagger_bp = get_swaggerui_blueprint(
    '/api/docs',
    '/api/swagger.json',
    config={
        'app_name': "SmartDoc Pioneer API"
    }
)

@docs_bp.route('/swagger.json', methods=['GET'])
def swagger_json():
    """返回Swagger JSON配置"""
    swagger_config = {
        "swagger": "2.0",
        "info": {
            "title": "SmartDoc Pioneer API",
            "description": "SmartDoc Pioneer API文档",
            "version": "1.0.0"
        },
        "basePath": "/api",
        "schemes": [
            "http",
            "https"
        ],
        "consumes": [
            "application/json"
        ],
        "produces": [
            "application/json"
        ],
        "paths": {
            "/chat/local": {
                "post": {
                    "summary": "本地文档聊天API",
                    "description": "使用datasets文件夹中的文档作为上下文，支持多轮对话。支持流式和非流式响应。",
                    "parameters": [
                        {
                            "name": "body",
                            "in": "body",
                            "required": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "messages": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "role": {
                                                    "type": "string",
                                                    "enum": ["system", "user", "assistant"]
                                                },
                                                "content": {
                                                    "type": "string"
                                                }
                                            }
                                        }
                                    },
                                    "stream": {
                                        "type": "boolean",
                                        "default": False
                                    }
                                }
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "成功响应",
                            "schema": {
                                "type": "object"
                            }
                        },
                        "400": {
                            "description": "请求错误"
                        },
                        "500": {
                            "description": "服务器错误"
                        }
                    }
                }
            },
            "/chat/rag": {
                "post": {
                    "summary": "RAG聊天API",
                    "description": "使用检索增强生成（RAG）技术回答问题，支持流式和非流式响应。",
                    "parameters": [
                        {
                            "name": "body",
                            "in": "body",
                            "required": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "messages": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "role": {
                                                    "type": "string",
                                                    "enum": ["system", "user", "assistant"]
                                                },
                                                "content": {
                                                    "type": "string"
                                                }
                                            }
                                        }
                                    },
                                    "stream": {
                                        "type": "boolean",
                                        "default": False
                                    }
                                }
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "成功响应",
                            "schema": {
                                "type": "object"
                            }
                        },
                        "400": {
                            "description": "请求错误"
                        },
                        "500": {
                            "description": "服务器错误"
                        }
                    }
                }
            },
            "/search": {
                "post": {
                    "summary": "搜索API",
                    "description": "搜索文档内容",
                    "parameters": [
                        {
                            "name": "body",
                            "in": "body",
                            "required": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string"
                                    },
                                    "top_k": {
                                        "type": "integer",
                                        "default": 5
                                    }
                                }
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "成功响应",
                            "schema": {
                                "type": "object"
                            }
                        },
                        "400": {
                            "description": "请求错误"
                        },
                        "500": {
                            "description": "服务器错误"
                        }
                    }
                }
            }
        },
        "definitions": {
            "Message": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["system", "user", "assistant"]
                    },
                    "content": {
                        "type": "string"
                    }
                }
            },
            "Error": {
                "type": "object",
                "properties": {
                    "error": {
                        "type": "string"
                    },
                    "code": {
                        "type": "integer"
                    }
                }
            }
        }
    }
    
    return jsonify(swagger_config) 