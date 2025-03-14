"""LLM响应处理模块

这个模块提供了处理LLM响应的功能。
"""

import json
from typing import Generator, Dict, Any, Union

from app.utils.logging.logger import get_logger
from app.utils.exceptions import LLMError
from app.core.llm.models import LLMStreamChunk

logger = get_logger("llm.response")

class LLMResponseHandler:
    """LLM响应处理器，处理LLM的响应"""
    
    @staticmethod
    def handle_stream_response(response) -> Generator[Dict[str, Any], None, None]:
        """处理流式响应
        
        Args:
            response: LLM的原始响应
            
        Returns:
            处理后的响应生成器
            
        Raises:
            LLMError: 处理响应失败时抛出
        """
        logger.info("开始处理流式响应")
        
        try:
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    
                    # 只处理数据行
                    if line_text.startswith('data: '):
                        data_text = line_text[6:].strip()
                        
                        # 跳过心跳消息
                        if data_text == '[DONE]':
                            logger.debug("收到[DONE]标记，流式响应结束")
                            yield {'done': True}
                            continue
                        
                        try:
                            # 尝试解析JSON
                            data_json = json.loads(data_text)
                            yield data_json
                        except json.JSONDecodeError:
                            logger.warning(f"无法解析JSON: {data_text}")
                            # 如果无法解析，则传递原始文本
                            yield {'content': data_text, 'raw': True}
            
            logger.info("流式响应处理完成")
        except Exception as e:
            error_msg = f"处理流式响应时出错: {str(e)}"
            logger.error(error_msg)
            yield {'error': str(e)}
            raise LLMError(error_msg)
    
    @staticmethod
    def handle_normal_response(response) -> Dict[str, Any]:
        """处理非流式响应
        
        Args:
            response: LLM的原始响应
            
        Returns:
            处理后的响应字典
            
        Raises:
            LLMError: 处理响应失败时抛出
        """
        logger.info("开始处理非流式响应")
        
        try:
            response_json = response.json()
            logger.info("非流式响应处理完成")
            return response_json
        except Exception as e:
            error_msg = f"处理非流式响应时出错: {str(e)}"
            logger.error(error_msg)
            raise LLMError(error_msg)
    
    @staticmethod
    def extract_content_from_response(response_json: Dict[str, Any]) -> str:
        """从响应JSON中提取内容
        
        Args:
            response_json: 响应JSON
            
        Returns:
            提取的内容
            
        Raises:
            LLMError: 提取内容失败时抛出
        """
        try:
            # 尝试从标准格式中提取
            if 'choices' in response_json and len(response_json['choices']) > 0:
                choice = response_json['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    return choice['message']['content']
                elif 'delta' in choice and 'content' in choice['delta']:
                    return choice['delta']['content']
            
            # 尝试从其他可能的位置提取
            if 'content' in response_json:
                return response_json['content']
            
            # 如果找不到内容，返回空字符串
            logger.warning(f"无法从响应中提取内容: {response_json}")
            return ""
        except Exception as e:
            error_msg = f"提取内容时出错: {str(e)}"
            logger.error(error_msg)
            raise LLMError(error_msg) 