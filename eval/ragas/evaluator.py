"""Ragas评估器模块

这个模块提供了基于Ragas的RAG系统评估功能。
"""

import os
import json
import time
import pandas as pd
from typing import List, Dict, Any, Optional

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
    context_precision
)
from ragas import evaluate
from datasets import Dataset

from app.utils.logger import logger
from app.services.rag_service import rag_service


class RagasEvaluator:
    """Ragas评估器，用于评估RAG应用的质量"""
    
    def __init__(self, rag_service_instance=None):
        """初始化Ragas评估器
        
        Args:
            rag_service_instance: RAG服务实例，如果为None则使用默认实例
        """
        self.rag_service = rag_service_instance or rag_service
        self.results_dir = os.path.join('eval', 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        logger.info("Ragas评估器初始化完成")
    
    def prepare_evaluation_data(self, eval_data_path: str) -> Dataset:
        """准备评估数据集
        
        Args:
            eval_data_path: 评估数据集文件路径，应为JSON格式，包含问题和参考答案
            
        Returns:
            处理后的数据集
        """
        logger.info(f"开始准备评估数据，数据文件: {eval_data_path}")
        
        try:
            # 读取评估数据
            with open(eval_data_path, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
            
            # 转换为Ragas所需的格式
            processed_data = []
            for item in eval_data:
                query = item['question']
                
                # 使用RAG系统生成答案
                logger.info(f"为问题生成答案: {query}")
                search_results = self.rag_service.search(query)
                answer = self.rag_service.process(query)
                
                # 提取上下文
                contexts = self._extract_contexts(search_results)
                
                processed_item = {
                    'question': query,
                    'answer': answer,
                    'contexts': contexts,
                    'ground_truth': item.get('reference_answer', '')
                }
                processed_data.append(processed_item)
            
            # 转换为Dataset格式
            dataset = Dataset.from_pandas(pd.DataFrame(processed_data))
            logger.info(f"评估数据准备完成，共 {len(processed_data)} 条数据")
            return dataset
            
        except Exception as e:
            logger.error(f"准备评估数据时出错: {str(e)}")
            raise
    
    def _extract_contexts(self, search_results: str) -> List[str]:
        """从搜索结果中提取上下文内容
        
        Args:
            search_results: 搜索结果文本
            
        Returns:
            提取的上下文列表
        """
        contexts = []
        
        # 如果没有找到相关内容
        if search_results == "未找到相关内容":
            return contexts
        
        # 解析搜索结果，提取内容部分
        lines = search_results.split('\n')
        content_flag = False
        current_content = []
        
        for line in lines:
            if line.startswith("内容："):
                content_flag = True
                continue
            elif line.startswith("-" * 50) and content_flag:
                content_flag = False
                if current_content:
                    contexts.append('\n'.join(current_content))
                    current_content = []
            elif content_flag:
                current_content.append(line)
        
        # 处理最后一个内容块
        if current_content:
            contexts.append('\n'.join(current_content))
        
        return contexts
    
    def evaluate(self, eval_data_path: str) -> Dict[str, Any]:
        """评估RAG应用的质量
        
        Args:
            eval_data_path: 评估数据集文件路径
            
        Returns:
            评估结果字典
        """
        logger.info("开始评估RAG应用质量")
        start_time = time.time()
        
        try:
            # 准备评估数据
            dataset = self.prepare_evaluation_data(eval_data_path)
            
            # 定义评估指标
            metrics = [
                faithfulness,
                answer_relevancy,
                context_relevancy,
                context_recall,
                context_precision
            ]
            
            # 执行评估
            logger.info("开始执行Ragas评估")
            result = evaluate(dataset, metrics)
            
            # 保存评估结果
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join(self.results_dir, f"ragas_eval_{timestamp}.json")
            
            # 转换评估结果为可序列化的格式
            result_dict = {}
            for metric_name, score in result.items():
                result_dict[metric_name] = float(score)
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
            
            elapsed = time.time() - start_time
            logger.info(f"评估完成，耗时: {elapsed:.2f}秒，结果已保存至: {result_path}")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"评估过程中出错: {str(e)}")
            raise
    
    def generate_report(self, eval_results: Dict[str, Any]) -> str:
        """生成评估报告
        
        Args:
            eval_results: 评估结果字典
            
        Returns:
            评估报告文本
        """
        logger.info("开始生成评估报告")
        
        report_lines = [
            "# RAG应用质量评估报告",
            f"\n## 评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## 评估指标说明",
            "- **Faithfulness (忠实度)**: 生成的答案是否忠实于提供的上下文，不包含虚构信息",
            "- **Answer Relevancy (答案相关性)**: 生成的答案与问题的相关程度",
            "- **Context Relevancy (上下文相关性)**: 检索的上下文与问题的相关程度",
            "- **Context Recall (上下文召回率)**: 检索的上下文是否包含回答问题所需的所有信息",
            "- **Context Precision (上下文精确度)**: 检索的上下文中有多少是与问题相关的",
            "\n## 评估结果"
        ]
        
        # 添加评估分数
        for metric, score in eval_results.items():
            report_lines.append(f"- **{metric}**: {score:.4f}")
        
        # 添加结果分析
        report_lines.append("\n## 结果分析")
        
        # 分析忠实度
        faithfulness_score = eval_results.get("faithfulness", 0)
        if faithfulness_score < 0.7:
            report_lines.append("- **忠实度较低**: 生成的答案可能包含上下文中不存在的信息，建议调整LLM参数减少幻觉")
        else:
            report_lines.append("- **忠实度良好**: 生成的答案基本忠实于提供的上下文")
        
        # 分析答案相关性
        answer_rel_score = eval_results.get("answer_relevancy", 0)
        if answer_rel_score < 0.7:
            report_lines.append("- **答案相关性较低**: 生成的答案可能未充分回答用户问题，建议优化提示模板")
        else:
            report_lines.append("- **答案相关性良好**: 生成的答案与用户问题高度相关")
        
        # 分析上下文相关性和精确度
        context_rel_score = eval_results.get("context_relevancy", 0)
        context_prec_score = eval_results.get("context_precision", 0)
        if context_rel_score < 0.7 or context_prec_score < 0.7:
            report_lines.append("- **检索质量有待提高**: 检索的上下文与问题相关性不足，建议优化检索策略或向量模型")
        else:
            report_lines.append("- **检索质量良好**: 检索的上下文与问题高度相关")
        
        # 添加改进建议
        report_lines.append("\n## 改进建议")
        report_lines.append("1. **优化检索策略**: 考虑调整相似度阈值或使用混合检索方法")
        report_lines.append("2. **改进提示模板**: 优化系统提示和用户提示，引导LLM生成更相关的回答")
        report_lines.append("3. **扩充知识库**: 增加更多高质量文档，提高知识覆盖面")
        report_lines.append("4. **调整LLM参数**: 尝试不同的温度和top_p值，平衡创造性和准确性")
        
        report = '\n'.join(report_lines)
        
        # 保存报告
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.results_dir, f"ragas_report_{timestamp}.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"评估报告生成完成，已保存至: {report_path}")
        return report 