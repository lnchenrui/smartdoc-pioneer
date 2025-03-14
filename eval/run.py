"""评估运行脚本

这个脚本用于运行RAG系统的评估。
"""

import os
import json
import argparse

from app.utils.logger import logger
from eval.ragas.evaluator import RagasEvaluator


def create_example_data(eval_data_path: str):
    """创建示例评估数据
    
    Args:
        eval_data_path: 评估数据文件路径
    """
    logger.info(f"创建示例评估数据: {eval_data_path}")
    
    # 创建目录
    os.makedirs(os.path.dirname(eval_data_path), exist_ok=True)
    
    # 创建示例评估数据
    example_data = [
        {
            "question": "百万英镑故事的主要情节是什么？",
            "reference_answer": "百万英镑故事讲述了一个名叫亨利·亚当斯的美国人偶然获得一张百万英镑的钞票，并在伦敦经历了一系列奇遇的故事。两位富有的兄弟打赌，给予亨利这张钞票一个月，看他能否在不兑换的情况下生存下来。由于拥有这张钞票，亨利获得了信誉，能够获取食物、衣物和住所，甚至在股票市场上获利。最终，亨利成功地度过了这一个月，还与一位年轻女士坠入爱河，并证明了富有兄弟中认为人性本善的那位是正确的。"
        },
        {
            "question": "百万英镑故事中两位富翁兄弟的赌注是什么？",
            "reference_answer": "在百万英镑故事中，两位富翁兄弟奥利弗和罗德里克·蒙特古打赌，争论一个身无分文的人如果得到一张不能兑换的百万英镑钞票能否在伦敦生存一个月。奥利弗认为此人会饿死，而罗德里克则相信此人能够利用钞票获得信誉生存下来。他们给予亨利·亚当斯这张钞票进行实验，并承诺如果他成功度过一个月，将给予他一份工作。"
        }
    ]
    
    # 保存示例数据
    with open(eval_data_path, 'w', encoding='utf-8') as f:
        json.dump(example_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"示例评估数据创建完成: {eval_data_path}")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行RAG系统评估')
    parser.add_argument('--data', type=str, default='eval/eval_data.json', help='评估数据文件路径')
    parser.add_argument('--create-example', action='store_true', help='是否创建示例数据')
    args = parser.parse_args()
    
    eval_data_path = args.data
    
    # 检查评估数据是否存在，如果不存在且指定了创建示例数据，则创建示例数据
    if not os.path.exists(eval_data_path):
        if args.create_example:
            create_example_data(eval_data_path)
        else:
            logger.error(f"评估数据文件不存在: {eval_data_path}")
            logger.info("可以使用 --create-example 参数创建示例数据")
            return
    
    # 创建评估器并运行评估
    logger.info("创建Ragas评估器")
    evaluator = RagasEvaluator()
    
    # 运行评估
    logger.info("开始运行评估")
    eval_results = evaluator.evaluate(eval_data_path)
    
    # 生成评估报告
    logger.info("生成评估报告")
    report = evaluator.generate_report(eval_results)
    
    # 打印报告预览
    print("\n评估报告预览:\n" + "="*50)
    print(report[:500] + "...\n" + "="*50)
    logger.info("评估完成")


if __name__ == "__main__":
    main() 