# RAG应用质量评估工具

本工具基于Ragas框架，用于评估检索增强生成（RAG）应用的质量。通过多维度指标分析，帮助开发者了解RAG系统的性能并找出改进点。

## 评估指标说明

本评估工具使用以下Ragas指标：

- **Faithfulness (忠实度)**: 评估生成的答案是否忠实于提供的上下文，不包含虚构信息
- **Answer Relevancy (答案相关性)**: 评估生成的答案与问题的相关程度
- **Context Relevancy (上下文相关性)**: 评估检索的上下文与问题的相关程度
- **Context Recall (上下文召回率)**: 评估检索的上下文是否包含回答问题所需的所有信息
- **Context Precision (上下文精确度)**: 评估检索的上下文中有多少是与问题相关的

## 使用方法

### 1. 安装依赖

确保已安装所有必要的依赖：

```bash
pip install -r requirements.txt
```

### 2. 准备评估数据

评估数据应为JSON格式，包含问题和参考答案：

```json
[
  {
    "question": "问题1",
    "reference_answer": "参考答案1"
  },
  {
    "question": "问题2",
    "reference_answer": "参考答案2"
  }
]
```

项目已包含示例评估数据文件 `eval_data.json`。

### 3. 运行评估

使用以下命令运行评估：

```bash
python eval/run_evaluation.py --data eval/eval_data.json --output-dir eval/results
```

参数说明：
- `--data`: 评估数据集文件路径 (默认: eval/eval_data.json)
- `--output-dir`: 评估结果输出目录 (默认: eval/results)
- `--verbose`: 显示详细日志

### 4. 查看评估结果

评估完成后，结果将保存在指定的输出目录中：
- 评估指标分数: `ragas_eval_YYYYMMDD_HHMMSS.json`
- 详细评估报告: `ragas_report_YYYYMMDD_HHMMSS.md`

## 自定义评估

如需自定义评估过程，可以修改 `ragas_evaluator.py` 文件：

- 添加或修改评估指标
- 调整评估数据处理逻辑
- 自定义评估报告格式

## 注意事项

- 评估过程可能需要较长时间，取决于评估数据量和LLM响应速度
- 确保评估数据中的问题与知识库内容相关，以获得有意义的评估结果
- 参考答案应尽可能全面准确，作为评估标准