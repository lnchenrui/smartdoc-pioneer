# SmartDoc Pioneer

SmartDoc Pioneer 是一个基于 RAG（检索增强生成）技术的智能文档助手系统，能够实现智能问答和文档检索功能。系统利用大型语言模型（LLM）和向量数据库，为用户提供准确、相关的文档信息和回答。

## 系统架构

SmartDoc Pioneer 采用模块化设计，主要由以下几个核心组件构成：

```
┌─────────────────────────────────────────────────────────────────┐
│                        SmartDoc Pioneer                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────┼─────────────────────────────────┐
│                               │                                  │
│  ┌─────────────────┐    ┌─────▼──────┐    ┌──────────────────┐  │
│  │     API 层      │◄───┤  服务层    │◄───┤    核心层        │  │
│  └─────────────────┘    └─────┬──────┘    └──────────────────┘  │
│                               │                                  │
│                         ┌─────▼──────┐                          │
│                         │  依赖注入  │                          │
│                         └─────┬──────┘                          │
│                               │                                  │
│                         ┌─────▼──────┐                          │
│                         │  工具层    │                          │
│                         └────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1. 核心层 (Core Layer)

核心层提供基础功能和组件，包括：

- **LLM 模块**：集成多种大型语言模型，如 OpenAI GPT、Anthropic Claude 等
- **向量存储模块**：支持多种向量数据库，如 Elasticsearch、Chroma、FAISS 等
- **嵌入模块**：提供文本嵌入功能，支持 OpenAI、Voyage 等嵌入模型
- **文档处理模块**：处理各种格式的文档，如 PDF、Word、Markdown 等

### 2. 服务层 (Service Layer)

服务层封装业务逻辑，提供高级功能：

- **RAG 服务**：实现检索增强生成功能，结合文档检索和 LLM 生成
- **聊天服务**：处理用户对话，支持多轮对话和上下文管理
- **文档加载服务**：加载和处理各种格式的文档
- **索引服务**：管理文档索引，支持增删改查
- **搜索服务**：提供文档搜索功能，支持语义搜索和关键词搜索
- **提示服务**：管理和生成提示模板，优化 LLM 输出

### 3. API 层 (API Layer)

API 层提供 RESTful 接口，供前端或其他系统调用：

- **聊天 API**：提供聊天功能，支持流式响应
- **RAG API**：提供 RAG 功能，结合文档检索和 LLM 生成
- **文档 API**：提供文档管理功能，如上传、删除等
- **索引 API**：提供索引管理功能，如创建、更新等

### 4. 依赖注入 (Dependency Injection)

依赖注入模块管理系统组件的依赖关系，实现松耦合设计：

- **服务容器**：管理服务实例，提供依赖注入功能
- **配置管理**：加载和管理系统配置

### 5. 工具层 (Utility Layer)

工具层提供通用功能和辅助工具：

- **日志模块**：记录系统日志，支持不同级别的日志
- **错误处理**：统一处理系统错误，提供友好的错误信息
- **配置加载**：加载系统配置，支持不同环境的配置

## 数据流程

SmartDoc Pioneer 的数据流程如下：

```
┌──────────────┐     ┌───────────────┐     ┌───────────────┐
│              │     │               │     │               │
│   文档输入   ├────►│  文档处理     ├────►│  文档索引     │
│              │     │               │     │               │
└──────────────┘     └───────────────┘     └───────┬───────┘
                                                   │
                                                   ▼
┌──────────────┐     ┌───────────────┐     ┌───────────────┐
│              │     │               │     │               │
│   用户查询   ├────►│  查询处理     ├────►│  文档检索     │
│              │     │               │     │               │
└──────────────┘     └───────────────┘     └───────┬───────┘
                                                   │
                                                   ▼
┌──────────────┐     ┌───────────────┐     ┌───────────────┐
│              │     │               │     │               │
│   生成回答   │◄────┤  RAG 处理     │◄────┤  相关文档     │
│              │     │               │     │               │
└──────────────┘     └───────────────┘     └───────────────┘
```

1. **文档处理阶段**：
   - 加载各种格式的文档
   - 分割文档为适当大小的块
   - 生成文档嵌入向量
   - 将文档索引到向量数据库

2. **查询处理阶段**：
   - 接收用户查询
   - 生成查询嵌入向量
   - 在向量数据库中检索相关文档
   - 根据相似度排序文档

3. **RAG 处理阶段**：
   - 将检索到的文档作为上下文
   - 构建 RAG 提示模板
   - 调用 LLM 生成回答
   - 返回生成的回答给用户

## 主要特性

- **多模型支持**：支持多种 LLM 和嵌入模型
- **多向量存储支持**：支持多种向量数据库
- **多文档格式支持**：支持多种文档格式
- **流式响应**：支持流式响应，提供实时反馈
- **多轮对话**：支持多轮对话，保持上下文连贯
- **错误处理**：提供友好的错误处理和日志记录
- **可扩展性**：模块化设计，易于扩展和定制

## 技术栈

- **后端框架**：Flask
- **LLM 集成**：LangChain
- **向量数据库**：Elasticsearch、Chroma、FAISS
- **嵌入模型**：OpenAI、Voyage
- **文档处理**：LangChain Document Loaders
- **依赖注入**：自定义服务容器

## 安装与配置

### 环境要求

- Python 3.9+
- 依赖库（见 requirements.txt）

### 安装步骤

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/smartdoc-pioneer.git
cd smartdoc-pioneer
```

2. 创建虚拟环境：

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

4. 配置环境变量：

复制 `.env.example` 文件为 `.env`，并填写必要的配置：

```bash
cp .env.example .env
```

主要配置项包括：

- LLM API 密钥（如 OpenAI API Key）
- 向量数据库连接信息
- 系统配置参数

### 运行应用

```bash
python run.py
```

默认情况下，应用将在 http://localhost:5000 上运行。

## API 文档

### 聊天 API

#### 普通聊天

```
POST /api/chat/completions
```

请求体：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": "你好，请介绍一下自己。"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": false
}
```

#### RAG 聊天

```
POST /api/chat/rag
```

请求体：

```json
{
  "messages": [
    {"role": "user", "content": "什么是 RAG 技术？"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": false,
  "include_sources": true
}
```

## 项目结构

```
smartdoc-pioneer/
├── app/                    # 应用主目录
│   ├── api/                # API 层
│   │   ├── chat.py         # 聊天 API
│   │   ├── docs.py         # 文档 API
│   │   ├── index.py        # 索引 API
│   │   ├── search.py       # 搜索 API
│   │   └── response/       # 响应处理
│   ├── core/               # 核心层
│   │   ├── llm/            # LLM 模块
│   │   ├── embedding.py    # 嵌入模块
│   │   ├── search.py       # 搜索模块
│   │   └── vector_stores.py # 向量存储模块
│   ├── services/           # 服务层
│   │   ├── chat_service.py # 聊天服务
│   │   ├── rag_service.py  # RAG 服务
│   │   ├── document_loader_service.py # 文档加载服务
│   │   └── vector_store_service.py # 向量存储服务
│   ├── di/                 # 依赖注入
│   │   └── container.py    # 服务容器
│   └── utils/              # 工具层
│       ├── logging/        # 日志模块
│       └── error_handler.py # 错误处理
├── config/                 # 配置文件
├── logs/                   # 日志文件
├── tests/                  # 测试文件
├── .env.example            # 环境变量示例
├── requirements.txt        # 依赖库
└── run.py                  # 应用入口
```

## 贡献指南

欢迎贡献代码、报告问题或提出建议。请遵循以下步骤：

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 联系方式

如有任何问题或建议，请通过以下方式联系我们：

- 电子邮件：your.email@example.com
- GitHub Issues：[https://github.com/yourusername/smartdoc-pioneer/issues](https://github.com/yourusername/smartdoc-pioneer/issues) 