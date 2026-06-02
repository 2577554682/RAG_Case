# RAG Case

基于 LangChain + ChromaDB 的中文检索增强生成（RAG）全流程案例。

## 项目简介

实现完整的 RAG 流水线：从文档加载、文本分块、向量化存储，到语义检索、相关性重排、LLM 生成回答。核心设计面向中文场景，使用 BGE 系列中文优化模型。

### RAG 流水线

```
data/test.docx                # 原始文档
       │
       ▼
┌──────────────────┐
│  UnstructuredLoader        │  文档加载
└──────────────────┘
       │
       ▼
┌──────────────────┐
│  RecursiveCharacterTextSplitter │  文本分块 (chunk_size=500, overlap=20)
└──────────────────┘
       │
       ▼
┌──────────────────┐
│  BGE-large-zh-v1.5          │  向量嵌入 (HuggingFaceEmbeddings)
└──────────────────┘
       │
       ▼
┌──────────────────┐
│  ChromaDB                   │  向量存储
└──────────────────┘
       │
       ▼
┌──────────────────┐
│  向量检索 (top_k=10)        │  初筛
└──────────────────┘
       │
       ▼
┌──────────────────┐
│  BGE-reranker-large         │  重排精筛 (top_n=3)
└──────────────────┘
       │
       ▼
┌──────────────────┐
│  NVIDIA NIM (LLM)           │  生成回答
└──────────────────┘
       │
       ▼
   最终答案
```

## 核心组件

| 阶段 | 组件 | 说明 |
|---|---|---|
| **文档加载** | `UnstructuredLoader` | 支持 `.docx` 等多种格式 |
| **文本分块** | `RecursiveCharacterTextSplitter` | 中文优化分隔符（句号、问号等），块大小 500，重叠 20 |
| **向量嵌入** | `BAAI/bge-large-zh-v1.5` | 中文语义向量模型，1024 维 |
| **向量存储** | `ChromaDB` | 轻量级本地向量数据库 |
| **语义检索** | `Chroma.as_retriever` | 余弦相似度检索，返回 top-10 |
| **相关性重排** | `BAAI/bge-reranker-large` | Cross-encoder 精排，取 top-3 |
| **答案生成** | NVIDIA NIM (`openai/gpt-oss-120b`) | LLM 基于检索片段生成回答 |

## 快速开始

### 前置要求

- Python 3.10+
- NVIDIA API Key（[获取](https://build.nvidia.com)）
- 至少 8GB 显存（GPU 运行嵌入和重排模型）

### 安装

```bash
# 克隆仓库
git clone https://github.com/2577554682/RAG_Case.git
cd RAG_Case

# 安装依赖
pip install langchain-openai langchain-huggingface langchain-chroma \
            langchain-text-splitters langchain-unstructured \
            sentence-transformers chromadb python-dotenv
```

### 配置

```bash
cp .env.example .env
```

编辑 `.env` 填入你的配置：

| 变量 | 说明 | 默认值 |
|---|---|---|
| `NVIDIA_API_KEY` | NVIDIA API 密钥 | — |
| `NVIDIA_BASE_URL` | NVIDIA API 地址 | `https://integrate.api.nvidia.com/v1` |
| `EMBED_MODEL_PATH` | 向量模型路径 | `./models/bge-large-zh-v1.5` |
| `RERANKER_MODEL_PATH` | 重排模型路径 | `./models/bge-reranker-large` |
| `TEST_DOC_PATH` | 测试文档路径 | `./data/test.docx` |
| `CHROMA_DB_PATH` | 向量数据库路径 | `./data/chroma_db` |

### 下载模型

```bash
# 安装 Git LFS
git lfs install

# 向量模型
git clone https://huggingface.co/BAAI/bge-large-zh-v1.5 ./models/bge-large-zh-v1.5

# 重排模型
git clone https://huggingface.co/BAAI/bge-reranker-large ./models/bge-reranker-large
```

### 使用

**1. 文档入库** — 将文档切分、嵌入后存入 ChromaDB：

```bash
python -m rag.text_splitter_chromadb
```

> 将 `data/test.docx` 处理为向量块，默认每块 500 字符。

**2. 检索问答** — 输入问题，触发完整 RAG 流水线：

```bash
python -m rag.retriever
```

> 流程：向量检索 → 重排精筛 → LLM 生成回答，流式输出。

## 项目结构

```
RAG_Case/
├── rag/
│   ├── __init__.py
│   ├── my_llm.py                    # NVIDIA NIM LLM 配置
│   ├── text_splitter_chromadb.py    # 文档分块 + 向量入库
│   └── retriever.py                 # 检索重排 + 答案生成
├── env_utils.py                     # 环境变量加载
├── data/
│   ├── .gitkeep
│   └── test.docx                    # 测试文档（用户自备）
├── models/                          # 本地模型文件（用户下载）
├── .env.example                     # 环境变量模板
├── .gitignore
└── README.md
```

## 依赖

| 包 | 用途 |
|---|---|
| `langchain-openai` | OpenAI/NVIDIA 兼容 LLM 接口 |
| `langchain-huggingface` | HuggingFace 嵌入模型集成 |
| `langchain-chroma` | ChromaDB 向量存储集成 |
| `langchain-text-splitters` | 文本分块工具 |
| `langchain-unstructured` | 多格式文档加载 |
| `sentence-transformers` | 嵌入与交叉编码器推理 |
| `chromadb` | 向量数据库 |
| `python-dotenv` | 环境变量管理 |

## 许可证

MIT
