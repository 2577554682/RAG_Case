# RAG Case

基于 LangChain 和 Chroma 的检索增强生成（RAG）案例。

## 功能

- 文档加载与分块处理
- 中文向量嵌入（BGE-large-zh-v1.5）
- 向量相似度检索
- 重排模型优化（bge-reranker-large）
- LLM 智能问答

## 环境配置

1. 复制配置模板：
```bash
cp .env.example .env
```

2. 编辑 `.env`，填入你的配置：
- `NVIDIA_API_KEY`：NVIDIA API 密钥
- `EMBED_MODEL_PATH`：BGE 向量模型路径
- `RERANKER_MODEL_PATH`：重排模型路径

3. 下载模型：
```bash
# BGE 向量模型
git lfs install
git clone https://huggingface.co/BAAI/bge-large-zh-v1.5 ./models/bge-large-zh-v1.5

# BGE 重排模型
git clone https://huggingface.co/BAAI/bge-reranker-large ./models/bge-reranker-large
```

## 使用

### 1. 文档入库

```bash
python -m rag.text_splitter_chromadb
```

将 `data/test.docx` 文档切分后存入向量数据库。

### 2. 检索问答

```bash
python -m rag.retriever
```

根据用户问题从向量库检索相关片段，生成答案。

## 项目结构

```
RAG_case/
├── .env              # 本地配置（不上传）
├── .env.example      # 配置模板
├── env_utils.py      # 配置工具
├── rag/
│   ├── my_llm.py           # LLM 配置
│   ├── text_splitter_chromadb.py  # 文档入库
│   └── retriever.py        # 检索问答
├── models/           # 本地模型（不上传）
├── data/
│   ├── test.docx    # 测试文档
│   └── chroma_db/   # 向量数据库（不上传）
└── README.md
```

## 依赖

```
langchain-openai
langchain-huggingface
langchain-chroma
langchain-text-splitters
langchain-unstructured
sentence-transformers
chromadb
python-dotenv
```