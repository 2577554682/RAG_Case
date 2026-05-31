import os
from dotenv import load_dotenv
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL")

# 本地模型路径配置
EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH", "./models/bge-large-zh-v1.5")
RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", "./models/bge-reranker-large")

# 数据路径配置
DATA_DIR = os.getenv("DATA_DIR", "./data")
TEST_DOC_PATH = os.getenv("TEST_DOC_PATH", os.path.join(DATA_DIR, "test.docx"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.join(DATA_DIR, "chroma_db"))