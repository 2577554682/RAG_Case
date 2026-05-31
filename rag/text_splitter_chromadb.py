from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_unstructured import UnstructuredLoader
from langchain_chroma import Chroma

from env_utils import TEST_DOC_PATH, EMBED_MODEL_PATH, CHROMA_DB_PATH

loader = UnstructuredLoader(
    file_path=TEST_DOC_PATH,
    mode="elements"
)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    separators=[ "\n\n","\n", "。", "？", "！", "......", "，", ""],
    chunk_size=500,
    chunk_overlap=20,
    length_function=len,
    add_start_index=True
)

chunks = text_splitter.split_documents(docs)

embed_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_PATH,
    model_kwargs={'device': 'cuda'}
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embed_model,
    persist_directory=CHROMA_DB_PATH
)


# 查看存储了多少个文档块
print(f"成功存储 {vectorstore._collection.count()} 个文档块")