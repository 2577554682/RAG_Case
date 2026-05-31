from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.document_compressors.cross_encoder import BaseCrossEncoder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from sentence_transformers.cross_encoder import CrossEncoder
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Tuple

from env_utils import EMBED_MODEL_PATH, RERANKER_MODEL_PATH, CHROMA_DB_PATH
from my_llm import chatgpt

# 加载embedding模型（将文本转换成向量的模型）
embed_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_PATH,
    model_kwargs={'device': 'cuda'}
)

# 加载向量数据库（存储文档向量的地方）
vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embed_model
)

# 加载本地重排模型（用于对初步检索结果进行重新排序）
cross_encoder = CrossEncoder(RERANKER_MODEL_PATH)

# 定义包装器类：将sentence-transformers的CrossEncoder适配成LangChain需要的格式
class CrossEncoderWrapper(BaseCrossEncoder):
    """包装器：让重排模型符合LangChain的接口规范"""

    def __init__(self, model):
        """初始化时保存重排模型"""
        self.model = model

    def score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        计算查询-文档对的相关性分数

        Args:
            pairs: 列表，每个元素是(查询文本, 文档文本)的元组

        Returns:
            List[float]: 每个文档的相关性分数列表，分数越高表示越相关
        """
        scores = self.model.predict(pairs)
        return scores.tolist() if hasattr(scores, 'tolist') else list(scores)

# 将原始的重排模型包装成LangChain可用的格式
wrapped_encoder = CrossEncoderWrapper(cross_encoder)

# 创建重排压缩器（负责对文档进行筛选和重排序）
compressor = CrossEncoderReranker(
    model=wrapped_encoder,
    top_n=3
)

# 创建最终的重排检索器（结合了初步检索和重排两个步骤）
rerank_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )
)

# 使用重排检索器
query = "这个系统有什么特点？"
print(f"查询: {query}\n")
relevant_docs = rerank_retriever.invoke(query)

# 将文档列表转换为字符串
docs_list =[doc.page_content for doc in relevant_docs]
docs_list = "\n".join(docs_list)
# 创建提示词模板
template = PromptTemplate.from_template(
    template="你是一个智能助手，根据用户的问题,和下列片段生成准确的答案:\n用户的问题:{query}\n片段:{docs_list}\n请基于上述内容生成答案,不要编造"
)

chain = template | chatgpt | StrOutputParser()
for chunk in chain.stream({"query": query, "docs_list": docs_list}):
    print(chunk,end='',flush=True)