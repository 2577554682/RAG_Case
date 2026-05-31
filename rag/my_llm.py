from langchain_openai import ChatOpenAI


from env_utils import NVIDIA_BASE_URL, NVIDIA_API_KEY

chatgpt = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=NVIDIA_API_KEY,
    base_url=NVIDIA_BASE_URL,
    temperature=1,
)