from langchain_huggingface import HuggingFaceEndpoint
from settings import get_settings

env = get_settings()


def get_vanillaMISTRAL() -> HuggingFaceEndpoint:
    llm = HuggingFaceEndpoint(
    repo_id= env.MISTRAL7b_REPO_ID,       
    huggingfacehub_api_token=env.HUGGINGFACEHUB_API_TOKEN
    )
    return llm

def get_vanillaZephyr_7b_beta() -> HuggingFaceEndpoint:
    llm = HuggingFaceEndpoint(
        repo_id = env.ZEPHYR_REPO_ID,
        huggingfacehub_api_token = env.HUGGINGFACEHUB_API_TOKEN
    )

    return llm

def get_vanillaLlama3_8B_Instruct() -> HuggingFaceEndpoint:
    llm = HuggingFaceEndpoint(
                    repo_id=env.Llama3_8B_Instruct,       
                    huggingfacehub_api_token=env.HUGGINGFACEHUB_API_TOKEN,
                    task = "text-generation"
                    )  

    return llm
