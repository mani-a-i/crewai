from langchain_huggingface import HuggingFaceEndpoint
import google.generativeai as genai
from google.generativeai.generative_models import GenerativeModel
from settings import get_settings
from crewai import LLM

env = get_settings()

def get_vanillaMISTRAL() -> HuggingFaceEndpoint:
    try:
        llm = HuggingFaceEndpoint(
        repo_id= env.MISTRAL7b_REPO_ID,       
        huggingfacehub_api_token=env.HUGGINGFACEHUB_API_TOKEN
        )
        return llm
    except Exception as e:
        print(f"Mistral not working....{e}")

def get_vanillaZephyr_7b_beta() -> HuggingFaceEndpoint:
    try:
        llm = HuggingFaceEndpoint(
            repo_id = env.ZEPHYR_REPO_ID,
            huggingfacehub_api_token = env.HUGGINGFACEHUB_API_TOKEN
        )

        return llm
    except Exception as e:
        print(f"Zephyr not working....{e}")

def get_vanillaLlama3_8B_Instruct() -> HuggingFaceEndpoint:
    try:
        llm = HuggingFaceEndpoint(
                        repo_id=env.Llama3_8B_Instruct,       
                        huggingfacehub_api_token=env.HUGGINGFACEHUB_API_TOKEN,
                        task = "text-generation"
                        )  

        return llm
    except Exception as e:
        print(f"Llama3 not working....{e}")

def get_gemini_flash()->GenerativeModel:
    try:
        if env.GEMINI_API_KEY:        
            llm =  LLM(model="gemini/gemini-1.5-flash",api_key=env.GEMINI_API_KEY)
            return llm
        else:
            raise Exception("No api key found for Gemini,")
        
    except Exception as e:
        print(f"gemini not working...{e}")

