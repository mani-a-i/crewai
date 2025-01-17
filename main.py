from models import llm_registry as global_llm
from langchain_huggingface import HuggingFaceEndpoint
from typing import Type
from models.get_models import get_vanillaMISTRAL,get_vanillaZephyr_7b_beta, get_vanillaLlama3_8B_Instruct
from fastapi import FastAPI
from routers.agent import router
import warnings
warnings.filterwarnings('ignore')

def load_models()->None:
    print("LOADING MODELS........")
    global_llm.Mistral8x7b: HuggingFaceEndpoint = get_vanillaMISTRAL() #type: ignore
    global_llm.Zephyr_7b_beta: HuggingFaceEndpoint = get_vanillaZephyr_7b_beta() #type: ignore
    global_llm.llama3_8B_Instruct: HuggingFaceEndpoint = get_vanillaLlama3_8B_Instruct() #type: ignore

load_models()

def create_app()->FastAPI:

    app = FastAPI(
        title="CREWAI POC",
        openapi_tags= [{
            "name":"vanillaEndpoints"
        }]
    )

    app.include_router(router)
    return app

app:FastAPI = create_app()