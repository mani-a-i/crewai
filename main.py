from models import llm_registry as global_llm
from models.get_models import get_vanillaMISTRAL,get_vanillaZephyr_7b_beta, get_vanillaLlama3_8B_Instruct
from fastapi import FastAPI
from routers.agent import router
import warnings
warnings.filterwarnings('ignore')

def load_models():
    print("LOADING MODELS........")
    global_llm.Mistral8x7b = get_vanillaMISTRAL()
    global_llm.Zephyr_7b_beta = get_vanillaZephyr_7b_beta()
    global_llm.llama3_8B_Instruct = get_vanillaLlama3_8B_Instruct()

load_models()

def create_app():

    app = FastAPI(
        title="CREWAI POC",
        openapi_tags= [{
            "name":"vanillaEndpoints"
        }]
    )

    app.include_router(router)
    return app

app = create_app()