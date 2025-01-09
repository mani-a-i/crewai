from models import llm_registry as global_llm
from models.get_models import get_vanillaMISTRAL8x7b
from fastapi import FastAPI
from routers.agent import router

def load_models():
    print("LOADING MODELS........")
    global_llm.Mistral8x7b = get_vanillaMISTRAL8x7b()

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