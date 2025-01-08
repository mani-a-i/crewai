from fastapi import FastAPI
from settings import get_settings
from schemas.schema import VanillaChatPayload
from models import get_vanillaMISTRAL8x7b
env = get_settings()

def agentic_routers():
    router = FastAPI()

    @router.post("/vanillaMistral",
                 tags=['vanillaEndpoints'],
                 summary="Endpoints to call MISTRAL8x7b LLMs")
    async def call_llm(payload: VanillaChatPayload):
        prompt = payload.prompt

