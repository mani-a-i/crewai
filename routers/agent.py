from fastapi import APIRouter
from settings import get_settings
from schemas.schema import VanillaChatPayload
from models import llm_registry as global_llm
from helper import mistral_prompt_template
env = get_settings()

def agentic_routers():
    router = APIRouter()

    @router.post("/vanillaMistral",
                 tags=['vanillaEndpoints'],
                 summary="Endpoints to call MISTRAL8x7b LLMs")
    async def call_llm(payload: VanillaChatPayload):
        prompt = mistral_prompt_template(payload.prompt)       

        output = global_llm.Mistral8x7b.invoke(prompt)[len(prompt):]

        return {
            "llm_output":output
        }
    
    return router

router = agentic_routers()


