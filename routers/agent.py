from fastapi import APIRouter
from settings import get_settings
from schemas.schema import VanillaChatPayload
from models import llm_registry as global_llm
from helper import mistral_prompt_template, zephyr_prompt_template,llama_prompt_template
env = get_settings()

def agentic_routers():
    router = APIRouter()

    @router.post("/vanillaMistral",
                 tags=['vanillaEndpoints'],
                 summary="Endpoints to call MISTRAL8x7b LLM")
    async def call_llm(payload: VanillaChatPayload):
        prompt = mistral_prompt_template(payload.prompt)       

        output = global_llm.Mistral8x7b.invoke(prompt)[len(prompt):]

        return {
            "llm_output":output
        }    
    

    @router.post("/vanillaZephyr",
                 tags=['vanillaEndpoints'],
                 summary="Endpoints to call Zephyr 7b betaLLM")
    async def call_llm(payload: VanillaChatPayload):
        prompt = zephyr_prompt_template(payload.prompt)   
        output = global_llm.Zephyr_7b_beta.invoke(prompt)[len(prompt):]

        return {
            "llm_output":output
        }
    
    @router.post("/vanillallama3b",
                 tags=['vanillaEndpoints'],
                 summary="Endpoints to call vanilla llama 3b Instruct")
    async def call_llm(payload: VanillaChatPayload):
        prompt = llama_prompt_template(payload.prompt)
        output = global_llm.llama3_8B_Instruct.invoke(prompt)[len(prompt):]

        return {
            "llm_output":output
        }
    
    return router

router = agentic_routers()


