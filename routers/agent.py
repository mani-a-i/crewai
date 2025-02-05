from fastapi import APIRouter
from settings import get_settings
from schemas.schema import VanillaChatPayload
from models import llm_registry as global_llm
from helper import mistral_prompt_template, zephyr_prompt_template,llama_prompt_template
from google.generativeai.types.generation_types import GenerateContentResponse
env = get_settings()

def agentic_routers() -> APIRouter:
    router = APIRouter()

    @router.post("/vanillaMistral",
                 tags=['vanillaEndpoints'],
                 summary="Endpoints to call MISTRAL8x7b LLM")
    async def call_llm(payload: VanillaChatPayload) -> dict[str,str]:
        prompt:str = mistral_prompt_template(payload.prompt)     

        output:str = global_llm.Mistral8x7b.invoke(prompt)

        return {
            "llm_output":output
        }    
    

    @router.post("/vanillaZephyr",
                 tags=['vanillaEndpoints'],
                 summary="Endpoints to call Zephyr 7b betaLLM")
    async def call_llm(payload: VanillaChatPayload) -> dict[str,str]:
        prompt:str = zephyr_prompt_template(payload.prompt)   
        output:str = global_llm.Zephyr_7b_beta.invoke(prompt)

        return {
            "llm_output":output
        }
    
    @router.post("/vanillallama3b",
                 tags=['vanillaEndpoints'],
                 summary="Endpoints to call vanilla llama 3b Instruct")
    async def call_llm(payload: VanillaChatPayload) -> dict[str,str]:
        prompt:str = llama_prompt_template(payload.prompt)
        output:str = global_llm.llama3_8B_Instruct.invoke(prompt)

        return {
            "llm_output":output
        }
    
    
    @router.post("/geminiFlash",
                 tags=['vanillaEndpoints'],
                 summary="Endpoints to call gemini 1.5 flash")
    async def call_llm(payload: VanillaChatPayload) -> dict[str,str]:        
        output:GenerateContentResponse = global_llm.gemini_flash.generate_content(payload.prompt)

        return {
            "llm_output":output.text
        }
    
    return router

router = agentic_routers()


