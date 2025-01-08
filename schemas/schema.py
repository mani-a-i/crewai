from pydantic import BaseModel

class VanillaChatPayload(BaseModel):
    prompt:str