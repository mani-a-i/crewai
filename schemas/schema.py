from pydantic import BaseModel

class VanillaChatPayload(BaseModel):
    prompt:str

class AgentOutput(BaseModel):
    classifiedSubIntent:str
    CustomerSpecificQuestion:str