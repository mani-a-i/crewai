from functools import lru_cache
from pydantic_settings import (BaseSettings, SettingsConfigDict)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8'
    )

    MISTRAL_REPO_ID: str|None = None
    MISTRAL8x7b_REPO_ID: str|None = None
    ZEPHYR_REPO_ID: str|None = None
    Llama3_8B_Instruct: str|None = None
    HF_API_TOKEN: str|None = None


@lru_cache
def get_settings():
    return Settings()

