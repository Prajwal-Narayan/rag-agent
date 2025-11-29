import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import SecretStr # <--- Added Import

load_dotenv()

class Settings(BaseSettings):
    # App Config
    PROJECT_NAME: str = "Advanced RAG Agent"
    
    # LLM Keys (Now Secure Types)
    OPENAI_API_KEY: SecretStr  # <--- Changed from str to SecretStr
    COHERE_API_KEY: SecretStr | None = None
    
    # LangSmith Config
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: SecretStr | None = None # <--- Also SecretStr
    LANGCHAIN_PROJECT: str = "rag-ecosystem-demo"
    
    # Vector DB Config
    CHROMA_PERSIST_DIRECTORY: str = "chroma_db_data"
    COLLECTION_NAME: str = "rag_collection"
    
    # Model Config
    LLM_MODEL: str = "gpt-3.5-turbo" 
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()