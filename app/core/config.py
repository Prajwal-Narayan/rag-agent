import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import SecretStr

load_dotenv()

class Settings(BaseSettings):
    # App Config
    PROJECT_NAME: str = "Advanced RAG Agent"
    
    # LLM Keys
    OPENAI_API_KEY: SecretStr
    COHERE_API_KEY: SecretStr | None = None
    
    # LangSmith Config
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: SecretStr | None = None
    LANGCHAIN_PROJECT: str = "rag-ecosystem-demo"
    
    # Vector DB Config
    CHROMA_PERSIST_DIRECTORY: str = "chroma_db_data"
    COLLECTION_NAME: str = "rag_collection"
    
    # 👇 NEW: Database Config (Reads from .env)
    POSTGRES_USER: str = "rag_user"
    POSTGRES_PASSWORD: str = "rag_password"
    POSTGRES_DB: str = "rag_db"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"
    
    # Computed property for the Connection String
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # Model Config
    LLM_MODEL: str = "gpt-3.5-turbo" 
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings() #type: ignore