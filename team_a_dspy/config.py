from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_base_url: str
    llm_model_name: str
    llm_api_key: str = "not-required"
    
    # Elasticsearch
    es_host: str
    es_username: str
    es_password: str
    es_index: str = "gkg"
    es_verify_ssl: bool = False
    es_request_timeout_seconds: int = 60  # <-- ADD THIS LINE
    
    # Chroma
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    chroma_collection: str = "gkg_mapping"
    schema_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Sandbox Elasticsearch
    sandbox_es_host: str = "http://sandbox_es:9200"

settings = Settings()