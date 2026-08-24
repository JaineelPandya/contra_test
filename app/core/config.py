"""Application configuration settings."""
from functools import lru_cache
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """RegGraph global configuration settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    APP_NAME: str = "RegGraph"
    APP_ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8080
    HOST: str = "0.0.0.0"

    # Uploads
    UPLOAD_DIR: Path = Path("uploads")

    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "reggraph_password"
    NEO4J_DATABASE: str = "neo4j"

    # LLM Settings
    LLM_PROVIDER: Literal["ollama", "openai", "gemini", "groq", "mock"] = "groq"

    # Groq Settings
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-20b"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_TEMPERATURE: float = 0.0
    GROQ_TPM_LIMIT: int = 8000
    GROQ_TPM_SAFETY_MARGIN: float = 0.90
    GROQ_MAX_REQUEST_INPUT_TOKENS: int = 4000
    GROQ_RESERVED_OUTPUT_TOKENS: int = 600
    GROQ_MAX_RATE_LIMIT_RETRIES: int = 3
    GROQ_MAX_RETRY_WAIT_SECONDS: float = 60.0
    GROQ_RATE_LIMIT_SAFETY_BUFFER_SECONDS: float = 2.0
    GROQ_CIRCUIT_BREAKER_CONSECUTIVE_FAILURES: int = 3
    GROQ_CIRCUIT_BREAKER_COOLDOWN_SECONDS: float = 60.0

    # Gemini Settings
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-flash-latest"
    GEMINI_TEMPERATURE: float = 0.0

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:latest"
    OLLAMA_TEMPERATURE: float = 0.0

    # OpenAI-Compatible API
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.0

    # Chunking & Processing
    MAX_CHUNK_TOKENS: int = 800
    CHUNK_OVERLAP_TOKENS: int = 100
    MAX_CONCURRENT_CHUNKS: int = 4

    # LLM Batching & Concurrency Optimization
    FACT_EXTRACTION_BATCH_SIZE: int = 3
    RELATIONSHIP_TOP_K: int = 5
    RELATIONSHIP_BATCH_SIZE: int = 10
    GEMINI_MAX_CONCURRENT_REQUESTS: int = 2
    GEMINI_REQUESTS_PER_MINUTE: int = 30
    GEMINI_MAX_RETRIES: int = 5
    GEMINI_INITIAL_RETRY_DELAY: float = 1.0
    GEMINI_MAX_RETRY_DELAY: float = 60.0

    @property
    def upload_path(self) -> Path:
        """Ensure upload directory exists and return resolved path."""
        path = self.UPLOAD_DIR
        if not path.is_absolute():
            path = Path.cwd() / path
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
