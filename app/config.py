from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, EmailStr, SecretStr


class Settings(BaseSettings):
    APP_NAME: str = "BookItRAG API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    DATABASE_URL: str = Field(
        ...,
        description="SQLAlchemy DB Connection string (e.g. postgresql+psycopg2://user:pass@localhost:5432/bookitrag_db)"
    )

    REDIS_URL: str = Field("redis://localhost:6379/0", description="Redis connection URL for chat history")
    REDIS_SESSION_TTL: int = Field(86400, description="Chat session TTL in seconds (default 24h)")

    GOOGLE_API_KEY: SecretStr = Field(..., description="API key for Google Gemini")
    LLM_MODEL_NAME: str = Field("gemini-1.5-flash", description="Gemini model identifier")

    EMBEDDING_MODEL_NAME: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model for embeddings"
    )
    EMBEDDING_DIMENSION: int = Field(384, description="Vector embedding dimension size")

    PINECONE_API_KEY: SecretStr = Field(..., description="Pinecone API key")
    PINECONE_INDEX_NAME: str = Field(..., description="Pinecone index name")

    SMTP_SERVER: str = Field(..., description="SMTP server address")
    SMTP_PORT: int = Field(587, description="SMTP server port")
    SMTP_USERNAME: str = Field(..., description="SMTP login username")
    SMTP_PASSWORD: SecretStr = Field(..., description="SMTP login password")
    SENDER_EMAIL: EmailStr = Field(..., description="Sender email address for confirmation emails")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()