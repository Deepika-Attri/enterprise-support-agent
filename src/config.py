"""Central settings and the LLM factory."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # also loads LangSmith vars if you add them to .env

ROOT = Path(__file__).resolve().parent.parent


class Settings:
    """Every tunable value in one place, overridable through .env."""

    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" | "gemini"
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    SAMPLE_DIR = ROOT / "data" / "sample_documents"  # demo data shipped with the repo
    DOCS_DIR = ROOT / "data" / "documents"  # files uploaded by users
    CHROMA_DIR = ROOT / "data" / "chroma_db"
    DB_PATH = ROOT / "data" / "mock_database.db"
    COLLECTION = "enterprise_kb"

    CHUNK_SIZE = 800  # tuned in Phase 2 experiments
    CHUNK_OVERLAP = 100
    TOP_K = 4
    MAX_RETRIES = 2  # query rewrites before escalating


settings = Settings()


def get_llm(temperature: float = 0):
    """Return the configured chat model (free: Ollama local or Gemini free tier)."""
    if settings.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=settings.OLLAMA_MODEL, temperature=temperature)

    if settings.LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=temperature,
            google_api_key=settings.GOOGLE_API_KEY,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER}")
