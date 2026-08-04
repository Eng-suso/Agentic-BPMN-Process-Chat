import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def bool_env(name: str, default: bool = False) -> bool:
    fallback = "true" if default else "false"
    return os.getenv(name, fallback).lower() == "true"


MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")
FINAL_MAX_COMPLETION_TOKENS = int_env("FINAL_MAX_COMPLETION_TOKENS", 3000)
ANSWER_MAX_CONTINUATIONS = int_env("ANSWER_MAX_CONTINUATIONS", 2)

DOCUMENTS_FOLDER = os.getenv("DOCUMENTS_FOLDER", "documents")
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

PDF_VISION_ENABLED = bool_env("PDF_VISION_ENABLED")
PDF_VISION_MAX_PAGES = int_env("PDF_VISION_MAX_PAGES", 25)
PDF_VISION_PAGES = os.getenv("PDF_VISION_PAGES", "").strip()
PDF_VISION_DPI = int_env("PDF_VISION_DPI", 160)
VISION_CACHE_PATH = Path(os.getenv("PDF_VISION_CACHE_PATH", "vision_cache.json"))

REBUILD_INDEX = bool_env("REBUILD_INDEX")
INDEX_PATH = os.getenv(
    "FAISS_INDEX_PATH",
    "faiss_index_documents_vision" if PDF_VISION_ENABLED else "faiss_index_documents",
)

MAX_DOCUMENT_CHARS = int_env("MAX_DOCUMENT_CHARS", 1200)
MAX_WEB_CONTENT_CHARS = int_env("MAX_WEB_CONTENT_CHARS", 800)
MAX_RETRIEVAL_DOCS = int_env("MAX_RETRIEVAL_DOCS", 6)
MAX_SOURCE_RETRIEVAL_DOCS = int_env("MAX_SOURCE_RETRIEVAL_DOCS", 1)
MAX_DIVERSE_SOURCES = int_env("MAX_DIVERSE_SOURCES", 5)

MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "agentic_memory.db")
