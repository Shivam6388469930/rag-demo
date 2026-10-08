# config.py
from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()


@dataclass
class RAGConfig:
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
    llm_model: str = os.getenv("LLM_MODEL", "phi3:mini")
    vectorstore_path: str = os.getenv("VECTORSTORE_PATH", "vectorstore")
    max_retries: int = int(os.getenv("MAX_RETRIES", 2))
    data_path: str = "data/sample.json"
    chunk_size: int = 600
    chunk_overlap: int = 80
    retriever_k: int = 3
    retriever_fetch_k: int = 5
    retriever_lambda: float = 0.7
    llm_temperature: float = 0.0


config = RAGConfig()
