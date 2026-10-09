import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
MANUALS_FOLDER = DATA_DIR / "manuals"
MARKDOWN_FOLDER = DATA_DIR / "markdown_temp"
CHUNKS_FOLDER = DATA_DIR / "chunks_temp"
IMAGES_FOLDER = DATA_DIR / "images_temp"
VECTOR_DB_PATH = DATA_DIR / "vector_db"

for folder in [MANUALS_FOLDER, MARKDOWN_FOLDER, CHUNKS_FOLDER, IMAGES_FOLDER, VECTOR_DB_PATH]:
    folder.mkdir(parents=True, exist_ok=True)

# ─── Models ───────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3.5:27b"
VISION_MODEL = "qwen3-vl:8b"

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80

# ─── ChromaDB ─────────────────────────────────────────────────────────────────
COLLECTION_NAME = "rag_manuals"

# ─── GA Reranker ──────────────────────────────────────────────────────────────
GA_WEIGHTS_PATH = DATA_DIR / "ga_weights.json"
GA_RETRIEVAL_POOL = 50  # candidates fetched from ChromaDB for reranking (must match training)

# ─── Evaluation ───────────────────────────────────────────────────────────────
EVAL_QUESTIONS_PATH   = DATA_DIR / "evaluation_questions.xls"
EVAL_RESULTS_PATH     = DATA_DIR / "evaluation_results.xls"
EVAL_INCREMENTAL_PATH = DATA_DIR / "eval_incremental.json"
OPENAI_API_KEY        = os.environ.get("OPENAI_API_KEY", "")
