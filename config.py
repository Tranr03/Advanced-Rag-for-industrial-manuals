from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
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
OLLAMA_MODEL = "qwen3:8b"
VISION_MODEL = "qwen3-vl:8b"

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80

# ─── ChromaDB ─────────────────────────────────────────────────────────────────
COLLECTION_NAME = "rag_manuals"
