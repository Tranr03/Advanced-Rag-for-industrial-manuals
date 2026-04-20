from __future__ import annotations

import requests
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL, OLLAMA_URL, OLLAMA_MODEL,
    VECTOR_DB_PATH, COLLECTION_NAME
)


class RAGRetriever:
    """
    Handles retrieval and answer generation:
      1. Embed query + vector search in ChromaDB
      2. Source diversity filtering
      3. Answer generation via Ollama
    """

    def __init__(self):
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
        try:
            self.collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not found. "
                "Run RAGPreprocessor first to index your PDFs."
            )

        print(f"Ready. Collection has {self.collection.count()} documents.")

    # ─── Public ───────────────────────────────────────────────────────────────

    def retrieve(self, query: str, n_results: int = 10,
                 source_filter: str | None = None,
                 score_threshold: float = 0.55) -> list[dict]:
        """
        Vector search. Returns list of {text, metadata, score}.
        source_filter: if set, restricts search to a single manual.
        score_threshold: drops chunks below this cosine similarity.
        """
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        safe_n = min(n_results, self.collection.count())

        kwargs: dict = {}
        if source_filter:
            kwargs["where"] = {"source": {"$eq": source_filter}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=safe_n,
            include=["documents", "metadatas", "distances"],
            **kwargs
        )

        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        docs = [
            {
                "text":     documents[0][i],
                "metadata": metadatas[0][i] or {"source": "unknown", "page": "?"},
                "score":    1 - distances[0][i]
            }
            for i in range(len(documents[0]))
        ]

        # Score threshold — αφαίρεση χαμηλής εμπιστοσύνης chunks
        return [d for d in docs if d["score"] >= score_threshold]

    def generate_answer(self, query: str, docs: list[dict]) -> str:
        """Builds context from docs and calls Ollama to generate an answer."""
        if not docs:
            return "No relevant information found in the indexed manuals."

        context = self._build_context(docs)
        system = (
            "You are a technical assistant specialized in industrial machinery manuals. "
            "Answer using only the provided context. Be precise and concise. "
            "If the context contains no relevant information, say: "
            "'I could not find this information in the available manuals.'"
        )
        prompt = (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            "ANSWER:"
        )

        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model":   OLLAMA_MODEL,
                    "system":  system,
                    "prompt":  prompt,
                    "stream":  False,
                    "think":   True,
                    "options": {"num_ctx": 4096, "temperature": 0.1, "num_gpu": 99}
                },
                timeout=200
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            return f"Error generating answer: {e}"

    def query(self, question: str, n_results: int = 5,
              source_filter: str | None = None,
              score_threshold: float = 0.55) -> tuple[str, list[dict]]:
        """
        Convenience method: retrieve + generate in one call.
        Returns (answer, docs).
        """
        docs = self.retrieve(question, n_results,
                             source_filter, score_threshold)
        answer = self.generate_answer(question, docs)
        return answer, docs

    def list_sources(self) -> list[str]:
        """Returns sorted list of all indexed source names."""
        all_meta = self.collection.get(include=["metadatas"])
        metadatas = all_meta["metadatas"] or []
        return sorted({str(m["source"]) for m in metadatas})

    def count(self) -> int:
        return self.collection.count()

    # ─── Private ──────────────────────────────────────────────────────────────

    def _build_context(self, docs: list[dict]) -> str:
        # Source diversity: max 2 docs per source
        seen: dict[str, int] = {}
        diverse = []
        for doc in docs:
            if not doc.get("metadata"):
                continue
            src = str(doc["metadata"]["source"])
            count = seen.get(src, 0)
            if count < 2:
                diverse.append(doc)
                seen[src] = count + 1

        parts = []
        for doc in diverse:
            source = doc["metadata"]["source"]
            page = doc["metadata"].get("page", "?")
            doc_type = doc["metadata"].get("type", "text")
            label = "Visual Element" if doc_type == "image_description" else "Text"
            text = doc["text"][:1200]  # ~300 tokens max per chunk
            parts.append(f"[{label} — {source}, p.{page}]\n{text}")

        return "\n\n---\n\n".join(parts)
