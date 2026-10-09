from __future__ import annotations

import requests
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL, OLLAMA_URL, OLLAMA_MODEL,
    VECTOR_DB_PATH, COLLECTION_NAME,
    GA_WEIGHTS_PATH, GA_RETRIEVAL_POOL,
)
from ga_reranker import GAReranker


class RAGRetriever:
    """
    Handles retrieval and answer generation:
      1. Single vector search in ChromaDB
      2. GA reranking of the candidate pool
      3. Answer generation via Ollama
    """

    def __init__(self, embedding_model=None, collection=None):
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            print("Loading embedding model...")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        if collection is not None:
            self.collection = collection
        else:
            client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
            try:
                self.collection = client.get_collection(COLLECTION_NAME)
            except Exception:
                self.collection = client.create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )

        self.reranker = GAReranker(weights_path=str(GA_WEIGHTS_PATH))

        print(f"Ready. Collection has {self.collection.count()} documents.")

    # ─── Public ───────────────────────────────────────────────────────────────

    def retrieve(self, query: str, n_results: int = 10,
                 source_filter: str | None = None,
                 score_threshold: float = 0.55) -> list[dict]:
        """Single vector search, GA-reranks the candidate pool."""
        if self.collection.count() == 0:
            return []

        per_query_k = max(GA_RETRIEVAL_POOL, n_results * 2)
        safe_n = min(per_query_k, self.collection.count())

        kwargs: dict = {}
        if source_filter:
            kwargs["where"] = {"source": {"$eq": source_filter}}

        q_emb = self.embedding_model.encode([query]).tolist()[0]
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=safe_n,
            include=["documents", "metadatas", "distances"],
            **kwargs,
        )
        docs  = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        candidates = [
            {
                "text":     doc,
                "metadata": meta or {"source": "unknown", "page": "?"},
                "score":    1.0 - dist,
            }
            for doc, meta, dist in zip(docs, metas, dists)
            if (meta or {}).get("type") != "vision_skipped"
        ]
        if not candidates:
            return []

        reranked = self.reranker.rerank(query, candidates)
        if not reranked:
            return sorted(candidates, key=lambda d: d["score"], reverse=True)[:3]

        return reranked

    def generate_answer(self, query: str, docs: list[dict]) -> str:
        """Builds context from docs and calls Ollama to generate an answer."""
        if not docs:
            return "No relevant information found in the indexed manuals."

        context = self._build_context(docs)

        system = (
            "You are a chatbot providing answers to user queries. You will be given one or more context documents, "
            "and a question. Use the information in the documents to answer the question. "
            "If the documents do not provide enough information for you to answer the question, then say: \""
            "The documents are missing some of the information required to answer the question.\" "
            "Do not quote any external knowledge that is not in the documents. Do not make up an answer."
        )

        prompt = (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n"
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
                    "think":   False,
                    "options": {"num_ctx": 8192, "temperature": 0.1, "num_gpu": 99},
                },
                timeout=200,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            return f"Error generating answer: {e}"

    def query(self, question: str, n_results: int = 5,
              source_filter: str | None = None,
              score_threshold: float = 0.55) -> tuple[str, list[dict]]:
        """Convenience method: retrieve + generate in one call. Returns (answer, docs)."""
        docs   = self.retrieve(question, n_results, source_filter, score_threshold)
        answer = self.generate_answer(question, docs)
        return answer, docs

    def list_sources(self) -> list[str]:
        """Returns sorted list of all indexed source names."""
        all_meta  = self.collection.get(include=["metadatas"])
        metadatas = all_meta["metadatas"] or []
        return sorted({str(m["source"]) for m in metadatas})

    def count(self) -> int:
        return self.collection.count()

    # ─── Private ──────────────────────────────────────────────────────────────

    def _build_context(self, docs: list[dict]) -> str:
        """Source diversity (max 2 per source) + formatted context block."""
        seen: dict[str, int] = {}
        diverse = []
        for doc in docs:
            if not doc.get("metadata"):
                continue
            src   = str(doc["metadata"]["source"])
            count = seen.get(src, 0)
            if count < 2:
                diverse.append(doc)
                seen[src] = count + 1

        parts = []
        for doc in diverse:
            meta   = doc["metadata"]
            source = meta.get("source", "unknown source")
            page   = meta.get("page")
            header = f"[Source: {source}" + (f" — p.{page}]" if page is not None else "]")
            parts.append(f"{header}\n{doc['text'][:1200].strip()}")

        return "\n\n---\n\n".join(parts)
