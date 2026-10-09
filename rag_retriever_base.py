from __future__ import annotations

import requests
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL, OLLAMA_URL, OLLAMA_MODEL,
    VECTOR_DB_PATH, COLLECTION_NAME,
    GA_RETRIEVAL_POOL,
)


class RAGRetriever:
    """
    Baseline retriever — no GA reranking, no query rewriting, no few-shot prompting.
    Single cosine-similarity search, instruction-only system prompt.
    Use this as the control when evaluating improvements.
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

        print(f"Ready. Collection has {self.collection.count()} documents.")

    # ─── Public ───────────────────────────────────────────────────────────────

    def retrieve(self, query: str, n_results: int = 10,
                 source_filter: str | None = None,
                 score_threshold: float = 0.55) -> list[dict]:
        """Single vector search, cosine order, no reranking."""
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        safe_n = min(max(GA_RETRIEVAL_POOL, n_results), self.collection.count())

        kwargs: dict = {}
        if source_filter:
            kwargs["where"] = {"source": {"$eq": source_filter}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=safe_n,
            include=["documents", "metadatas", "distances"],
            **kwargs,
        )

        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        docs = [
            {
                "text":     documents[0][i],
                "metadata": metadatas[0][i] or {"source": "unknown", "page": "?"},
                "score":    1.0 - distances[0][i],
            }
            for i in range(len(documents[0]))
        ]

        return [
            d for d in docs[:n_results]
            if d["score"] >= score_threshold
            and d.get("metadata", {}).get("type") != "vision_skipped"
        ]

    def generate_answer(self, query: str, docs: list[dict]) -> str:
        """Instruction-only prompt, no few-shot examples."""
        if not docs:
            return "No relevant information found in the indexed manuals."

        context = self._build_context(docs)

        system = (
            "You are a technical assistant for industrial equipment manuals. "
            "Answer questions using ONLY the provided context.\n\n"
            "RULES:\n"
            "1. Be concise. Answer the question directly, then stop. Do not elaborate unless explicitly asked.\n"
            "2. Each answer is from one source only.\n"
            "3. No extra commentary. Do not add notes, caveats, cross-references, or \"additional context\" beyond what the question asks for.\n"
            "4. Answer using a single fully developed sentence.\n"
            "5. Match the scope of the question. If the question asks for a single value, give that value. If it asks for a list, give the list. Do not expand the scope.\n\n"
            "FORMAT:\n"
            "- Direct answer first\n"
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
        all_meta  = self.collection.get(include=["metadatas"])
        metadatas = all_meta["metadatas"] or []
        return sorted({str(m["source"]) for m in metadatas})

    def count(self) -> int:
        return self.collection.count()

    # ─── Private ──────────────────────────────────────────────────────────────

    def _build_context(self, docs: list[dict]) -> str:
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
            source   = doc["metadata"].get("source", "unknown")
            page     = doc["metadata"].get("page", "?")
            doc_type = doc["metadata"].get("type", "text")
            label    = "Visual Element" if doc_type == "image_description" else "Text"
            parts.append(f"[{label} — {source}, p.{page}]\n{doc['text'][:1200].strip()}")

        return "\n\n---\n\n".join(parts)
