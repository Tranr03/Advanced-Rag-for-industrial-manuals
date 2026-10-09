from __future__ import annotations

import json
import requests
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL, OLLAMA_URL, OLLAMA_MODEL,
    VECTOR_DB_PATH, COLLECTION_NAME,
    GA_WEIGHTS_PATH, GA_RETRIEVAL_POOL,
    DATA_DIR,
)
from ga_reranker import GAReranker

BOW_PATH = DATA_DIR / "bow.json"
BOW_TOP_N = 40   # how many corpus-top terms to inject into the rewrite prompt


class RAGRetriever:
    """
    Full pipeline + BM25-BoW vocabulary injected into the query rewriter.
    Identical to rag_retriever.py except _rewrite_query also receives the
    top-N most discriminative corpus terms from bow.json (built by bm25_bow.py).
    Use this as the experimental arm against rag_retriever.py in ablation.
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

        self.reranker        = GAReranker(weights_path=str(GA_WEIGHTS_PATH))
        self._section_titles = self._load_section_titles()
        self._bow_terms      = self._load_bow_terms()

        print(f"Ready. Collection has {self.collection.count()} documents.")
        if self._bow_terms:
            print(f"BoW vocabulary loaded: {len(self._bow_terms)} top terms from {BOW_PATH.name}")
        else:
            print(f"BoW vocabulary not found at {BOW_PATH} — run bm25_bow.py first.")

    # ─── Public ───────────────────────────────────────────────────────────────

    def retrieve(self, query: str, n_results: int = 10,
                 source_filter: str | None = None,
                 score_threshold: float = 0.55) -> list[dict]:
        """
        Multi-query retrieval: rewrites query into 3 variants, searches each,
        merges by best score, GA-reranks the candidate pool.
        """
        if self.collection.count() == 0:
            return []

        queries = self._rewrite_query(query)
        per_query_k = max(GA_RETRIEVAL_POOL, n_results * 2)

        kwargs: dict = {}
        if source_filter:
            kwargs["where"] = {"source": {"$eq": source_filter}}

        seen: dict[str, dict] = {}
        for q in queries:
            q_emb  = self.embedding_model.encode([q]).tolist()[0]
            safe_n = min(per_query_k, self.collection.count())
            res = self.collection.query(
                query_embeddings=[q_emb],
                n_results=safe_n,
                include=["documents", "metadatas", "distances"],
                **kwargs,
            )
            ids   = res["ids"][0]
            docs  = res["documents"][0]
            metas = res["metadatas"][0]
            dists = res["distances"][0]

            for cid, doc, meta, dist in zip(ids, docs, metas, dists):
                score = 1.0 - dist
                if cid not in seen or score > seen[cid]["score"]:
                    seen[cid] = {
                        "text":     doc,
                        "metadata": meta or {"source": "unknown", "page": "?"},
                        "score":    score,
                    }

        candidates = list(seen.values())
        if not candidates:
            return []

        reranked = self.reranker.rerank(query, candidates)

        return [
            d for d in reranked
            if d["score"] >= score_threshold
            and d.get("metadata", {}).get("type") != "vision_skipped"
        ]

    def generate_answer(self, query: str, docs: list[dict]) -> str:
        """Builds context from docs and calls Ollama to generate an answer."""
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
                timeout=300,
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

    def _load_section_titles(self) -> list[str]:
        """Seed section-title vocabulary from indexed source names."""
        if self.collection.count() == 0:
            return []
        all_meta = self.collection.get(include=["metadatas"])
        return sorted({str(m["source"]) for m in (all_meta["metadatas"] or [])})

    def _load_bow_terms(self) -> list[str]:
        """Load top-N discriminative terms from bow.json (built by bm25_bow.py)."""
        if not BOW_PATH.exists():
            return []
        try:
            with open(BOW_PATH, encoding="utf-8") as f:
                bow = json.load(f)
            return [entry["term"] for entry in bow.get("corpus_top_terms", [])[:BOW_TOP_N]]
        except Exception:
            return []

    def _rewrite_query(self, query: str) -> list[str]:
        """
        Few-shot rewrite toward manual section language.
        Injects BM25-weighted corpus vocabulary so the LLM anchors rewrites
        to actual domain terms found in the manuals.
        Returns [original, ...rewrites].
        """
        titles_block = "\n".join(f"- {t}" for t in self._section_titles)
        terms_block  = ", ".join(self._bow_terms) if self._bow_terms else "(not available)"

        prompt = (
            "You reformulate a user question into search queries that match the "
            "section titles of a machine manual. Use the wording of the titles and "
            "key technical terms below when relevant. "
            "Output 3 queries, one per line, no numbering, no preamble.\n\n"
            f"MANUAL SECTION TITLES:\n{titles_block}\n\n"
            f"KEY TECHNICAL TERMS (most domain-specific words in the corpus):\n{terms_block}\n\n"
            "USER: I hear weird noises from the gearbox, what is it?\n"
            "Excessive, Unusual Noises\n"
            "Gearbox noise meaning\n"
            "Gearbox troubleshooting\n\n"
            "USER: error code E47 keeps showing\n"
            "error code E47\n"
            "fault codes and diagnostics\n"
            "troubleshooting error indications\n\n"
            "USER: How can I power the machine and zero the axes?\n"
            "Startup and Initialization\n"
            "Axes configuration\n"
            "Axes set to zero\n\n"
            "USER: Why does my machine vibrate when I run it at high speed?\n"
            "Abnormal vibration at high speed\n"
            "Vibration troubleshooting\n"
            "Operating speed limits\n\n"
            f"USER: {query}\n"
        )

        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model":   OLLAMA_MODEL,
                    "prompt":  prompt,
                    "stream":  False,
                    "think":   False,
                    "options": {"num_ctx": 2048, "temperature": 0.3, "num_gpu": 99},
                },
                timeout=30,
            )
            resp.raise_for_status()
            lines = [l.strip() for l in resp.json().get("response", "").splitlines() if l.strip()]
            return [query] + lines[:3]
        except Exception:
            return [query]

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
