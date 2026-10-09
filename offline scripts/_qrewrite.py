def _rewrite_query(self, query: str) -> list[str]:
    """Few-shot rewrite προς τη γλώσσα των τίτλων. Επιστρέφει [original, ...rewrites]."""
    titles_block = "\n".join(f"- {t}" for t in self._section_titles)

    
    prompt = (
        "You reformulate a user question into search queries that match the "
        "section titles of a machine manual. Use the wording of the titles below "
        "when relevant. Output 3 queries, one per line, no numbering, no preamble.\n\n" \
        "MANUAL SECTION TITLES:\n{titles_block}\n\n" \
        "USER: I hear weird noises form the gearbox, what is it?\n"
        "Excessive, Unusual Noises\n" 
        "Gearbox noise meaning\n" 
        "Gearbox troubleshooting\n"
        "USER: error code E47 keeps showing\n" \
        "error code E47\n"
        "fault codes and diagnostics\n"
        "troubleshooting error indications\n\n"
        "USER: How can I power the machine and zero the axes?\n"
        "Sartup and Initialization\n"
        "Axes configuration\n"
        "Axes set to zero\n\n"
        "USER: Why does my machine vibrate when I run it at high speed?\n"
        "Abnormal vibration at high speed\n"
        "Vibration troubleshooting\n"
        "Operating speed limits\n\n"
        f"USER: {query}\n"
    )
    # ────────────────────────────────────────────────────────────────

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":   OLLAMA_MODEL,
                "prompt":  prompt,
                "stream":  False,
                "think":   False,        # δομικό task, χωρίς reasoning
                "options": {"num_ctx": 2048, "temperature": 0.3, "num_gpu": 99},
            },
            timeout=30,
        )
        resp.raise_for_status()
        lines = [l.strip() for l in resp.json().get("response", "").splitlines() if l.strip()]
        return [query] + lines[:3]       # το ORIGINAL επιβιώνει πάντα
    except Exception:
        return [query]                   # graceful fallback


def retrieve(self, query: str, top_k: int = 5, per_query_k: int | None = None) -> list[dict]:
    """Multi-query search -> merge/dedup -> ο υπάρχων GA reranker."""
    if per_query_k is None:
        per_query_k = top_k * 2          # μοχλός recall/κόστους ανά query

    queries = self._rewrite_query(query)

    seen: dict[str, dict] = {}           # chunk_id -> {score, text, meta}
    for q in queries:
        q_emb = self.embedding_model.encode([q]).tolist()[0]
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=per_query_k,
            include=["documents", "metadatas", "distances"],
        )
        ids   = res["ids"][0]
        docs  = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            score = 1.0 - dist           # cosine distance -> similarity
            if cid not in seen or score > seen[cid]["score"]:
                seen[cid] = {"score": score, "text": doc, "meta": meta or {}}

    candidates = list(seen.values())
    if not candidates:
        return []

    # Ο GA κρίνει ως προς το ORIGINAL query (όχι τα rewrites).
    # ── Προσαρμογή: αν ο _ga_rerank σου περιμένει άλλο σχήμα εισόδου
    #    (π.χ. σκέτα strings), κάνε το mapping εδώ.
    return self._ga_rerank(query, candidates)[:top_k]


def _build_context(self, docs: list[dict]) -> str:
    """Εκθέτει source + page σε header που το μοντέλο μπορεί να αντιγράψει.
    Τα ονόματα Figure/Table ζουν ΜΕΣΑ στο text του chunk — αν επιβίωσαν στο ingestion."""
    blocks = []
    for d in docs:
        meta   = d.get("meta", {})
        source = meta.get("source", "unknown source")   # ← ταίριαξε στα κλειδιά σου
        page   = meta.get("page")                       # ← ταίριαξε στα κλειδιά σου
        header = f"[Source: {source}" + (f" — p.{page}]" if page is not None else "]")
        blocks.append(f"{header}\n{d['text'].strip()}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(self, query: str, docs: list[dict]) -> str:
    if not docs:
        return "No relevant information found in the indexed manuals."

    context = self._build_context(docs)

    system   = "You are a technical assistant for industrial machinery manuals." \
    "Answer MOSTLY from the provided context." \
    "Anything you answer outside the provided context must be clearly indicated."   # <-- domain anchor σου
    few_shot = "CONTEXT:\n"
    "[Text — PumpManual.pdf, p.47]\n"
    "Figure 3.2 shows the impeller assembly. The clearance between impeller "
    "and casing must be 0.3-0.5 mm, measured with a feeler gauge.\n\n"
    "QUESTION: What is the correct impeller clearance?\n"
    "ANSWER: The impeller-to-casing clearance must be 0.3-0.5 mm, measured with "
    "a feeler gauge, as shown in Figure 3.2 (PumpManual.pdf, p.47).\n\n"
    "---\n\n"
    "CONTEXT:\n"
    "[Text — PumpManual.pdf, p.88]\n"
    "Table 5 lists torque values: M8 bolts 25 Nm, M10 bolts 49 Nm.\n\n"
    "QUESTION: What torque should I use on the M10 bolts?\n"
    "ANSWER: M10 bolts require 49 Nm of torque (Table 5, PumpManual.pdf, p.88).\n\n"
    "---\n\n"
    "CONTEXT:\n"
    "[Text — PumpManual.pdf, p.12]\n"
    "The unit operates at temperatures between -10°C and 60°C.\n\n"
    "QUESTION: What is the maximum rotation speed?\n"
    "ANSWER: The available manuals do not contain information about the maximum "
    "rotation speed.\n\n"
    "---\n\n"
    # ────────────────────────────────────────────────────────────────

    prompt = (
        f"{few_shot}"
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
                "think":   True,
                "options": {"num_ctx": 4096, "temperature": 0.1, "num_gpu": 99},
            },
            timeout=200,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"Error generating answer: {e}"