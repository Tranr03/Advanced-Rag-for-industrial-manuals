# RAG Diploma Project — CLAUDE.md

## Περιγραφή
RAG pipeline για ανάγνωση και επίλυση προβλημάτων βιομηχανικών μηχανημάτων μέσα από operation manuals (PDFs).

## Αρχιτεκτονική

```
ragV2/
├── config.py                    # Shared constants (paths, models, chunking params)
├── rag_preprocessor.py          # class RAGPreprocessor — PDF ingestion pipeline
├── rag_retriever.py             # class RAGRetriever — retrieval + answer generation
├── app.py                       # Streamlit UI — χρησιμοποιεί τα παραπάνω classes
├── rag_preprocessing_.ipynb     # Notebook — thin wrapper του RAGPreprocessor
├── rag_vectordb_retrieval.ipynb # Notebook — thin wrapper του RAGRetriever
├── rag_evaluation.ipynb         # Notebook — RAGAS evaluation
└── data/
    ├── manuals/                 # PDF αρχεία εισόδου
    ├── markdown_temp/           # Εξαγόμενο κείμενο ανά PDF
    ├── chunks_temp/             # JSON chunks ανά PDF + all_chunks.json
    ├── images_temp/             # PNG εικόνες σελίδων + enriched_images.json
    └── vector_db/               # ChromaDB persistent storage
```

## Models
| Χρήση | Μοντέλο |
|-------|---------|
| Embeddings | `BAAI/bge-base-en-v1.5` (local, SentenceTransformer) |
| Answer generation | `llama3.1:8b` (Ollama) |
| Vision / diagrams | `qwen3-vl:8b` (Ollama) |

## Βασικές κλάσεις

### RAGPreprocessor (`rag_preprocessor.py`)
- `process_pdf(pdf_path, status_fn=None)` → `(n_text_chunks, n_vision_chunks)`
- `process_all(manuals_folder, status_fn=None)` → skip αν ήδη indexed
- `is_indexed(pdf_stem)` → bool

### RAGRetriever (`rag_retriever.py`)
- `retrieve(query, n_results=5)` → `list[{text, metadata, score}]`
- `generate_answer(query, docs)` → str
- `query(question, n_results=5)` → `(answer, docs)` — convenience method
- `list_sources()` → sorted list of indexed PDFs
- `count()` → αριθμός documents στη βάση

## Σημαντικές αποφάσεις
- **Chunking ανά σελίδα** — κάθε chunk έχει `page` στο metadata για citation
- **Source diversity** — max 2 docs ανά source στο context (στο `_build_context`)
- **ChromaDB cosine similarity** — `hnsw:space: cosine`
- **Vision processing** — μόνο σελίδες με `obj.type == 2` (graphics objects)
- **Incremental save** — το evaluation notebook κάνει save μετά από κάθε ερώτηση για recovery από crash

## Βήματα διπλωματικής
1. ✅ Scripts (preprocessing, vectordb/retrieval, evaluation)
2. 🔄 UI (Streamlit app — σε εξέλιξη)
3. ⬜ Scale: 1 PDF → 25-30 PDFs
4. ⬜ Finetune embedding model για μηχανολογική ορολογία
5. ⬜ Τελικό evaluation

## Conventions
- Τα notebooks είναι **thin wrappers** — η λογική ανήκει στα `.py` files
- Αλλαγές στα notebooks δίνονται ως **copy-paste blocks** (το NotebookEdit tool δεν λειτουργεί αξιόπιστα)
- Ζήτα πάντα άδεια πριν κάνεις αλλαγές στον κώδικα

## Εκκίνηση app
```bash
cd C:\Users\User\Desktop\rag_diploma\ragV2
venv\Scripts\activate
streamlit run app.py
```
