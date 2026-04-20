import json
import time
import base64
import requests
import pypdfium2 as pdfium
from pathlib import Path
from io import BytesIO
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

from config import (
    EMBEDDING_MODEL, OLLAMA_URL, VISION_MODEL,
    CHUNKS_FOLDER, IMAGES_FOLDER, MARKDOWN_FOLDER, VECTOR_DB_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME
)


class RAGPreprocessor:
    """
    Handles the full PDF ingestion pipeline:
      1. Text extraction per page
      2. Chunking + embedding + ChromaDB indexing
      3. Vision model description for pages with graphics
    """

    def __init__(self):
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

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

    def process_pdf(self, pdf_path: Path, status_fn=None, progress_fn=None) -> tuple[int, int]:
        """
        Full pipeline for a single PDF.
        Returns (n_text_chunks, n_vision_chunks).
        status_fn:   optional callable(str) for live status text updates.
        progress_fn: optional callable(done, total, avg_secs_per_page) for progress bar.
        """
        pdf_path = Path(pdf_path)
        self._log(status_fn, f"Opening {pdf_path.name}...")

        doc = pdfium.PdfDocument(str(pdf_path.resolve()))
        try:
            n_pages    = len(doc)
            page_texts = self._extract_text(doc, n_pages, status_fn)

            # Save full markdown
            full_text = "\n\n".join(page_texts)
            (MARKDOWN_FOLDER / f"{pdf_path.stem}.md").write_text(full_text, encoding="utf-8")

            n_text   = self._index_text(pdf_path, page_texts, status_fn)
            n_vision = self._index_vision(doc, pdf_path, page_texts, n_pages, status_fn, progress_fn)

        finally:
            doc.close()

        self._log(status_fn, f"Done — {n_text} text chunks, {n_vision} vision chunks.")
        return n_text, n_vision

    def process_all(self, manuals_folder: Path, status_fn=None) -> dict:
        """
        Processes all PDFs in a folder.
        Returns {filename: (n_text, n_vision)}.
        """
        manuals_folder = Path(manuals_folder)
        pdf_files      = list(manuals_folder.glob("*.pdf"))
        self._log(status_fn, f"Found {len(pdf_files)} PDFs.")

        results = {}
        for pdf_file in pdf_files:
            existing = self.collection.get(where={"source": {"$eq": pdf_file.stem}})
            if existing["ids"]:
                self._log(status_fn, f"Skipping '{pdf_file.name}' (already indexed).")
                continue
            results[pdf_file.name] = self.process_pdf(pdf_file, status_fn)

        return results

    def is_indexed(self, pdf_stem: str) -> bool:
        return bool(self.collection.get(where={"source": {"$eq": pdf_stem}})["ids"])

    def delete_pdf(self, pdf_stem: str) -> int:
        """
        Removes all documents for a given PDF from ChromaDB.
        Also deletes associated chunk JSON and image files.
        Returns number of documents deleted.
        """
        existing = self.collection.get(where={"source": {"$eq": pdf_stem}}, include=[])
        ids      = existing["ids"]
        if ids:
            self.collection.delete(ids=ids)

        # Διαγραφή αρχείων από δίσκο
        for f in CHUNKS_FOLDER.glob(f"{pdf_stem}*.json"):
            f.unlink(missing_ok=True)
        for f in IMAGES_FOLDER.glob(f"{pdf_stem}*.png"):
            f.unlink(missing_ok=True)
        for f in IMAGES_FOLDER.glob(f"{pdf_stem}*.json"):
            f.unlink(missing_ok=True)
        md = MARKDOWN_FOLDER / f"{pdf_stem}.md"
        md.unlink(missing_ok=True)

        return len(ids)

    # ─── Private ──────────────────────────────────────────────────────────────

    def _extract_text(self, doc, n_pages: int, status_fn) -> list[str]:
        self._log(status_fn, f"Extracting text from {n_pages} pages...")
        page_texts = []
        for page_idx in range(n_pages):
            page     = doc[page_idx]
            textpage = page.get_textpage()
            try:
                page_texts.append(textpage.get_text_range())
            finally:
                textpage.close()
        return page_texts

    def _index_text(self, pdf_path: Path, page_texts: list[str], status_fn) -> int:
        # Αν τα text chunks υπάρχουν ήδη, skip
        existing = self.collection.get(
            where={"$and": [{"source": {"$eq": pdf_path.stem}}, {"type": {"$eq": "text"}}]},
            include=[]
        )
        if existing["ids"]:
            self._log(status_fn, f"Text already indexed ({len(existing['ids'])} chunks) — skipping.")
            return len(existing["ids"])

        chunks_data = []
        for page_num, page_text in enumerate(page_texts, start=1):
            if not page_text.strip():
                continue
            for i, chunk in enumerate(self.splitter.split_text(page_text)):
                chunks_data.append({
                    "chunk_id": f"{pdf_path.stem}_p{page_num}_chunk_{i}",
                    "source":   pdf_path.stem,
                    "page":     page_num,
                    "text":     chunk
                })

        if not chunks_data:
            return 0

        self._log(status_fn, f"Embedding {len(chunks_data)} text chunks...")
        texts      = [c["text"] for c in chunks_data]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False).tolist()

        self.collection.add(
            ids       = [c["chunk_id"] for c in chunks_data],
            embeddings= embeddings,
            documents = texts,
            metadatas = [{"source": c["source"], "page": c["page"], "type": "text"}
                         for c in chunks_data]
        )

        (CHUNKS_FOLDER / f"{pdf_path.stem}_chunks.json").write_text(
            json.dumps(chunks_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return len(chunks_data)

    def _index_vision(self, doc, pdf_path: Path, page_texts: list[str],
                      n_pages: int, status_fn, progress_fn=None) -> int:
        # Βρες ποιες σελίδες έχουν ήδη γίνει vision — για resume από crash
        existing_vision = self.collection.get(
            where={"$and": [{"source": {"$eq": pdf_path.stem}}, {"type": {"$eq": "image_description"}}]},
            include=[]
        )
        already_done = set(existing_vision["ids"])
        if already_done:
            self._log(status_fn, f"Resuming vision — {len(already_done)} pages already done, continuing...")

        # Pre-scan: βρες όλες τις σελίδες με γραφικά που δεν έχουν γίνει ακόμα
        pending_pages = []
        for page_idx in range(n_pages):
            vid = f"{pdf_path.stem}_vision_p{page_idx+1}"
            if vid in already_done:
                continue
            page         = doc[page_idx]
            has_graphics = any(getattr(obj, "type", None) == 2 for obj in page.get_objects())
            if has_graphics:
                pending_pages.append(page_idx)

        total_pending = len(pending_pages)
        self._log(status_fn, f"Vision — {total_pending} pages to process...")

        vision_count = len(already_done)
        elapsed_times = []

        for done_count, page_idx in enumerate(pending_pages, start=1):
            self._log(status_fn, f"Vision — page {page_idx + 1}/{n_pages}  ({done_count}/{total_pending})")

            t_start = time.time()

            bitmap = doc[page_idx].render(scale=2)
            try:
                pil_img  = bitmap.to_pil()
                img_path = IMAGES_FOLDER / f"{pdf_path.stem}_page{page_idx+1:03d}.png"
                pil_img.save(str(img_path), "PNG")
            finally:
                bitmap.close()

            context     = page_texts[page_idx][:300]
            description = self._vision_describe(str(img_path), context)

            elapsed_times.append(time.time() - t_start)

            # Progress callback — rolling avg των τελευταίων 5 σελίδων
            if progress_fn:
                avg_secs = sum(elapsed_times[-5:]) / len(elapsed_times[-5:])
                progress_fn(done_count, total_pending, avg_secs)

            if not description or "NO_DIAGRAM" in description.upper():
                continue

            vid        = f"{pdf_path.stem}_vision_p{page_idx+1}"
            vision_emb = self.embedding_model.encode([description], show_progress_bar=False).tolist()
            self.collection.add(
                ids       = [vid],
                embeddings= vision_emb,
                documents = [description],
                metadatas = [{"source": pdf_path.stem, "page": page_idx + 1,
                              "type": "image_description", "image_path": str(img_path)}]
            )
            vision_count += 1

        return vision_count

    def _vision_describe(self, image_path: str, context: str) -> str:
        img = Image.open(image_path)
        if img.width > 1024:
            ratio = 1024 / img.width
            img   = img.resize((1024, int(img.height * ratio)), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = (
            "Analyze this industrial manual page. "
            "If it contains diagrams, schematics, tables or technical drawings, "
            "describe all components, labels, measurements and relationships in detail. "
            "If it contains only plain text or decorative elements, respond with 'NO_DIAGRAM'. "
            f"Surrounding context: {context[:200]}"
        )
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": VISION_MODEL, "prompt": prompt,
                      "images": [img_b64], "stream": False,
                      "options": {"num_ctx": 4096, "temperature": 0.1}},
                timeout=120
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception:
            return ""

    @staticmethod
    def _log(status_fn, msg: str):
        if status_fn:
            status_fn(msg)
        else:
            print(msg)
