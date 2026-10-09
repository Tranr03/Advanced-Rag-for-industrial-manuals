import streamlit as st

from config import MANUALS_FOLDER
from rag_preprocessor import RAGPreprocessor
from rag_retriever_v2 import RAGRetriever

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Manual Assistant",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load resources (cached) ──────────────────────────────────────────────────
@st.cache_resource
def load_preprocessor():
    return RAGPreprocessor()

@st.cache_resource
def load_retriever(_preprocessor: RAGPreprocessor):
    return RAGRetriever(
        embedding_model=_preprocessor.embedding_model,
        collection=_preprocessor.collection,
    )

preprocessor = load_preprocessor()
retriever    = load_retriever(preprocessor)

# ─── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "queue" not in st.session_state:
    st.session_state.queue = []

# ─── Queue helpers ────────────────────────────────────────────────────────────
_STATUS_ICON = {
    "pending":    "⏳",
    "processing": "🔄",
    "done":       "✅",
    "skipped":    "⏭️",
    "failed":     "❌",
}

def _format_eta(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s_  = divmod(rem, 60)
    if h: return f"{h}h {m}m remaining"
    if m: return f"{m}m {s_}s remaining"
    return f"{s_}s remaining"

def _run_queue():
    """Process all pending queue items sequentially with live progress."""
    pending_indices = [
        i for i, item in enumerate(st.session_state.queue)
        if item["status"] == "pending"
    ]
    if not pending_indices:
        return

    with st.status(f"Processing {len(pending_indices)} manual(s)…", expanded=True) as status_box:
        for idx in pending_indices:
            item     = st.session_state.queue[idx]
            pdf_path = MANUALS_FOLDER / item["name"]
            item["status"] = "processing"

            st.write(f"📄 **{item['name']}**")

            try:
                if preprocessor.is_indexed(pdf_path.stem):
                    item["status"] = "skipped"
                    st.write("  ⏭️ Already indexed — skipped.")
                    continue

                prog_bar   = st.empty()
                time_label = st.empty()

                def on_progress(done, total, avg_secs, _p=prog_bar, _t=time_label):
                    pct = done / total if total > 0 else 0
                    _p.progress(pct, text=f"Vision: {done}/{total}  —  ~{_format_eta(avg_secs * (total - done))}")
                    _t.caption(f"Avg per page: {avg_secs:.1f}s")

                n_text, n_vision = preprocessor.process_pdf(
                    pdf_path,
                    status_fn=lambda _: None,
                    progress_fn=on_progress,
                )
                prog_bar.progress(1.0, text="Vision complete!")
                time_label.empty()
                item["status"]   = "done"
                item["n_text"]   = n_text
                item["n_vision"] = n_vision
                st.write(f"  ✅ {n_text} text · {n_vision} vision chunks indexed.")

            except Exception as e:
                item["status"] = "failed"
                item["error"]  = str(e)
                st.write(f"  ❌ Failed: {e}")

        done_n = sum(1 for item in st.session_state.queue if item["status"] == "done")
        skip_n = sum(1 for item in st.session_state.queue if item["status"] == "skipped")
        fail_n = sum(1 for item in st.session_state.queue if item["status"] == "failed")
        status_box.update(
            label=f"Done — ✅ {done_n} indexed  ⏭️ {skip_n} skipped  ❌ {fail_n} failed",
            state="complete" if fail_n == 0 else "error",
            expanded=False,
        )

    st.rerun()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Manual Assistant")
    st.divider()

    # ── Upload ────────────────────────────────────────────────────────────────
    st.subheader("Add to Queue")
    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        existing_keys = {(item["name"], item["size"]) for item in st.session_state.queue}
        added = 0
        for uf in uploaded_files:
            if (uf.name, uf.size) not in existing_keys:
                (MANUALS_FOLDER / uf.name).write_bytes(uf.read())
                st.session_state.queue.append({
                    "name":    uf.name,
                    "size":    uf.size,
                    "status":  "pending",
                    "n_text":  0,
                    "n_vision": 0,
                    "error":   None,
                })
                existing_keys.add((uf.name, uf.size))
                added += 1
        if added:
            st.rerun()

    st.divider()

    # ── Queue display ─────────────────────────────────────────────────────────
    st.subheader("Queue")
    if not st.session_state.queue:
        st.caption("No files queued yet.")
    else:
        for item in st.session_state.queue:
            icon = _STATUS_ICON.get(item["status"], "❓")
            if item["status"] == "done":
                st.markdown(f"{icon} **{item['name']}**  \n`{item['n_text']} text · {item['n_vision']} vision`")
            elif item["status"] == "failed":
                st.markdown(f"{icon} **{item['name']}**  \n`{item['error']}`")
            elif item["status"] == "skipped":
                st.markdown(f"{icon} **{item['name']}**  \n`already indexed`")
            else:
                st.markdown(f"{icon} **{item['name']}**")

        pending_n = sum(1 for item in st.session_state.queue if item["status"] == "pending")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ Process Queue", type="primary",
                         disabled=(pending_n == 0), use_container_width=True):
                _run_queue()
        with col2:
            if st.button("🗑️ Clear Queue", use_container_width=True):
                st.session_state.queue = []
                st.rerun()

    st.divider()

    # ── Indexed manuals ───────────────────────────────────────────────────────
    st.subheader("Indexed Manuals")
    sources = retriever.list_sources()
    if sources:
        all_meta = retriever.collection.get(include=["metadatas"])
        for src in sources:
            count = sum(1 for m in all_meta["metadatas"] if m["source"] == src)
            st.markdown(f"📄 **{src}**  \n`{count} chunks`")
    else:
        st.caption("No manuals indexed yet.")

    st.divider()

    # ── Delete manual ─────────────────────────────────────────────────────────
    st.subheader("Delete Manual")
    if sources:
        to_delete = st.selectbox("Select manual", sources, label_visibility="collapsed")
        if st.button("🗑️ Delete", type="primary", use_container_width=True):
            st.session_state["_confirm_delete"] = to_delete

        if st.session_state.get("_confirm_delete") == to_delete:
            st.warning(f"Delete **{to_delete}**? This cannot be undone.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, delete", use_container_width=True):
                    n = preprocessor.delete_pdf(to_delete)
                    st.session_state.pop("_confirm_delete", None)
                    st.success(f"Deleted {n} documents.")
                    st.rerun()
            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.pop("_confirm_delete", None)
                    st.rerun()
    else:
        st.caption("No manuals to delete.")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ─── Main chat ────────────────────────────────────────────────────────────────
st.title("Industrial Manual Assistant")
st.caption("Ask questions about your indexed machinery manuals.")

# Source filter
sources_list = retriever.list_sources()
if sources_list:
    col_filter, col_spacer = st.columns([2, 3])
    with col_filter:
        selected_source = st.selectbox(
            "Search in",
            options=["All manuals"] + sources_list,
            label_visibility="collapsed"
        )
    source_filter = None if selected_source == "All manuals" else selected_source
else:
    source_filter = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for src in msg["sources"]:
                    st.caption(f"📄 **{src['source']}** — p.{src['page']}  |  score: {src['score']:.2f}")

if prompt := st.chat_input("Ask about your manuals..."):
    if retriever.count() == 0:
        st.warning("Please upload at least one manual first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching manuals..."):
                answer, docs = retriever.query(prompt, source_filter=source_filter)
            st.markdown(answer)
            sources = [
                {
                    "source": d["metadata"]["source"],
                    "page":   d["metadata"].get("page", "?"),
                    "score":  d["score"]
                }
                for d in docs
            ]
            if sources:
                with st.expander("Sources", expanded=False):
                    for src in sources:
                        st.caption(f"📄 **{src['source']}** — p.{src['page']}  |  score: {src['score']:.2f}")

        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "sources": sources
        })
