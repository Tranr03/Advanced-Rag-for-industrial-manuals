import streamlit as st

from config import MANUALS_FOLDER
from rag_preprocessor import RAGPreprocessor
from rag_retriever import RAGRetriever

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
def load_retriever():
    return RAGRetriever()

preprocessor = load_preprocessor()
retriever    = load_retriever()

# ─── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Manual Assistant")
    st.divider()

    # Upload
    st.subheader("Add Manual")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        pdf_path = MANUALS_FOLDER / uploaded_file.name
        pdf_path.write_bytes(uploaded_file.read())
        is_resume = preprocessor.is_indexed(pdf_path.stem)
        label     = f"Resuming '{uploaded_file.name}'..." if is_resume else f"Processing '{uploaded_file.name}'..."

        with st.status(label, expanded=True) as s:
            progress_bar = st.progress(0.0)
            time_label   = st.empty()

            def _format_eta(seconds: float) -> str:
                seconds = int(seconds)
                h, rem  = divmod(seconds, 3600)
                m, s_   = divmod(rem, 60)
                if h:    return f"{h}h {m}m remaining"
                if m:    return f"{m}m {s_}s remaining"
                return f"{s_}s remaining"

            def on_progress(done: int, total: int, avg_secs: float):
                pct = done / total if total > 0 else 0
                eta = _format_eta(avg_secs * (total - done))
                progress_bar.progress(pct, text=f"Vision: {done}/{total} pages  —  ~{eta}")
                time_label.caption(f"Avg per page: {avg_secs:.1f}s")

            n_text, n_vision = preprocessor.process_pdf(
                pdf_path,
                status_fn=lambda msg: s.update(label=msg),
                progress_fn=on_progress
            )
            progress_bar.progress(1.0, text="Vision complete!")
            time_label.empty()
            s.update(label="Done!", state="complete", expanded=False)

        st.success(f"✓ {n_text} text chunks  |  {n_vision} vision chunks indexed.")
        st.rerun()

    st.divider()

    # Indexed manuals
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

    # Delete manual
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
