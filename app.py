# app.py
"""
Adaptive Multi-Stage RAG System — Streamlit UI
Run with: streamlit run app.py
"""

import os
import sys
import logging
import tempfile
import streamlit as st

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/rag_system.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── Path fix so modules resolve correctly ────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from rag.pipeline import RAGPipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background: #0d1117; color: #e6edf3; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #58a6ff; }

    .main-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem; font-weight: 600; color: #58a6ff;
        border-bottom: 2px solid #21262d;
        padding-bottom: 0.5rem; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 1rem; text-align: center;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem; font-weight: 700; color: #58a6ff;
    }
    .metric-label {
        font-size: 0.8rem; color: #8b949e;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .badge {
        display: inline-block; padding: 0.2em 0.7em;
        border-radius: 999px; font-size: 0.78rem; font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
    }
    .badge-factual       { background: #1f4870; color: #58a6ff; }
    .badge-summarization { background: #1f3d1f; color: #3fb950; }
    .badge-analytical    { background: #3d2a00; color: #d29922; }
    .answer-box {
        background: #161b22; border: 1px solid #30363d;
        border-left: 4px solid #58a6ff; border-radius: 8px;
        padding: 1.5rem; margin: 1rem 0; line-height: 1.8;
    }
    .source-card {
        background: #0d1117; border: 1px solid #21262d;
        border-radius: 6px; padding: 0.8rem 1rem; margin: 0.4rem 0;
        font-size: 0.85rem; color: #8b949e;
        font-family: 'IBM Plex Mono', monospace;
    }
    .stButton>button {
        background: #238636 !important; color: white !important;
        border: none !important; border-radius: 6px !important;
        font-weight: 600 !important; padding: 0.5rem 1.5rem !important;
    }
    .stButton>button:hover { background: #2ea043 !important; }
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        background: #161b22 !important; color: #e6edf3 !important;
        border: 1px solid #30363d !important; border-radius: 6px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }
    .stSelectbox>div>div {
        background: #161b22 !important; color: #e6edf3 !important;
        border: 1px solid #30363d !important;
    }
    .stSidebar { background: #161b22 !important; }
    .stFileUploader {
        background: #161b22 !important;
        border: 1px dashed #30363d !important; border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
if "history" not in st.session_state:
    st.session_state.history = []

pipeline: RAGPipeline = st.session_state.pipeline

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    api_key = st.text_input(
        "OpenRouter API Key", type="password",
        placeholder="sk-or-...", help="Get your key at openrouter.ai",
    )
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key

    st.markdown("---")
    st.markdown("### 🤖 Model Settings")

    model_choice = st.selectbox(
        "LLM Model",
        options=[
            "google/gemma-3-4b-it:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "qwen/qwen3-8b:free",
            "deepseek/deepseek-r1:free",
            "mistralai/mistral-small-3.1-24b",
        ],
        index=0,
    )
    pipeline.llm_model = model_choice

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.markdown("---")
    st.markdown("### 🔧 Retrieval Settings")

    semantic_weight = st.slider(
        "Semantic vs Keyword weight", 0.0, 1.0, 0.6, 0.05,
        help="1.0 = pure semantic, 0.0 = pure keyword",
    )
    pipeline.retriever.alpha = semantic_weight

    candidate_pool = st.slider(
        "Candidate pool size", 5, 20, 10,
        help="How many candidates to retrieve before re-ranking",
    )
    pipeline.candidate_pool = candidate_pool

    use_cache = st.checkbox("Cache answers", value=True)
    pipeline.use_cache = use_cache

    st.markdown("---")
    if st.button("🗑️ Reset All Data"):
        pipeline.reset()
        st.session_state.history = []
        st.success("Pipeline reset!")

    st.markdown("---")
    st.markdown("### 📊 Status")
    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{pipeline.document_count}</div>'
        f'<div class="metric-label">Documents</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{pipeline.chunk_count}</div>'
        f'<div class="metric-label">Indexed Chunks</div></div>',
        unsafe_allow_html=True,
    )

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">🔍 Adaptive Multi-Stage RAG System</div>',
    unsafe_allow_html=True,
)

tab_ingest, tab_query, tab_history = st.tabs(
    ["📄 Document Ingestion", "💬 Query Interface", "🕒 Query History"]
)

# ─── TAB 1: Ingestion ────────────────────────────────────────────────────────
with tab_ingest:
    st.markdown("#### Upload Documents")
    st.markdown("Upload **PDF** or **TXT** files. Multiple files are supported.")

    uploaded_files = st.file_uploader(
        "Drop files here", type=["pdf", "txt"], accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("📥 Ingest Documents"):
            progress = st.progress(0)
            status = st.empty()
            total = len(uploaded_files)
            for i, uf in enumerate(uploaded_files):
                status.info(f"Processing: {uf.name} …")
                suffix = ".pdf" if uf.name.endswith(".pdf") else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name
                try:
                    n = pipeline.ingest_file(tmp_path)
                    st.success(f"✅ **{uf.name}** — {n} chunks indexed")
                except Exception as e:
                    st.error(f"❌ Failed to ingest {uf.name}: {e}")
                progress.progress((i + 1) / total)
            status.empty()
            st.balloons()

    st.markdown("---")
    st.markdown("#### Or paste text directly")
    paste_name = st.text_input("Source label", "pasted_text")
    paste_text = st.text_area("Paste document text here", height=200)
    if st.button("📥 Ingest Pasted Text") and paste_text.strip():
        try:
            n = pipeline.ingest_text(paste_text, source_name=paste_name)
            st.success(f"✅ Ingested {n} chunks from pasted text.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

    if pipeline.document_count > 0:
        st.markdown("---")
        st.markdown("#### Ingested Sources")
        for src in pipeline.ingested_files:
            st.markdown(f"- `{src}`")

# ─── TAB 2: Query ────────────────────────────────────────────────────────────
with tab_query:
    if not pipeline.is_ready:
        st.warning("⚠️ No documents indexed yet. Go to **Document Ingestion** tab to add files.")
    elif not os.getenv("OPENROUTER_API_KEY"):
        st.warning("⚠️ OpenRouter API key not set. Enter it in the sidebar.")
    else:
        st.markdown("#### Ask a Question")

        col1, col2 = st.columns([3, 1])
        with col1:
            user_query = st.text_area(
                "Your question",
                placeholder="e.g. What are the main contributions of this paper?",
                height=100, label_visibility="collapsed",
            )
        with col2:
            override = st.selectbox(
                "Query type override",
                ["Auto-detect", "factual", "summarization", "analytical"],
            )
            submit = st.button("🔍 Ask", use_container_width=True)

        if submit and user_query.strip():
            override_type = None if override == "Auto-detect" else override
            with st.spinner("🧠 Retrieving and generating …"):
                try:
                    result = pipeline.query(
                        user_query, override_type=override_type,
                        temperature=temperature,
                    )
                    st.session_state.history.insert(0, {"query": user_query, "result": result})
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    result = None

            if result:
                badge_class = f"badge-{result['query_type']}"
                cached_tag = " 🔄 cached" if result.get("cached") else ""
                st.markdown(
                    f'<span class="badge {badge_class}">{result["query_type"].upper()}</span>'
                    f'&nbsp;<small style="color:#8b949e;">'
                    f'{result["chunks_used"]} chunks used &bull; {result["model"]}{cached_tag}'
                    f'</small>',
                    unsafe_allow_html=True,
                )

                st.markdown("#### 💡 Answer")
                st.markdown(
                    f'<div class="answer-box">{result["answer"]}</div>',
                    unsafe_allow_html=True,
                )

                with st.expander("📚 Retrieved Context Chunks", expanded=False):
                    for i, src in enumerate(result["sources"], 1):
                        preview = src[:400] + ("…" if len(src) > 400 else "")
                        st.markdown(
                            f'<div class="source-card">[{i}] {preview}</div>',
                            unsafe_allow_html=True,
                        )

                with st.expander("📋 Export as JSON", expanded=False):
                    import json
                    export = {
                        "query": user_query,
                        "query_type": result["query_type"],
                        "answer": result["answer"],
                        "model": result["model"],
                        "chunks_used": result["chunks_used"],
                        "sources": result["sources"],
                    }
                    st.code(json.dumps(export, indent=2), language="json")

# ─── TAB 3: History ──────────────────────────────────────────────────────────
with tab_history:
    if not st.session_state.history:
        st.info("No queries yet. Ask something in the Query Interface tab.")
    else:
        st.markdown(f"#### {len(st.session_state.history)} queries this session")
        for i, item in enumerate(st.session_state.history):
            q = item["query"]
            r = item["result"]
            with st.expander(f"**Q{i+1}:** {q[:80]}{'…' if len(q) > 80 else ''}", expanded=(i == 0)):
                badge_class = f"badge-{r['query_type']}"
                st.markdown(
                    f'<span class="badge {badge_class}">{r["query_type"].upper()}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="answer-box">{r["answer"]}</div>',
                    unsafe_allow_html=True,
                )