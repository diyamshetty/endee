"""
app.py — ArXiv-RAG Streamlit frontend.
A polished, dark-themed UI for exploring recent ArXiv LLM papers with RAG.
"""
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ArXiv-RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Font: Inter ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Root variables ── */
    :root {
        --bg-primary:   #0a0e1a;
        --bg-card:      #111827;
        --bg-card2:     #1a2235;
        --accent:       #6366f1;
        --accent-light: #818cf8;
        --accent-glow:  rgba(99,102,241,0.25);
        --success:      #10b981;
        --warning:      #f59e0b;
        --error:        #ef4444;
        --text-primary: #f1f5f9;
        --text-muted:   #94a3b8;
        --border:       rgba(99,102,241,0.2);
        --radius:       12px;
    }

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }
    .main .block-container { padding: 1.5rem 2rem; max-width: 1200px; }
    section[data-testid="stSidebar"] { background: var(--bg-card) !important; border-right: 1px solid var(--border); }

    /* ── Hero header ── */
    .hero-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 40%, #0a0e1a 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 2rem 2.5rem;
        margin-bottom: 1.75rem;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute; top: -50%; left: -20%;
        width: 60%; height: 200%;
        background: radial-gradient(ellipse, var(--accent-glow) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ffffff 0%, var(--accent-light) 60%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.4rem 0;
    }
    .hero-subtitle {
        color: var(--text-muted);
        font-size: 1rem;
        font-weight: 400;
        margin: 0;
    }

    /* ── Status badges ── */
    .badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-weight: 500;
        letter-spacing: 0.03em;
    }
    .badge-success { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
    .badge-warning { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
    .badge-info    { background: var(--accent-glow);    color: var(--accent-light); border: 1px solid var(--border); }

    /* ── Paper cards ── */
    .paper-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .paper-card:hover {
        border-color: var(--accent);
        box-shadow: 0 0 20px var(--accent-glow);
    }
    .paper-rank {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: var(--accent-light);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }
    .paper-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.4rem;
        line-height: 1.4;
    }
    .paper-meta {
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-bottom: 0.75rem;
    }
    .paper-abstract {
        font-size: 0.87rem;
        color: #cbd5e1;
        line-height: 1.65;
        border-left: 2px solid var(--accent);
        padding-left: 0.9rem;
        margin: 0.6rem 0;
    }
    .paper-link {
        font-size: 0.8rem;
        color: var(--accent-light);
        text-decoration: none;
    }
    .paper-link:hover { text-decoration: underline; }

    /* ── Score chip ── */
    .score-chip {
        display: inline-block;
        background: linear-gradient(135deg, var(--accent) 0%, #7c3aed 100%);
        color: white;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        margin-left: 0.5rem;
        letter-spacing: 0.02em;
    }

    /* ── Answer box ── */
    .answer-box {
        background: linear-gradient(135deg, #0f172a 0%, #1a1033 100%);
        border: 1px solid rgba(99,102,241,0.35);
        border-radius: var(--radius);
        padding: 1.5rem 1.75rem;
        margin-top: 0.5rem;
        box-shadow: 0 0 30px rgba(99,102,241,0.1);
        font-size: 0.95rem;
        line-height: 1.75;
        color: #e2e8f0;
    }
    .answer-header {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--accent-light);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Query box ── */
    .stTextArea textarea, .stTextInput input {
        background: var(--bg-card2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--accent-glow) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover { transform: translateY(-1px) !important; }

    /* ── Sidebar ── */
    .sidebar-title {
        font-weight: 700;
        font-size: 1.1rem;
        color: var(--accent-light);
        margin-bottom: 0.25rem;
    }
    .sidebar-section {
        background: rgba(99,102,241,0.07);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin: 0.75rem 0;
        font-size: 0.83rem;
    }
    .sidebar-label { color: var(--text-muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em; }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: var(--accent) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Imports (after CSS) ────────────────────────────────────────────────────────
import time
import ingest
import embedder
import vector_store
import llm

# ── Session state defaults ─────────────────────────────────────────────────────
if "index" not in st.session_state:
    st.session_state.index = None
if "endee_client" not in st.session_state:
    st.session_state.endee_client = None
if "paper_count" not in st.session_state:
    st.session_state.paper_count = 0
if "index_status" not in st.session_state:
    st.session_state.index_status = "not_built"   # not_built | building | ready | error
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_query" not in st.session_state:
    st.session_state.last_query = ""


# ── Helpers ────────────────────────────────────────────────────────────────────
def _status_badge(status: str) -> str:
    if status == "ready":
        return '<span class="badge badge-success">● READY</span>'
    elif status == "building":
        return '<span class="badge badge-warning">◌ BUILDING</span>'
    elif status == "error":
        return '<span class="badge" style="background:rgba(239,68,68,.15);color:#f87171;border:1px solid rgba(239,68,68,.3)">✕ ERROR</span>'
    return '<span class="badge badge-info">○ NOT BUILT</span>'


def _paper_card_html(rank: int, paper: dict) -> str:
    abstract_snippet = paper["abstract"][:380] + ("…" if len(paper["abstract"]) > 380 else "")
    authors = paper.get("authors", "")
    if authors:
        authors = f"👥 {authors}"
    date = f"📅 {paper['published']}" if paper.get("published") else ""
    meta_parts = [p for p in [authors, date] if p]
    meta_str = " &nbsp;|&nbsp; ".join(meta_parts)

    return f"""
    <div class="paper-card">
        <div class="paper-rank">#{rank} &nbsp;·&nbsp; Similarity
            <span class="score-chip">{paper['score']:.3f}</span>
        </div>
        <div class="paper-title">{paper['title']}</div>
        <div class="paper-meta">{meta_str}</div>
        <div class="paper-abstract">{abstract_snippet}</div>
        <a class="paper-link" href="{paper['url']}" target="_blank">🔗 View on ArXiv →</a>
    </div>
    """


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">📚 ArXiv-RAG</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#94a3b8;font-size:0.82rem;margin-bottom:1rem;">'
        "LLM research · Powered by Endee + LLaMA 3.1"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Index status ──
    st.markdown(
        f"""
        <div class="sidebar-section">
            <div class="sidebar-label">Index Status</div>
            <div style="margin-top:0.4rem;">{_status_badge(st.session_state.index_status)}</div>
            {"<div style='color:#94a3b8;font-size:0.78rem;margin-top:0.5rem;'>"+str(st.session_state.paper_count)+" papers indexed</div>" if st.session_state.paper_count else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Build index button ──
    build_label = (
        "🔄 Rebuild Index" if st.session_state.index_status == "ready" else "⚡ Build Index"
    )
    if st.button(build_label, use_container_width=True, type="primary"):
        st.session_state.index_status = "building"
        st.session_state.index = None
        st.session_state.endee_client = None

        progress_bar = st.progress(0, text="Initializing...")
        status_text = st.empty()

        try:
            # Step 1 — Fetch papers
            status_text.markdown("*📥 Fetching ArXiv papers …*")
            progress_bar.progress(5, text="Fetching papers from ArXiv…")
            papers = ingest.fetch_papers()
            progress_bar.progress(20, text=f"Fetched {len(papers)} papers")
            st.session_state.paper_count = len(papers)

            # Step 2 — Connect to Endee
            status_text.markdown("*🔗 Connecting to Endee …*")
            progress_bar.progress(25, text="Connecting to Endee server…")
            client = vector_store.init_client()
            index = vector_store.create_or_get_index(client)
            st.session_state.endee_client = client
            progress_bar.progress(30, text="Endee index ready")

            # Step 3 — Generate embeddings + upsert (streamed)
            status_text.markdown("*🧠 Generating embeddings and indexing …*")
            abstracts = [p["abstract"] for p in papers]
            total = len(papers)
            all_embeddings: list[list[float]] = []
            embedded_so_far = 0

            for batch_vecs in embedder.embed_texts_batched(abstracts, batch_size=64):
                all_embeddings.extend(batch_vecs)
                embedded_so_far += len(batch_vecs)
                pct = 30 + int((embedded_so_far / total) * 40)
                progress_bar.progress(pct, text=f"Embedding {embedded_so_far}/{total} abstracts…")

            # Step 4 — Upsert into Endee
            status_text.markdown("*⬆️ Upserting vectors into Endee …*")

            def _upsert_progress(done, total_count):
                pct = 70 + int((done / total_count) * 28)
                progress_bar.progress(pct, text=f"Upserting {done}/{total_count} vectors…")

            vector_store.upsert_papers(index, papers, all_embeddings, progress_cb=_upsert_progress)

            # Done
            progress_bar.progress(100, text="✅ Index built successfully!")
            status_text.empty()
            st.session_state.index = index
            st.session_state.index_status = "ready"
            st.success(f"✅ Index built with {len(papers)} papers!", icon="🎉")

        except Exception as exc:
            st.session_state.index_status = "error"
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ **Build failed:** {exc}", icon="🚨")

    st.markdown("---")

    # ── Info panel ──
    st.markdown(
        """
        <div class="sidebar-section">
            <div class="sidebar-label">Tech Stack</div>
            <div style="margin-top:0.5rem;color:#e2e8f0;line-height:1.8;font-size:0.82rem;">
                📄 <b>Data</b>: ArXiv API (2 000 papers)<br>
                🧠 <b>Embed</b>: all-MiniLM-L6-v2<br>
                🗄️ <b>Store</b>: Endee (cosine, INT8)<br>
                🦙 <b>LLM</b>: LLaMA 3.1 8B via Ollama<br>
                🌐 <b>UI</b>: Streamlit
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="color:#475569;font-size:0.72rem;margin-top:1rem;text-align:center;">
        Endee @ localhost:8080 · Ollama @ localhost:11434
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-header">
        <div class="hero-title">📚 ArXiv-RAG</div>
        <div class="hero-subtitle">
            Ask questions about 2 000 recent machine learning papers · powered by a local LLaMA 3.1 8B
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Query Section ──
col_input, col_btn = st.columns([5, 1], vertical_alignment="bottom")

with col_input:
    query = st.text_input(
        "🔍 Ask a question about recent LLM research",
        value=st.session_state.last_query,
        placeholder="e.g. What are the latest techniques for reducing hallucinations in LLMs?",
        label_visibility="visible",
        key="query_input",
    )

with col_btn:
    search_clicked = st.button(
        "Search & Ask →",
        use_container_width=True,
        type="primary",
        disabled=(st.session_state.index_status != "ready"),
    )

# ── Run Search ──
if search_clicked and query.strip():
    st.session_state.last_query = query.strip()

    with st.spinner("🔎 Searching Endee index…"):
        q_vec = embedder.embed_texts([query.strip()])[0]
        results = vector_store.search(st.session_state.index, q_vec, top_k=3)
        st.session_state.last_results = results

    with st.spinner("🦙 Generating answer with LLaMA 3.1…"):
        answer = llm.generate_answer(query.strip(), results)
        st.session_state.last_answer = answer

elif search_clicked and not query.strip():
    st.warning("Please enter a question before searching.", icon="✏️")

elif st.session_state.index_status != "ready" and search_clicked:
    st.warning("Please build the index first using the sidebar button.", icon="⚠️")

# ── Show results ──
if st.session_state.last_results:
    st.markdown("---")
    st.markdown("### 📑 Top Retrieved Papers")
    for i, paper in enumerate(st.session_state.last_results, 1):
        st.markdown(_paper_card_html(i, paper), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        """
        <div class="answer-header">
            🦙 &nbsp;LLaMA 3.1 8B &nbsp;·&nbsp; Synthesized Answer
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="answer-box">{st.session_state.last_answer}</div>',
        unsafe_allow_html=True,
    )

elif st.session_state.index_status == "ready":
    # Empty state hint
    st.markdown(
        """
        <div style="text-align:center;padding:3rem 1rem;color:#475569;">
            <div style="font-size:3rem;margin-bottom:1rem;">🔍</div>
            <div style="font-size:1.05rem;font-weight:500;color:#64748b;">
                Type a question above and click <b style="color:#818cf8;">Search &amp; Ask</b>
            </div>
            <div style="font-size:0.85rem;margin-top:0.5rem;">
                The top 3 most relevant papers will appear here, followed by a synthesized answer.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Not-built state
    st.markdown(
        """
        <div style="text-align:center;padding:3rem 1rem;color:#475569;">
            <div style="font-size:3rem;margin-bottom:1rem;">⚡</div>
            <div style="font-size:1.05rem;font-weight:500;color:#64748b;">
                Click <b style="color:#818cf8;">Build Index</b> in the sidebar to get started
            </div>
            <div style="font-size:0.85rem;margin-top:0.5rem;">
                This will fetch 2 000 ArXiv papers and embed them into Endee.<br>
                Takes ~5–10 minutes on first run; cached on subsequent runs.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
