import streamlit as st
import os
import time
import pandas as pd
from pathlib import Path

from src.document_loader import (
    load_academic_documents,
    save_uploaded_document,
    DEFAULT_SUBJECTS,
    DEFAULT_TYPES
)
from src.inverted_index import build_inverted_index_from_docs
from src.tfidf_model import build_tfidf
from src.search import search_academic_documents
from src.evaluation import run_benchmark_evaluation, evaluate_retrieval, BENCHMARK_QUERIES
from src.spell_correction import correct_query_spelling

DATA_DIR = "data"

# Page configuration
st.set_page_config(
    page_title="AcademiaIR — College Document Retrieval System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# PROFESSIONAL MODERN WEBSITE CSS (CLEAN LIGHT THEME, MODERN CARDS, NO CLUTTER)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0F172A;
    }

    /* Main background */
    .stApp {
        background-color: #F8FAFC;
    }

    /* Hide standard Streamlit header and footer elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar Navigation Container */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] div, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] p {
        color: #E2E8F0 !important;
    }

    /* Sidebar Brand Section */
    .sidebar-brand-box {
        padding: 8px 4px 18px 4px;
        border-bottom: 1px solid #1E293B;
        margin-bottom: 20px;
    }
    .sidebar-brand-logo {
        font-size: 1.8rem;
        margin-right: 8px;
    }
    .sidebar-brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #FFFFFF !important;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .sidebar-brand-sub {
        font-size: 0.72rem;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Styled Radio Navigation in Sidebar */
    div[data-testid="stRadio"] > div {
        gap: 6px;
    }
    div[data-testid="stRadio"] label {
        background: transparent;
        border-radius: 8px;
        padding: 10px 14px;
        transition: all 0.15s ease-in-out;
        border: 1px solid transparent;
        cursor: pointer;
    }
    div[data-testid="stRadio"] label:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(255, 255, 255, 0.1);
    }
    div[data-testid="stRadio"] label[data-checked="true"],
    div[data-testid="stRadio"] label:has(input:checked) {
        background: #1E293B !important;
        border-color: #3B82F6 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
    }

    /* Sidebar Status Card */
    .sidebar-status-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px;
        margin-top: 24px;
    }
    .status-stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 0;
        font-size: 0.8rem;
    }
    .status-stat-val {
        font-weight: 700;
        color: #38BDF8 !important;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Top Page Header */
    .page-title-banner {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .page-title-text {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .page-desc-text {
        font-size: 0.9rem;
        color: #64748B;
        margin-top: 4px;
        margin-bottom: 0;
    }

    /* Modern Document Card */
    .doc-result-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease-in-out;
    }
    .doc-result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -3px rgba(15, 23, 42, 0.08), 0 4px 6px -4px rgba(15, 23, 42, 0.04);
        border-color: #93C5FD;
    }
    .doc-top-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 16px;
    }
    .rank-badge {
        font-size: 1rem;
        font-weight: 800;
        color: #2563EB;
        background: #EFF6FF;
        border: 1px solid #DBEAFE;
        border-radius: 8px;
        padding: 4px 10px;
        display: inline-block;
    }
    .doc-title-heading {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        margin-left: 10px;
    }
    .doc-filename-meta {
        font-size: 0.82rem;
        color: #64748B;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 3px;
    }

    /* Relevance badges */
    .score-pill-high {
        background: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.88rem;
    }
    .score-pill-mid {
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.88rem;
    }
    .score-pill-low {
        background: #F8FAFC;
        color: #475569;
        border: 1px solid #E2E8F0;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.88rem;
    }

    /* Tag Pills */
    .meta-tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
        margin-right: 6px;
        margin-top: 8px;
    }
    .tag-notes { background: #DBEAFE; color: #1E40AF; }
    .tag-qp { background: #FEF3C7; color: #92400E; }
    .tag-assign { background: #DCFCE7; color: #166534; }
    .tag-syllab { background: #F3E8FF; color: #6B21A8; }
    .tag-other { background: #F1F5F9; color: #334155; }
    .tag-subject { background: #F1F5F9; color: #0F172A; border: 1px solid #E2E8F0; }
    .tag-semester { background: #F1F5F9; color: #475569; border: 1px solid #E2E8F0; }

    /* Highlight Snippet Container */
    .snippet-highlight-box {
        background: #F8FAFC;
        border-left: 4px solid #3B82F6;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-top: 14px;
        color: #334155;
        font-size: 0.94rem;
        line-height: 1.6;
    }
    .query-highlight {
        background-color: #FEF08A;
        color: #854D0E;
        font-weight: 700;
        padding: 2px 5px;
        border-radius: 4px;
    }

    /* KPI Metric Cards */
    .kpi-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .kpi-num {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin: 4px 0;
    }
    .kpi-footnote {
        font-size: 0.78rem;
        font-weight: 600;
        color: #059669;
    }

    /* Card Box for Forms & Panels */
    .panel-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# APPLICATION STATE & RETRIEVAL ENGINE
# -----------------------------------------------------------------------------
def get_corpus_and_model():
    if "documents" not in st.session_state or "vectorizer" not in st.session_state:
        reload_index()
    return st.session_state.documents, st.session_state.vectorizer, st.session_state.tfidf_matrix, st.session_state.inverted_index


def reload_index():
    docs = load_academic_documents(DATA_DIR)
    st.session_state.documents = docs
    if docs:
        corpus = [doc["content"] for doc in docs]
        vec, matrix = build_tfidf(corpus)
        inv_idx = build_inverted_index_from_docs(docs)
        st.session_state.vectorizer = vec
        st.session_state.tfidf_matrix = matrix
        st.session_state.inverted_index = inv_idx
    else:
        st.session_state.vectorizer = None
        st.session_state.tfidf_matrix = None
        st.session_state.inverted_index = {}


docs, vectorizer, tfidf_matrix, inverted_index = get_corpus_and_model()


# -----------------------------------------------------------------------------
# SIDEBAR AS PRIMARY WEBSITE NAVIGATION BAR
# -----------------------------------------------------------------------------
with st.sidebar:
    # Portal Brand Header
    st.markdown("""
    <div class="sidebar-brand-box">
        <div style="display: flex; align-items: center;">
            <span class="sidebar-brand-logo">🏛️</span>
            <div>
                <div class="sidebar-brand-title">AcademiaIR</div>
                <div class="sidebar-brand-sub">Academic IR Portal</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("PORTAL NAVIGATION")

    # Primary Navigation Menu
    nav_selection = st.radio(
        label="Navigation Menu",
        options=[
            "🔍  Search Documents",
            "📤  Upload & Index PDF",
            "🗂️  Inverted Index",
            "📈  Evaluation Benchmark",
            "📚  Document Repository",
            "💡  Viva & IR Architecture"
        ],
        index=0,
        label_visibility="collapsed"
    )

    # Clean the selection string to route
    current_page = nav_selection.split("  ")[1].strip()

    st.markdown("---")

    # System Status & Health Card
    vocab_size = len(vectorizer.get_feature_names_out()) if vectorizer is not None else 0
    st.markdown(f"""
    <div class="sidebar-status-card">
        <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; margin-bottom: 8px;">
            SYSTEM INDEX STATUS
        </div>
        <div class="status-stat-row">
            <span>Indexed Documents</span>
            <span class="status-stat-val">{len(docs)}</span>
        </div>
        <div class="status-stat-row">
            <span>Vocabulary Size</span>
            <span class="status-stat-val">{vocab_size}</span>
        </div>
        <div class="status-stat-row">
            <span>IR Engine</span>
            <span class="status-stat-val">TF-IDF + VSM</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    if st.button("🔄  Rebuild Corpus Index", use_container_width=True):
        with st.spinner("Rebuilding indexes..."):
            reload_index()
        st.success("Index synchronized!")
        st.rerun()

    st.caption("College IR Subject Project • Vector Space Model")


# =============================================================================
# PAGE 1: 🔍 SEARCH DOCUMENTS
# =============================================================================
if current_page == "Search Documents":
    # Top Page Header
    st.markdown("""
    <div class="page-title-banner">
        <div>
            <h1 class="page-title-text">Academic Document Search Engine</h1>
            <p class="page-desc-text">Retrieve relevant lecture notes, exam papers, syllabi, and study material using TF-IDF & Cosine Similarity ranking.</p>
        </div>
        <div style="display: flex; gap: 8px;">
            <span style="background: #EFF6FF; color: #1D4ED8; font-size: 0.78rem; font-weight: 700; padding: 6px 12px; border-radius: 9999px; border: 1px solid #BFDBFE;">
                Vector Space Model Active
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Search Bar Box
    search_col, btn_col = st.columns([5, 1])
    with search_col:
        query_text = st.text_input(
            "Search Query",
            placeholder="Search topics, concepts, theorems, algorithms... (e.g., central limit theorem)",
            label_visibility="collapsed",
            key="main_search_bar"
        )
    with btn_col:
        search_clicked = st.button("Search", use_container_width=True, type="primary")

    def set_search_query(text):
        st.session_state["main_search_bar"] = text

    # Quick Suggestion Chips
    st.caption("Quick test queries:")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.button("📌 Central Limit Theorem", use_container_width=True, on_click=set_search_query, args=("central limit theorem",))
    with c2:
        st.button("📌 Inverted Index TF-IDF", use_container_width=True, on_click=set_search_query, args=("inverted index tf idf",))
    with c3:
        st.button("📌 Relational Schema 3NF", use_container_width=True, on_click=set_search_query, args=("relational schema normalization 3nf",))
    with c4:
        st.button("📌 Linear Regression Gradient", use_container_width=True, on_click=set_search_query, args=("linear regression gradient descent",))

    st.write("")

    # Filter Toolbar Card
    with st.expander("Filter Options & Relevance Threshold", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        available_types = ["All Types"] + sorted(list({d.get("doc_type", "Other") for d in docs}))
        available_subjects = ["All Subjects"] + sorted(list({d.get("subject", "General") for d in docs}))
        available_sems = ["All Semesters"] + sorted(list({d.get("semester", "General") for d in docs}))

        with fc1:
            sel_type = st.selectbox("Document Type", available_types)
        with fc2:
            sel_subject = st.selectbox("Subject", available_subjects)
        with fc3:
            sel_sem = st.selectbox("Semester", available_sems)
        with fc4:
            min_score = st.slider("Min. Relevance Score", 0.0, 0.5, 0.02, 0.01)

    filters = {
        "doc_type": sel_type,
        "subject": sel_subject,
        "semester": sel_sem
    }

    # Execute Search
    query = st.session_state.get("main_search_bar", query_text).strip()

    if query:
        if vectorizer is None or tfidf_matrix is None or len(docs) == 0:
            st.warning("No documents available in the corpus index.")
        else:
            # Check vocabulary for spelling correction / query expansion
            vocab_set = set(vectorizer.get_feature_names_out())
            corrected_query, was_corrected, corrections = correct_query_spelling(query, vocab_set)

            # Retrieve with corrected query
            active_search_query = corrected_query if was_corrected else query

            start_time = time.time()
            results = search_academic_documents(
                query=active_search_query,
                vectorizer=vectorizer,
                tfidf_matrix=tfidf_matrix,
                documents=docs,
                filters=filters,
                min_score=min_score
            )
            latency_ms = (time.time() - start_time) * 1000

            st.markdown("---")

            # Google-style Did You Mean / Auto-Correct Banner
            if was_corrected:
                st.markdown(f"""
                <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 12px 18px; margin-bottom: 18px;">
                    <span style="color: #1D4ED8; font-weight: 700; font-size: 0.95rem;">💡 Auto-Corrected Query:</span> 
                    <span style="color: #0F172A; font-weight: 800; font-size: 1.05rem; margin-left: 6px;">"{corrected_query}"</span>
                    <span style="color: #64748B; font-size: 0.85rem; margin-left: 10px;">(Original search: <em>"{query}"</em>)</span>
                </div>
                """, unsafe_allow_html=True)

            # Result count and latency row
            rh_col1, rh_col2 = st.columns([4, 2])
            with rh_col1:
                st.markdown(f"### Results for *'{active_search_query}'* ({len(results)} found)")
            with rh_col2:
                st.markdown(f"<div style='text-align: right; color: #64748B; font-size: 0.85rem; padding-top: 8px;'>Retrieved in <strong>{latency_ms:.2f} ms</strong> • Cosine Ranking</div>", unsafe_allow_html=True)

            if results:
                for res in results:
                    score = res["score"]
                    if score >= 0.35:
                        score_class = "score-pill-high"
                        score_label = f"High Relevance • {score:.4f}"
                    elif score >= 0.15:
                        score_class = "score-pill-mid"
                        score_label = f"Moderate • {score:.4f}"
                    else:
                        score_class = "score-pill-low"
                        score_label = f"Relevance • {score:.4f}"

                    dt = res["doc_type"].lower()
                    if "note" in dt:
                        tag_class = "tag-notes"
                        type_icon = "📄"
                    elif "question" in dt or "paper" in dt:
                        tag_class = "tag-qp"
                        type_icon = "📝"
                    elif "assign" in dt:
                        tag_class = "tag-assign"
                        type_icon = "📑"
                    elif "syllab" in dt:
                        tag_class = "tag-syllab"
                        type_icon = "📋"
                    else:
                        tag_class = "tag-other"
                        type_icon = "📁"

                    # Document Result Card
                    st.markdown(f"""
                    <div class="doc-result-card">
                        <div class="doc-top-row">
                            <div style="display: flex; align-items: center;">
                                <span class="rank-badge">#{res['rank']}</span>
                                <div>
                                    <span class="doc-title-heading">{res['title']}</span>
                                    <div class="doc-filename-meta">{res['filename']} • Format: {res.get('extension', '.pdf').upper()}</div>
                                </div>
                            </div>
                            <div>
                                <span class="{score_class}">{score_label}</span>
                            </div>
                        </div>
                        <div style="margin-top: 10px;">
                            <span class="meta-tag {tag_class}">{type_icon} {res['doc_type']}</span>
                            <span class="meta-tag tag-subject">📘 {res['subject']}</span>
                            <span class="meta-tag tag-semester">🎓 {res['semester']}</span>
                        </div>
                        <div class="snippet-highlight-box">
                            {res.get('html_snippet', res['snippet'])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander(f"📄 Read Full Extracted Text — {res['filename']}"):
                        st.text_area(
                            label="Document Text Content",
                            value=res["full_content"],
                            height=180,
                            key=f"text_area_{res['id']}_{res['rank']}"
                        )
            else:
                st.warning("No relevant documents found matching your query and filter criteria. Try using different keywords or lowering the minimum relevance score threshold.")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; color: #64748B;">
            <span style="font-size: 3.5rem;">🔍</span>
            <h3 style="color: #0F172A; margin-top: 16px;">Ready to Search</h3>
            <p style="max-width: 520px; margin: 0 auto; font-size: 0.95rem;">
                Type an academic topic above, or pick one of the suggested query chips to experience real-time TF-IDF Vector Space document retrieval.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2: 📤 UPLOAD & INDEX PDF
# =============================================================================
elif current_page == "Upload & Index PDF":
    st.markdown("""
    <div class="page-title-banner">
        <div>
            <h1 class="page-title-text">Document Ingestion & Indexing Portal</h1>
            <p class="page-desc-text">Upload college lecture notes, question papers, or syllabi (PDF or TXT) to instantly extract text and update the TF-IDF search index.</p>
        </div>
        <div>
            <span style="background: #F0FDF4; color: #166534; font-size: 0.78rem; font-weight: 700; padding: 6px 12px; border-radius: 9999px; border: 1px solid #BBF7D0;">
                Live PyPDF Text Extractor
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_info = st.columns([3, 2])

    with col_form:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Upload Academic Document")
        st.caption("Upload your file and attach academic metadata for search filtering.")

        with st.form("dedicated_upload_form", clear_on_submit=True):
            uploaded_file = st.file_uploader("Select Academic Document (PDF or TXT)", type=["pdf", "txt"])
            doc_title = st.text_input("Document Title (optional)", placeholder="e.g. Unit 4 Distributed Systems Notes")
            doc_type = st.selectbox("Document Type", DEFAULT_TYPES)
            doc_subject = st.selectbox("Subject", DEFAULT_SUBJECTS)
            doc_semester = st.selectbox("Semester", ["Semester 5", "Semester 6", "Semester 7", "Semester 8", "General"])

            submit_upload = st.form_submit_button("Index & Add Document", use_container_width=True, type="primary")

            if submit_upload:
                if uploaded_file is not None:
                    with st.spinner("Extracting text and updating TF-IDF vocabulary..."):
                        new_doc = save_uploaded_document(
                            uploaded_file=uploaded_file,
                            doc_type=doc_type,
                            subject=doc_subject,
                            semester=doc_semester,
                            title=doc_title
                        )
                        reload_index()
                    st.success(f"✓ Document **'{new_doc['filename']}'** was parsed, indexed, and is now searchable!")
                    st.session_state.last_uploaded = new_doc
                    st.rerun()
                else:
                    st.error("Please choose a file to upload.")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("How Text Extraction Works")
        st.markdown("""
        1. **Binary Stream Extraction:** `pypdf.PdfReader` extracts page-by-page text streams.
        2. **Preprocessing Pipeline:** Case normalization, non-alphabet stripping, and NLTK stop-word removal.
        3. **Inverted Index Update:** Postings lists automatically map new terms to this document ID.
        4. **Vector Model Re-fit:** Scikit-learn refits the $N$-document TF-IDF matrix in real-time.
        """)

        # If a document was recently uploaded, show its preview
        if "last_uploaded" in st.session_state:
            last = st.session_state.last_uploaded
            st.divider()
            st.markdown(f"**Recently Indexed:** `{last['filename']}`")
            st.caption(f"Title: {last['title']} • Type: {last['doc_type']} • Subject: {last['subject']}")
            with st.expander("Preview Extracted Content"):
                st.text(last["content"][:400] + ("..." if len(last["content"]) > 400 else ""))

        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE 3: 🗂️ INVERTED INDEX
# =============================================================================
elif current_page == "Inverted Index":
    st.markdown("""
    <div class="page-title-banner">
        <div>
            <h1 class="page-title-text">Inverted Index Data Structure Visualizer</h1>
            <p class="page-desc-text">Inspect the core dictionary-to-postings list mapping that powers fast Information Retrieval candidate lookup.</p>
        </div>
        <div>
            <span style="background: #FAF5FF; color: #6B21A8; font-size: 0.78rem; font-weight: 700; padding: 6px 12px; border-radius: 9999px; border: 1px solid #E9D5FF;">
                Postings Lists Explorer
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    lookup_col1, lookup_col2 = st.columns([4, 1])
    with lookup_col1:
        term_search = st.text_input("Look up term in vocabulary index:", placeholder="e.g. theorem, regression, normalization, retrieval")
    with lookup_col2:
        st.write("")
        st.write("")
        btn_lookup = st.button("Lookup Term", use_container_width=True)

    if inverted_index:
        if term_search:
            term_clean = term_search.strip().lower()
            if term_clean in inverted_index:
                postings = inverted_index[term_clean]
                st.success(f"✓ Found term **'{term_clean}'** — Appears in **{len(postings)}** document(s) (Document Frequency = {len(postings)})")

                st.markdown(f"#### Postings List for `{term_clean}`:")
                for doc_id in postings:
                    doc_obj = next((d for d in docs if d["filename"] == doc_id), None)
                    subj = doc_obj["subject"] if doc_obj else "General"
                    dtype = doc_obj["doc_type"] if doc_obj else "Document"
                    st.markdown(f"- 📄 **{doc_id}** — *[{dtype} | {subj}]*")
            else:
                st.warning(f"Term **'{term_clean}'** not found in the indexed vocabulary.")

        st.markdown("---")
        st.subheader("Top Vocabulary Terms by Document Frequency (DF)")
        st.caption("Shows how widely distributed specific academic terms are across the collection.")

        term_freqs = [(term, len(postings)) for term, postings in inverted_index.items()]
        term_freqs.sort(key=lambda x: x[1], reverse=True)
        top_df = pd.DataFrame(term_freqs[:16], columns=["Vocabulary Term", "Document Frequency"])

        st.bar_chart(top_df.set_index("Vocabulary Term"), use_container_width=True)

        with st.expander("Browse Complete Inverted Index Dictionary (JSON)"):
            st.json(dict(list(inverted_index.items())[:60]))


# =============================================================================
# PAGE 4: 📈 EVALUATION BENCHMARK
# =============================================================================
elif current_page == "Evaluation Benchmark":
    st.markdown("""
    <div class="page-title-banner">
        <div>
            <h1 class="page-title-text">Information Retrieval Evaluation Dashboard</h1>
            <p class="page-desc-text">Evaluate retrieval precision, recall, and harmonic F1-score against predefined academic ground-truth relevance sets.</p>
        </div>
        <div>
            <span style="background: #F0FDF4; color: #166534; font-size: 0.78rem; font-weight: 700; padding: 6px 12px; border-radius: 9999px; border: 1px solid #BBF7D0;">
                Ground-Truth Validated
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if vectorizer is not None and tfidf_matrix is not None and len(docs) > 0:
        summary = run_benchmark_evaluation(
            search_fn=search_academic_documents,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            documents=docs
        )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">Mean Precision</div>
                <div class="kpi-num">{summary['mean_precision']:.1%}</div>
                <div class="kpi-footnote">Proportion Relevant Retrieved</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">Mean Recall</div>
                <div class="kpi-num">{summary['mean_recall']:.1%}</div>
                <div class="kpi-footnote">Coverage of Relevant Docs</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">Mean F1-Score</div>
                <div class="kpi-num">{summary['mean_f1']:.1%}</div>
                <div class="kpi-footnote">Harmonic Mean Metric</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">Benchmark Queries</div>
                <div class="kpi-num">{len(BENCHMARK_QUERIES)}</div>
                <div class="kpi-footnote">Curated Academic Tests</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Detailed Query Benchmark Performance")

        b_rows = []
        for item in summary["query_results"]:
            b_rows.append({
                "Test Query": item["query"],
                "Scope / Concept": item["description"],
                "Retrieved": item["retrieved_count"],
                "Relevant (Truth)": item["relevant_count"],
                "True Positives": item["true_positives"],
                "Precision": f"{item['precision']:.4f}",
                "Recall": f"{item['recall']:.4f}",
                "F1-Score": f"{item['f1']:.4f}"
            })
        st.dataframe(pd.DataFrame(b_rows), use_container_width=True)

        st.markdown("---")
        st.subheader("Interactive Custom Query Evaluator")
        st.caption("Test any query against your own chosen relevant document ground-truth.")

        eval_q = st.text_input("Evaluation Query:", "central limit theorem", key="eval_custom_box")
        all_doc_names = [d["filename"] for d in docs]
        eval_truth = st.multiselect(
            "Select Ground Truth Relevant Documents:",
            options=all_doc_names,
            default=[d for d in all_doc_names if "DVM" in d or "Statistics" in d][:3]
        )

        if st.button("Calculate Metrics for Custom Query", type="primary"):
            res_custom = search_academic_documents(eval_q, vectorizer, tfidf_matrix, docs, min_score=0.05)
            ret_ids = [r["filename"] for r in res_custom]
            m = evaluate_retrieval(ret_ids, eval_truth)

            cp, cr, cf1 = st.columns(3)
            cp.metric("Precision", f"{m['precision']:.4f}")
            cr.metric("Recall", f"{m['recall']:.4f}")
            cf1.metric("F1-Score", f"{m['f1']:.4f}")


# =============================================================================
# PAGE 5: 📚 DOCUMENT REPOSITORY
# =============================================================================
elif current_page == "Document Repository":
    st.markdown(f"""
    <div class="page-title-banner">
        <div>
            <h1 class="page-title-text">Academic Document Repository</h1>
            <p class="page-desc-text">Catalogue of all academic files indexed in the current Vector Space corpus.</p>
        </div>
        <div>
            <span style="background: #EFF6FF; color: #1D4ED8; font-size: 0.78rem; font-weight: 700; padding: 6px 12px; border-radius: 9999px; border: 1px solid #BFDBFE;">
                Total Documents: {len(docs)}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    table_data = []
    for d in docs:
        table_data.append({
            "Filename": d["filename"],
            "Document Title": d["title"],
            "Type": d["doc_type"],
            "Subject": d["subject"],
            "Semester": d["semester"],
            "Character Count": len(d["content"]),
            "Format": d.get("extension", ".pdf").upper()
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True)


# =============================================================================
# PAGE 6: 💡 VIVA & IR ARCHITECTURE
# =============================================================================
elif current_page == "Viva & IR Architecture":
    st.markdown("""
    <div class="page-title-banner">
        <div>
            <h1 class="page-title-text">IR Architecture & Viva Presentation Guide</h1>
            <p class="page-desc-text">Technical documentation and viva answers explaining the retrieval pipeline and algorithms.</p>
        </div>
        <div>
            <span style="background: #FEF3C7; color: #92400E; font-size: 0.78rem; font-weight: 700; padding: 6px 12px; border-radius: 9999px; border: 1px solid #FDE68A;">
                Viva Reference Guide
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 1. Classical Information Retrieval Pipeline
    ```text
    User Query (e.g. "central limit theorem")
       ↓
    Text Preprocessing (Lowercasing, Regex Filtering, Tokenization, NLTK Stop-word Removal)
       ↓
    TF-IDF Vector Space Transformation (Scikit-Learn TfidfVectorizer)
       ↓
    Cosine Similarity Computation (Query Vector · Document Vectors / Norms)
       ↓
    Candidate Ranking & Filtering (Descending Score Sorting + Type/Subject Filters)
       ↓
    Context Snippet Extraction & Query Term Highlighting
       ↓
    Top-K Ranked Academic Results
    ```
    """)

    st.markdown("### 2. Core Mathematical Formulations")
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("""
        #### Term Frequency – Inverse Document Frequency (TF-IDF)
        $$\\text{TF-IDF}(t, d) = \\text{TF}(t, d) \\times \\log\\left(\\frac{N}{\\text{DF}(t)}\\right)$$
        - **$\\text{TF}(t, d)$:** Frequency of term $t$ in document $d$ (local importance).
        - **$\\text{IDF}(t)$:** Penalizes universally common words across the $N$ documents.
        """)
    with v2:
        st.markdown("""
        #### Cosine Similarity Ranking
        $$\\text{Cosine Similarity}(\\vec{q}, \\vec{d}) = \\frac{\\vec{q} \\cdot \\vec{d}}{\\|\\vec{q}\\| \\|\\vec{d}\\|}$$
        - Calculates the cosine of the angle between query and document vectors.
        - Inherently normalizes for document length variations.
        """)

    st.markdown("---")
    st.markdown("### 3. Top Viva Questions & Model Answers")
    with st.expander("Q1: Why is Cosine Similarity preferred over Euclidean Distance in text retrieval?"):
        st.write("""
        **Answer:** Euclidean distance is heavily distorted by document length. A long document containing the same word distribution as a short document will have a large Euclidean distance from the query simply because of word count. Cosine similarity evaluates the angle between vectors, which normalizes for length and measures pure thematic alignment.
        """)

    with st.expander("Q2: What is an Inverted Index and why is it essential?"):
        st.write("""
        **Answer:** An Inverted Index maps each unique vocabulary term to a postings list of documents where it appears. Without an inverted index, searching requires linearly scanning every document in the collection ($O(N \\times L)$). With an inverted index, candidate documents can be retrieved in $O(1)$ or $O(\\log |V|)$ time.
        """)

    with st.expander("Q3: Why is F1-score more meaningful than Precision or Recall alone?"):
        st.write("""
        **Answer:** A system can achieve 100% Precision trivially by returning only the single most confident document, or 100% Recall by returning every document in the corpus. F1-score is the harmonic mean of Precision and Recall, which penalizes unbalanced systems and provides a single balanced metric.
        """)
