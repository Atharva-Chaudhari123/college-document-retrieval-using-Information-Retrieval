import pytest
from src.document_loader import load_academic_documents, extract_text_from_txt
from src.tfidf_model import build_tfidf
from src.search import search_academic_documents, extract_snippet, highlight_terms
from src.evaluation import precision_recall, calculate_f1, evaluate_retrieval, run_benchmark_evaluation
from src.inverted_index import build_inverted_index


def test_load_academic_documents():
    docs = load_academic_documents("data")
    assert len(docs) >= 12
    # Check that both pdf and txt docs exist
    extensions = {d["extension"] for d in docs}
    assert ".pdf" in extensions
    assert ".txt" in extensions

    # Verify metadata fields
    first = docs[0]
    assert "filename" in first
    assert "doc_type" in first
    assert "subject" in first
    assert "content" in first
    assert len(first["content"]) > 10


def test_inverted_index_academic_terms():
    index = build_inverted_index("data")
    # CLT terms should exist in inverted index
    assert "central" in index
    assert "theorem" in index
    assert "normalization" in index
    assert any("DVM" in doc for doc in index["central"])


def test_search_academic_documents_clt():
    docs = load_academic_documents("data")
    corpus = [d["content"] for d in docs]
    vectorizer, matrix = build_tfidf(corpus)

    results = search_academic_documents("central limit theorem", vectorizer, matrix, docs)
    assert len(results) > 0

    top_filenames = [r["filename"] for r in results[:3]]
    # Either DVM or Statistics notes should be top ranked
    assert any("DVM" in f or "Statistics" in f for f in top_filenames)
    assert results[0]["score"] > 0.1
    assert "**" in results[0]["snippet"]  # Highlighted query terms


def test_metadata_filtering():
    docs = load_academic_documents("data")
    corpus = [d["content"] for d in docs]
    vectorizer, matrix = build_tfidf(corpus)

    # Filter by Document Type = 'Question Paper'
    qp_results = search_academic_documents(
        "central limit theorem",
        vectorizer,
        matrix,
        docs,
        filters={"doc_type": "Question Paper", "subject": "All Subjects", "semester": "All Semesters"}
    )
    for r in qp_results:
        assert r["doc_type"] == "Question Paper"

    # Filter by Subject = 'Data Visualization'
    dvm_results = search_academic_documents(
        "central limit theorem",
        vectorizer,
        matrix,
        docs,
        filters={"doc_type": "All Types", "subject": "Data Visualization", "semester": "All Semesters"}
    )
    for r in dvm_results:
        assert r["subject"] == "Data Visualization"


def test_snippet_and_highlighting():
    text = "The Central Limit Theorem is fundamental in inferential statistics and hypothesis testing."
    snippet = extract_snippet(text, "central limit theorem")
    assert "**Central Limit Theorem**" in snippet or ("**Central**" in snippet and "**Theorem**" in snippet)

    highlighted = highlight_terms("Machine learning uses neural networks.", ["machine", "learning"])
    assert "**Machine**" in highlighted
    assert "**learning**" in highlighted


def test_evaluation_metrics():
    retrieved = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    relevant = ["doc1.pdf", "doc2.pdf", "doc4.pdf"]

    p, r = precision_recall(retrieved, relevant)
    assert p == 2 / 3
    assert r == 2 / 3

    f1 = calculate_f1(p, r)
    assert abs(f1 - (2 / 3)) < 1e-4

    zero_f1 = calculate_f1(0, 0)
    assert zero_f1 == 0.0

    eval_dict = evaluate_retrieval(retrieved, relevant)
    assert eval_dict["true_positives"] == 2
    assert eval_dict["precision"] == 2 / 3


def test_spell_correction():
    from src.spell_correction import correct_query_spelling

    vocab = {"regression", "descent", "linear", "machine", "learning"}
    corrected, was_corr, details = correct_query_spelling("regresion decent", vocab)
    assert was_corr is True
    assert corrected == "regression descent"

    # Acronym expansion
    acronym_query, acr_corr, _ = correct_query_spelling("clt", vocab)
    assert acr_corr is True
    assert "central limit theorem" in acronym_query

