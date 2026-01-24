from src.tfidf_model import build_tfidf
from src.search import search

def test_search_results():
    docs = [
        "machine learning is powerful",
        "python for data science"
    ]
    files = ["doc1.txt", "doc2.txt"]

    vectorizer, matrix = build_tfidf(docs)
    results = search("machine learning", vectorizer, matrix, files)

    assert results[0][0] == "doc1.txt"
