import os
from src.inverted_index import build_inverted_index
from src.tfidf_model import build_tfidf
from src.search import search
from src.evaluation import precision_recall


DATA_DIR = "data"


def load_documents():
    documents = []
    filenames = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".txt"):
            filenames.append(file)
            with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
                documents.append(f.read())

    return documents, filenames


if __name__ == "__main__":
    # Load documents
    documents, files = load_documents()   # ← files is defined HERE

    # Build inverted index
    index = build_inverted_index(DATA_DIR)
    print("\nInverted Index:")
    print(index)

    # Build TF-IDF model
    vectorizer, tfidf_matrix = build_tfidf(documents)

    # Query
    query = "machine learning"
    results = search(query, vectorizer, tfidf_matrix, files)  # ← NO ERROR

    print("\nSearch Results:")
    for doc, score in results:
        print(f"{doc} -> {score:.4f}")

    # Evaluation
    relevant_docs = ["doc1.txt", "doc3.txt"]
    retrieved_docs = [doc for doc, score in results if score > 0]

    precision, recall = precision_recall(retrieved_docs, relevant_docs)

    print(f"\nPrecision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
