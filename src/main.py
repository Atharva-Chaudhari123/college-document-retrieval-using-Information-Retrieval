import os
from src.document_loader import load_academic_documents
from src.inverted_index import build_inverted_index
from src.tfidf_model import build_tfidf
from src.search import search, search_academic_documents
from src.evaluation import precision_recall, calculate_f1, evaluate_retrieval

DATA_DIR = "data"


def main():
    print("=" * 70)
    print("   COLLEGE ACADEMIC DOCUMENT INFORMATION RETRIEVAL SYSTEM")
    print("   Information Retrieval (IR) Subject Project")
    print("=" * 70)

    # 1. Load Academic Documents
    print("\n[Step 1] Loading academic document collection...")
    documents = load_academic_documents(DATA_DIR)
    print(f"Loaded {len(documents)} academic documents from '{DATA_DIR}' directory.")
    for idx, d in enumerate(documents[:5], 1):
        print(f"  {idx}. {d['filename']} [{d['doc_type']}] - {d['subject']} ({d['semester']})")
    if len(documents) > 5:
        print(f"  ... and {len(documents) - 5} more documents.")

    # 2. Build Inverted Index
    print("\n[Step 2] Constructing Inverted Index...")
    index = build_inverted_index(DATA_DIR)
    sample_terms = ["central", "limit", "theorem", "retrieval", "machine", "normalization"]
    print("Sample Inverted Index Entries:")
    for term in sample_terms:
        if term in index:
            print(f"  '{term}' -> {index[term]}")

    # 3. Build TF-IDF Model
    print("\n[Step 3] Building TF-IDF Vector Space Model...")
    corpus = [doc["content"] for doc in documents]
    vectorizer, tfidf_matrix = build_tfidf(corpus)
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape} (Documents x Vocabulary Terms)")

    # 4. Search Query Demo
    query = "central limit theorem"
    print(f"\n[Step 4] Executing Query: '{query}'")
    results = search_academic_documents(query, vectorizer, tfidf_matrix, documents)

    print("\n--- Top Ranked Results ---")
    for res in results[:5]:
        print(f"Rank #{res['rank']} | Score: {res['score']:.4f}")
        print(f"Document : {res['filename']} ({res['title']})")
        print(f"Metadata : Type: {res['doc_type']} | Subject: {res['subject']} | {res['semester']}")
        print(f"Snippet  : {res['snippet']}")
        print("-" * 50)

    # 5. IR Evaluation Demo
    print("\n[Step 5] Information Retrieval Evaluation Demo")
    # Ground truth documents expected for query "central limit theorem"
    relevant_docs = [
        "DVM_Unit_1_Notes.pdf",
        "Statistics_Notes.pdf",
        "DVM_Previous_Year_Question_Paper.pdf",
        "Data_Visualization_Syllabus.pdf"
    ]
    retrieved_docs = [res["filename"] for res in results if res["score"] > 0.05]

    eval_stats = evaluate_retrieval(retrieved_docs, relevant_docs)
    print(f"Query: '{query}'")
    print(f"Retrieved Documents: {len(retrieved_docs)}")
    print(f"Relevant Documents (Ground Truth): {len(relevant_docs)}")
    print(f"True Positives: {eval_stats['true_positives']}")
    print(f"Precision: {eval_stats['precision']:.4f}")
    print(f"Recall:    {eval_stats['recall']:.4f}")
    print(f"F1-Score:  {eval_stats['f1']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
