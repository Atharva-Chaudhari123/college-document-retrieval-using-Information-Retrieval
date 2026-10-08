def precision_recall(retrieved, relevant):
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)

    true_positives = retrieved_set & relevant_set

    precision = len(true_positives) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(true_positives) / len(relevant_set) if relevant_set else 0.0

    return precision, recall


def calculate_f1(precision, recall):
    """Calculates F1-score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def evaluate_retrieval(retrieved, relevant):
    """
    Computes Precision, Recall, and F1-score for retrieved documents against relevant documents.
    """
    precision, recall = precision_recall(retrieved, relevant)
    f1 = calculate_f1(precision, recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "retrieved_count": len(retrieved),
        "relevant_count": len(relevant),
        "true_positives": len(set(retrieved) & set(relevant))
    }


# Predefined evaluation benchmark for academic IR queries
BENCHMARK_QUERIES = [
    {
        "query": "central limit theorem",
        "description": "Foundational statistics limit theorem query",
        "relevant": [
            "DVM_Unit_1_Notes.pdf",
            "Statistics_Notes.pdf",
            "DVM_Previous_Year_Question_Paper.pdf",
            "Data_Visualization_Syllabus.pdf"
        ]
    },
    {
        "query": "inverted index tf idf",
        "description": "Core Information Retrieval concepts",
        "relevant": [
            "IR_Lecture_Notes_TFIDF.pdf",
            "IR_Assignment_1_Vector_Space.pdf",
            "IR_EndSem_Question_Paper_2025.pdf",
            "Information_Retrieval_Syllabus.pdf"
        ]
    },
    {
        "query": "relational schema normalization 3nf",
        "description": "DBMS Normalization theory",
        "relevant": [
            "DBMS_Unit_3_Normalization.pdf",
            "DBMS_Assignment_2_SQL_Queries.pdf"
        ]
    },
    {
        "query": "linear regression gradient descent",
        "description": "Supervised machine learning algorithms",
        "relevant": [
            "ML_Unit_2_Supervised_Learning.pdf",
            "ML_MidSem_Question_Paper.pdf"
        ]
    }
]


def run_benchmark_evaluation(search_fn, vectorizer, tfidf_matrix, documents, min_score=0.05):
    """
    Runs the benchmark queries through the retrieval pipeline and computes metrics.
    """
    results = []
    total_p, total_r, total_f1 = 0.0, 0.0, 0.0

    for item in BENCHMARK_QUERIES:
        q = item["query"]
        relevant = item["relevant"]
        retrieved_raw = search_fn(q, vectorizer, tfidf_matrix, documents, min_score=min_score)
        retrieved_ids = [r["filename"] for r in retrieved_raw]

        eval_res = evaluate_retrieval(retrieved_ids, relevant)
        eval_res["query"] = q
        eval_res["description"] = item["description"]
        eval_res["retrieved_docs"] = retrieved_ids
        eval_res["relevant_docs"] = relevant

        total_p += eval_res["precision"]
        total_r += eval_res["recall"]
        total_f1 += eval_res["f1"]
        results.append(eval_res)

    n = len(BENCHMARK_QUERIES) if BENCHMARK_QUERIES else 1
    summary = {
        "mean_precision": total_p / n,
        "mean_recall": total_r / n,
        "mean_f1": total_f1 / n,
        "query_results": results
    }
    return summary
