def precision_recall(retrieved, relevant):
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)

    true_positives = retrieved_set & relevant_set

    precision = len(true_positives) / len(retrieved_set) if retrieved_set else 0
    recall = len(true_positives) / len(relevant_set) if relevant_set else 0

    return precision, recall
