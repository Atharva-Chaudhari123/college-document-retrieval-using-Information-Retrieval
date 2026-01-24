from sklearn.metrics.pairwise import cosine_similarity

def search(query, vectorizer, tfidf_matrix, filenames):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    ranked = sorted(
        zip(filenames, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked
