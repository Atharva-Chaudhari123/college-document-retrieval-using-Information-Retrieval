from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf(documents):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix
