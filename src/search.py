import re
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocess import get_query_tokens


def search(query, vectorizer, tfidf_matrix, filenames):
    """
    Original baseline search function:
    1. Transforms user query into TF-IDF vector space
    2. Computes Cosine Similarity against all document vectors
    3. Ranks documents in descending order of similarity score
    """
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    ranked = sorted(
        zip(filenames, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked


def highlight_terms(text: str, query_terms: list) -> str:
    """
    Highlights query terms in text using markdown bold (**term**).
    Uses a single regex union to prevent double bolding and nested markdown asterisks.
    """
    if not text or not query_terms:
        return text

    terms_set = set()
    for t in query_terms:
        t_clean = t.strip()
        if t_clean:
            terms_set.add(t_clean)

    if not terms_set:
        return text

    sorted_terms = sorted(terms_set, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in sorted_terms) + r")\b", re.IGNORECASE)
    return pattern.sub(r"**\1**", text)


def extract_snippet(content: str, query: str, query_terms: list = None, char_window: int = 240) -> str:
    """
    Extracts a relevant contextual snippet containing query terms,
    and highlights matching keywords.
    """
    if not content:
        return "No text available."

    clean_content = " ".join(content.split())
    if not query:
        snippet = clean_content[:char_window] + ("..." if len(clean_content) > char_window else "")
        return snippet

    if query_terms is None:
        query_terms = get_query_tokens(query)

    # Find the position of the earliest matching query term in document
    match_pos = -1
    best_term = ""
    lower_content = clean_content.lower()

    # First check full query string
    q_clean = query.lower().strip()
    if q_clean and q_clean in lower_content:
        match_pos = lower_content.find(q_clean)
        best_term = q_clean
    else:
        # Search individual processed terms
        for term in query_terms:
            pos = lower_content.find(term.lower())
            if pos != -1 and (match_pos == -1 or pos < match_pos):
                match_pos = pos
                best_term = term

    if match_pos == -1:
        # Fallback to start of document if no term directly matches
        snippet = clean_content[:char_window] + ("..." if len(clean_content) > char_window else "")
    else:
        # Compute snippet window centered around match
        start = max(0, match_pos - (char_window // 3))
        end = min(len(clean_content), start + char_window)

        # Snap to nearest word boundary
        if start > 0:
            space_before = clean_content.find(" ", start)
            if space_before != -1 and space_before < match_pos:
                start = space_before + 1

        if end < len(clean_content):
            space_after = clean_content.rfind(" ", start, end)
            if space_after != -1 and space_after > match_pos:
                end = space_after

        snippet = clean_content[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(clean_content):
            snippet = snippet + "..."

    # Highlight terms in the snippet
    highlight_candidates = list(query_terms)
    if q_clean and q_clean not in highlight_candidates:
        highlight_candidates.append(q_clean)

    return highlight_terms(snippet, highlight_candidates)


def search_academic_documents(query, vectorizer, tfidf_matrix, documents, filters=None, min_score=0.0):
    """
    Search academic documents collection with metadata filtering and snippet generation.

    Parameters:
    - query: User search query string
    - vectorizer: Fitted TfidfVectorizer
    - tfidf_matrix: Document TF-IDF matrix
    - documents: List of document dictionaries (from document_loader)
    - filters: Dict with optional keys ('doc_type', 'subject', 'semester')
    - min_score: Minimum cosine similarity score threshold (default 0.0)

    Returns:
    - List of ranked result dictionaries with rank, score, metadata, and snippet
    """
    if not query.strip() or len(documents) == 0:
        return []

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    query_terms = get_query_tokens(query)

    ranked_results = []
    for doc, score in zip(documents, scores):
        if score <= min_score:
            continue

        # Apply metadata filters if specified
        if filters:
            if filters.get("doc_type") and filters["doc_type"] != "All Types":
                if doc.get("doc_type") != filters["doc_type"]:
                    continue

            if filters.get("subject") and filters["subject"] != "All Subjects":
                if doc.get("subject") != filters["subject"]:
                    continue

            if filters.get("semester") and filters["semester"] != "All Semesters":
                if doc.get("semester") != filters["semester"]:
                    continue

        raw_snippet = extract_snippet(doc.get("content", ""), query, query_terms)
        
        # Build HTML-formatted snippet for modern web card rendering
        highlight_candidates = list(query_terms)
        q_clean = query.lower().strip()
        if q_clean and q_clean not in highlight_candidates:
            highlight_candidates.append(q_clean)
            
        plain_snippet = " ".join(raw_snippet.replace("**", "").split())
        html_snippet = plain_snippet
        if highlight_candidates:
            sorted_terms = sorted({t.strip() for t in highlight_candidates if t.strip()}, key=len, reverse=True)
            if sorted_terms:
                pat = re.compile(r"\b(" + "|".join(re.escape(t) for t in sorted_terms) + r")\b", re.IGNORECASE)
                html_snippet = pat.sub(r'<mark class="query-highlight">\1</mark>', plain_snippet)

        ranked_results.append({
            "id": doc.get("id", doc.get("filename")),
            "filename": doc.get("filename"),
            "title": doc.get("title", doc.get("filename")),
            "doc_type": doc.get("doc_type", "Other"),
            "subject": doc.get("subject", "Computer Science"),
            "semester": doc.get("semester", ""),
            "score": float(score),
            "snippet": raw_snippet,
            "html_snippet": html_snippet,
            "full_content": doc.get("content", ""),
            "file_path": doc.get("file_path", "")
        })

    # Sort descending by cosine similarity score
    ranked_results.sort(key=lambda x: x["score"], reverse=True)

    # Assign 1-indexed ranks
    for rank_idx, item in enumerate(ranked_results, 1):
        item["rank"] = rank_idx

    return ranked_results
