import difflib
import re

ACADEMIC_SYNONYMS = {
    "clt": "central limit theorem",
    "nn": "neural networks",
    "db": "database",
    "dbms": "database management systems",
    "nlp": "natural language processing",
    "ir": "information retrieval",
    "vsm": "vector space model",
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "svm": "support vector machine",
    "pca": "principal component analysis",
    "ols": "ordinary least squares",
    "1nf": "first normal form",
    "2nf": "second normal form",
    "3nf": "third normal form",
    "bcnf": "boyce codd normal form",
    "acid": "atomicity consistency isolation durability"
}


def correct_query_spelling(query: str, vocabulary: set, cutoff: float = 0.65):
    """
    Information Retrieval Spelling Correction & Query Expansion:
    1. Expands academic acronyms (e.g. 'clt' -> 'central limit theorem').
    2. For out-of-vocabulary terms, computes minimum edit distance against
       the indexed vocabulary using character n-gram / SequenceMatcher similarity.
    3. Maps spelling mistakes (e.g. 'regresion decent' -> 'regression descent')
       to valid vocabulary tokens in the indexed corpus.
    
    Returns:
    - corrected_query (str): The refined query string
    - was_corrected (bool): True if any corrections or expansions occurred
    - corrections (dict): Mapping of {original_word: corrected_word}
    """
    if not query or not vocabulary:
        return query, False, {}

    raw_tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
    if not raw_tokens:
        return query, False, {}

    corrections = {}
    expanded_tokens = []

    for token in raw_tokens:
        # Check academic abbreviations first
        if token in ACADEMIC_SYNONYMS:
            expansion = ACADEMIC_SYNONYMS[token]
            corrections[token] = expansion
            expanded_tokens.append(expansion)
            continue

        # If token is already present in corpus vocabulary, retain it
        if token in vocabulary:
            expanded_tokens.append(token)
            continue

        # Out of vocabulary: find closest match in corpus dictionary
        matches = difflib.get_close_matches(token, vocabulary, n=1, cutoff=cutoff)
        if matches:
            match = matches[0]
            corrections[token] = match
            expanded_tokens.append(match)
        else:
            expanded_tokens.append(token)

    was_corrected = len(corrections) > 0
    corrected_query = " ".join(expanded_tokens)
    return corrected_query, was_corrected, corrections
