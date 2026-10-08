import re
import nltk
from nltk.corpus import stopwords

try:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
        "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for", 
        "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", 
        "him", "himself", "his", "how", "if", "in", "into", "is", "it", "its", "itself", "just", 
        "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", 
        "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", 
        "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them", 
        "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", 
        "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", 
        "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves"
    }


def preprocess(text: str):
    """
    Standard IR text preprocessing:
    1. Case folding (lowercase)
    2. Non-alphabetic character removal
    3. Tokenization (whitespace split)
    4. Stop-word removal
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return tokens


def get_query_tokens(query: str):
    """Extract processed search terms from a user query."""
    return preprocess(query)
