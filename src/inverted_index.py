import os
from collections import defaultdict
from pathlib import Path
from src.preprocess import preprocess
from src.document_loader import extract_text_from_pdf, extract_text_from_txt


def build_inverted_index(data_dir):
    """
    Constructs an Inverted Index mapping term -> list of document names.
    Supports both .txt and .pdf files across the data directory and subdirectories.
    Preserves exact backward compatibility with the base project.
    """
    index = defaultdict(list)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Walk through directory and subdirectories
    for root, _, files in os.walk(data_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in [".txt", ".pdf"]:
                file_path = os.path.join(root, file)
                if ext == ".pdf":
                    content = extract_text_from_pdf(file_path)
                else:
                    content = extract_text_from_txt(file_path)

                if not content:
                    continue

                tokens = preprocess(content)
                for token in set(tokens):
                    if file not in index[token]:
                        index[token].append(file)

    return dict(index)


def build_inverted_index_from_docs(documents):
    """
    Builds an inverted index from an in-memory collection of document dicts.
    Each dict should have 'id' (or 'filename') and 'content'.
    """
    index = defaultdict(list)
    for doc in documents:
        doc_id = doc.get("filename", doc.get("id", "doc"))
        tokens = preprocess(doc.get("content", ""))
        for token in set(tokens):
            if doc_id not in index[token]:
                index[token].append(doc_id)
    return dict(index)
