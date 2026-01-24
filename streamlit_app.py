import streamlit as st
import os

from src.inverted_index import build_inverted_index
from src.tfidf_model import build_tfidf
from src.search import search


DATA_DIR = "data"

st.set_page_config(page_title="TF-IDF Search Engine", layout="centered")

st.title(" TF-IDF Search Engine")
st.write("Search documents using **TF-IDF & Inverted Index**")

# Load documents
documents = []
filenames = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".txt"):
        filenames.append(file)
        with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
            documents.append(f.read())

# Build models
inverted_index = build_inverted_index(DATA_DIR)
vectorizer, tfidf_matrix = build_tfidf(documents)

# User input
query = st.text_input("Enter your search query:")

if query:
    results = search(query, vectorizer, tfidf_matrix, filenames)

    st.subheader("Search Results")

    found = False
    for doc, score in results:
        if score > 0:
            found = True
            st.write(f"**{doc}** — Score: `{score:.4f}`")

    if not found:
        st.warning("No relevant documents found.")

# Optional: Show inverted index
with st.expander(" View Inverted Index"):
    st.write(inverted_index)
