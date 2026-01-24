# TF-IDF Based Search Engine with Inverted Index
Deploymemt link: https://tf-idf-search-engine.streamlit.app/
## Overview
This project implements a basic search engine using classical Information Retrieval
techniques. It processes a collection of text documents, builds an inverted index for
efficient lookup, and ranks documents using TF-IDF and cosine similarity.

## Technologies Used
- Python
- NLTK
- scikit-learn
- NumPy
- pytest

## Features
- Text preprocessing using tokenization and stopword removal
- Inverted index construction for fast term-based lookup
- TF-IDF vectorization of documents
- Query-based document ranking using cosine similarity
- Evaluation using precision and recall metrics
- Unit tests for indexing and search functionality

## Project Structure
- data/        : Text document corpus
- src/         : Source code for indexing, search, and evaluation
- tests/       : Unit tests using pytest
- results/     : Sample queries and evaluation report

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Run the search engine:
   cd src
   python main.py

3. Run unit tests:
   pytest

## Conclusion
This project demonstrates a complete implementation of a TF-IDF based Vector Space
Information Retrieval model and serves as a foundation for building more advanced
search engines.
