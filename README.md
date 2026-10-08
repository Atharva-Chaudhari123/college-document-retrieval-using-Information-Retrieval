# Academic Document Information Retrieval System

An Information Retrieval (IR) system designed for college students to quickly search, retrieve, and rank academic documents (Lecture Notes, Question Papers, Assignments, Syllabus, and Study Material) using classical Vector Space Model techniques.

---

## Problem Statement

College students often accumulate large collections of unstructured academic documents across semesters—including PDF lecture notes, previous year question papers, assignment briefs, and course syllabi. Finding relevant concepts or exam topics manually across dozens of documents is slow and inefficient.

---

## Objective

Build a dedicated **Information Retrieval System** that indexes college academic documents (PDF and TXT), extracts text and metadata, and accurately retrieves and ranks the most relevant documents in response to student queries using classical IR methods (TF-IDF and Cosine Similarity), complete with metadata filtering and evaluation metrics (Precision, Recall, F1-score).

---

## Information Retrieval (IR) Techniques Used

- **Text Preprocessing:** Case normalization, non-alphabetic filtering, and tokenization.
- **Stop-word Removal:** NLTK English stop-word filtering to eliminate non-discriminative terms.
- **Inverted Index:** Term-to-postings list index mapping vocabulary terms to matching documents for fast lookup.
- **Vector Space Model (VSM):** High-dimensional vector representation of queries and academic documents.
- **TF-IDF Weighting:** Term Frequency - Inverse Document Frequency scheme:
  $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)$$
- **Cosine Similarity:** Measures the angular similarity between query and document vectors:
  $$\text{Cosine Similarity}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|}$$
- **Relevance Ranking & Snippets:** Documents ranked by descending similarity score with dynamic context snippets and query term highlighting.
- **IR Evaluation:** Quantitative evaluation using **Precision**, **Recall**, and **F1-Score** against ground-truth relevant document sets.

---

## Technologies Used

- **Python 3.10+**
- **Streamlit** (Interactive web application interface)
- **scikit-learn** (`TfidfVectorizer`, `cosine_similarity`)
- **NLTK** (Text tokenization and stop-word dictionaries)
- **NumPy & Pandas** (Vector calculations and tabular metric reporting)
- **pypdf** (Robust PDF text extraction)
- **reportlab** (Academic sample dataset generation)
- **pytest** (Automated unit testing suite)

---

## Project Structure

```text
CL2 (bio & ir) mini project/
│
├── app.py                     # Primary Streamlit web application
├── streamlit_app.py           # Alternate Streamlit entry point
├── create_sample_academic_data.py # Script generating sample academic PDFs
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive project documentation
│
├── data/                      # Academic document repository
│   ├── notes/                 # Lecture notes (PDFs: DVM, Statistics, IR, ML, DBMS)
│   ├── question_papers/       # Previous year question papers (PDFs)
│   ├── assignments/           # Course assignments (PDFs)
│   ├── syllabus/              # Course curriculum syllabi (PDFs)
│   ├── uploaded/              # User-uploaded documents via UI
│   ├── metadata.json          # Persistent document metadata store
│   ├── doc1.txt               # Baseline text document
│   ├── doc2.txt               # Baseline text document
│   └── doc3.txt               # Baseline text document
│
├── src/                       # Core IR algorithms & modules
│   ├── __init__.py
│   ├── preprocess.py          # Tokenization, cleaning & stop-word removal
│   ├── tfidf_model.py         # TF-IDF vectorizer & matrix construction
│   ├── search.py              # Cosine similarity ranking, snippet extraction & highlighting
│   ├── inverted_index.py      # Inverted index construction for TXT & PDF
│   ├── document_loader.py     # PDF/TXT extraction, metadata inference & upload handler
│   ├── evaluation.py          # Precision, Recall, F1-score & benchmark suite
│   └── main.py                # Command-line demonstration script
│
└── tests/                     # Unit test suite
    ├── conftest.py
    ├── test_index.py          # Baseline inverted index tests
    ├── test_search.py         # Baseline TF-IDF search tests
    └── test_academic_ir.py    # Academic retrieval, filtering, snippets & evaluation tests
```

---

## Installation & How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate / Verify Sample Academic Dataset (Optional)
The repository comes pre-populated with academic PDFs. To regenerate them:
```bash
python create_sample_academic_data.py
```

### 3. Run the Streamlit Web Application
```bash
streamlit run app.py
```
*(Alternatively: `streamlit run streamlit_app.py`)*

### 4. Run the Command-Line Demo
```bash
python -m src.main
```

### 5. Run the Automated Tests
```bash
pytest -v
```

---

## Viva & Demonstration Guide

During your Information Retrieval viva or project presentation, demonstrate the system in this order:

1. **Launch App:** Run `streamlit run app.py` and explain the problem statement.
2. **Execute a Query:** Search for `"central limit theorem"`.
3. **Analyze Results:**
   - Show how **Data Modeling Unit 1 Notes** and **Statistics Notes** rank at the top with high relevance scores.
   - Point out the extracted snippet and **highlighted query terms**.
4. **Metadata Filtering:**
   - Filter by Document Type: `Notes`, `Question Paper`, `Assignment`, `Syllabus`.
   - Filter by Subject: `Data Visualization`, `Information Retrieval`, `Machine Learning`, `Database Management`.
5. **Inverted Index Tab:**
   - Look up terms like `theorem`, `retrieval`, `normalization` to inspect their postings lists.
6. **Upload a Document:**
   - Use the sidebar to upload a new PDF document, assign its metadata, and search for phrases inside it immediately.
7. **IR Evaluation Benchmark:**
   - Switch to the **Evaluation Benchmark** tab and click *Evaluate All Benchmark Queries*.
   - Explain **Precision**, **Recall**, and **F1-Score** formulas and results.
