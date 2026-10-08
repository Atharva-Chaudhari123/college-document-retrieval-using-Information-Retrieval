import os
import json
import io
from pathlib import Path
from pypdf import PdfReader


METADATA_FILE = "data/metadata.json"

DEFAULT_SUBJECTS = [
    "Information Retrieval",
    "Data Visualization",
    "Machine Learning",
    "Database Management",
    "Artificial Intelligence",
    "Statistics",
    "Computer Science",
    "Other"
]

DEFAULT_TYPES = [
    "Notes",
    "Question Paper",
    "Assignment",
    "Syllabus",
    "Lab Manual",
    "Other"
]


def extract_text_from_pdf(file_source) -> str:
    """
    Extracts plain text from a PDF file path or file-like binary stream.
    """
    text_content = []
    try:
        reader = PdfReader(file_source)
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text.strip())
        return "\n\n".join(text_content).strip()
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return ""


def extract_text_from_txt(file_source) -> str:
    """
    Extracts text from a .txt file path or bytes stream.
    """
    try:
        if isinstance(file_source, (str, Path)):
            with open(file_source, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()
        elif hasattr(file_source, "read"):
            content = file_source.read()
            if isinstance(content, bytes):
                return content.decode("utf-8", errors="ignore").strip()
            return str(content).strip()
    except Exception as e:
        print(f"Error reading TXT: {e}")
        return ""
    return ""


def infer_metadata(file_path: Path):
    """
    Infers document type, subject, and semester based on folder structure and naming convention.
    """
    parent = file_path.parent.name.lower()
    name = file_path.stem.lower()

    # Document type inference
    doc_type = "Other"
    if "note" in parent or "note" in name:
        doc_type = "Notes"
    elif "question" in parent or "qp" in name or "paper" in name or "exam" in name:
        doc_type = "Question Paper"
    elif "assign" in parent or "assign" in name:
        doc_type = "Assignment"
    elif "syllab" in parent or "syllab" in name:
        doc_type = "Syllabus"
    elif "lab" in parent or "manual" in name:
        doc_type = "Lab Manual"

    # Subject inference
    subject = "Computer Science"
    if "dvm" in name or "data visual" in name:
        subject = "Data Visualization"
    elif "ir" in name or "information retrieval" in name:
        subject = "Information Retrieval"
    elif "ml" in name or "machine learning" in name:
        subject = "Machine Learning"
    elif "dbms" in name or "database" in name or "sql" in name:
        subject = "Database Management"
    elif "ai" in name or "artificial intelligence" in name:
        subject = "Artificial Intelligence"
    elif "stat" in name:
        subject = "Statistics"

    # Semester inference
    semester = "Semester 7"
    if "sem5" in name or "sem 5" in name:
        semester = "Semester 5"
    elif "sem6" in name or "sem 6" in name:
        semester = "Semester 6"
    elif "sem7" in name or "sem 7" in name:
        semester = "Semester 7"
    elif "sem8" in name or "sem 8" in name:
        semester = "Semester 8"

    return doc_type, subject, semester


def load_metadata_store():
    """Load persistent metadata mappings if available."""
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_metadata_store(metadata_dict):
    """Save persistent metadata mappings."""
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2)


def load_academic_documents(data_dir="data"):
    """
    Recursively scans the data directory, extracting text from .txt and .pdf documents,
    along with metadata.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    stored_metadata = load_metadata_store()
    documents = []

    for file_path in data_path.rglob("*"):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext not in [".txt", ".pdf"]:
            continue

        filename = file_path.name

        # Extract text
        if ext == ".pdf":
            content = extract_text_from_pdf(str(file_path))
        else:
            content = extract_text_from_txt(str(file_path))

        if not content:
            continue

        # Determine metadata
        inferred_type, inferred_subject, inferred_sem = infer_metadata(file_path)
        override = stored_metadata.get(filename, {})

        doc_record = {
            "id": filename,
            "filename": filename,
            "title": override.get("title", file_path.stem.replace("_", " ")),
            "doc_type": override.get("doc_type", inferred_type),
            "subject": override.get("subject", inferred_subject),
            "semester": override.get("semester", inferred_sem),
            "content": content,
            "file_path": str(file_path),
            "extension": ext
        }
        documents.append(doc_record)

    return documents


def save_uploaded_document(uploaded_file, doc_type, subject, semester, title=None, data_dir="data/uploaded"):
    """
    Saves an uploaded file to disk and registers its metadata.
    """
    save_folder = Path(data_dir)
    save_folder.mkdir(parents=True, exist_ok=True)

    filename = uploaded_file.name
    target_path = save_folder / filename

    file_bytes = uploaded_file.getvalue()
    with open(target_path, "wb") as f:
        f.write(file_bytes)

    # Extract text
    if filename.lower().endswith(".pdf"):
        content = extract_text_from_pdf(io.BytesIO(file_bytes))
    else:
        content = extract_text_from_txt(io.BytesIO(file_bytes))

    # Save metadata
    store = load_metadata_store()
    store[filename] = {
        "title": title or target_path.stem.replace("_", " "),
        "doc_type": doc_type,
        "subject": subject,
        "semester": semester
    }
    save_metadata_store(store)

    return {
        "id": filename,
        "filename": filename,
        "title": title or target_path.stem.replace("_", " "),
        "doc_type": doc_type,
        "subject": subject,
        "semester": semester,
        "content": content,
        "file_path": str(target_path),
        "extension": target_path.suffix.lower()
    }
