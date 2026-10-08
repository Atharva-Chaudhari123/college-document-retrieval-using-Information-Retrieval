"""
Script to generate sample academic PDF documents using reportlab.
Populates data/notes/, data/question_papers/, data/assignments/, and data/syllabus/.
"""

import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


DOCUMENTS_METADATA = {
    # Notes
    "DVM_Unit_1_Notes.pdf": {
        "folder": "data/notes",
        "title": "Data Modeling Unit 1 Notes",
        "doc_type": "Notes",
        "subject": "Data Visualization",
        "semester": "Semester 7",
        "paragraphs": [
            "DEPARTMENT OF COMPUTER ENGINEERING - ACADEMIC YEAR 2025-2026",
            "Subject: Data Visualization and Modeling | Unit 1: Foundations of Statistical Data Modeling",
            "Topics: Sampling Distributions, Central Limit Theorem, Probability Density, Normal Approximation.",
            "1. Introduction to the Central Limit Theorem (CLT):",
            "The Central Limit Theorem states that the sampling distribution of the sample mean approaches a normal distribution as the sample size increases, regardless of the shape of the population distribution. Specifically, given independent and identically distributed (i.i.d.) random variables with mean mu and finite variance sigma squared, the sample mean is approximately distributed as N(mu, sigma^2 / n).",
            "2. Practical Significance in Data Modeling:",
            "The Central Limit Theorem is fundamental in inferential statistics, data visualization, and hypothesis testing because it permits calculating confidence intervals and p-values even when working with skewed data distributions in real-world visual analytics.",
            "3. Visualizing Sampling Distributions:",
            "When sampling from uniform or exponential distributions, visual plots demonstrate that histograms of the sample mean form a symmetric bell curve as sample size n exceeds 30."
        ]
    },
    "Statistics_Notes.pdf": {
        "folder": "data/notes",
        "title": "Statistics and Probability Notes",
        "doc_type": "Notes",
        "subject": "Statistics",
        "semester": "Semester 5",
        "paragraphs": [
            "DEPARTMENT OF MATHEMATICS & DATA SCIENCE",
            "Course: Probability Theory and Mathematical Statistics | Unit 3: Limit Theorems",
            "1. Law of Large Numbers and Central Limit Theorem:",
            "The Central Limit Theorem provides the foundation for statistical estimation. When drawing independent random samples from any distribution with finite variance, the standardized sum converges in distribution to a standard normal distribution.",
            "2. Standard Error of the Mean:",
            "The standard error equals sigma divided by the square root of n. As sample size grows, the standard error shrinks, meaning our sample mean concentrates around the population parameter.",
            "3. Applications in Statistical Inference:",
            "Z-tests, t-tests, and regression analysis rely on the normality assumption guaranteed asymptotically by the Central Limit Theorem."
        ]
    },
    "IR_Lecture_Notes_TFIDF.pdf": {
        "folder": "data/notes",
        "title": "Information Retrieval Lecture Notes: TF-IDF & Vector Space Model",
        "doc_type": "Notes",
        "subject": "Information Retrieval",
        "semester": "Semester 7",
        "paragraphs": [
            "DEPARTMENT OF INFORMATION TECHNOLOGY - COURSE: INFORMATION RETRIEVAL",
            "Unit 2: Text Preprocessing, Term Weighting, and Vector Space Retrieval",
            "1. Text Preprocessing in Information Retrieval:",
            "Raw documents undergo tokenization, case normalization, punctuation removal, and stop-word filtering using curated stop-word dictionaries.",
            "2. Inverted Index Data Structure:",
            "An inverted index is an essential IR data structure mapping each dictionary term to a postings list containing document identifiers and term frequencies.",
            "3. Term Frequency - Inverse Document Frequency (TF-IDF):",
            "TF-IDF measures the importance of a term within a document relative to the entire corpus. Term Frequency (TF) captures local importance, while Inverse Document Frequency (IDF) penalizes common terms across all documents: IDF(t) = log(N / DF(t)).",
            "4. Vector Space Model & Cosine Similarity:",
            "Documents and queries are represented as high-dimensional TF-IDF vectors. Retrieval ranking evaluates the cosine similarity between the query vector and candidate document vectors."
        ]
    },
    "ML_Unit_2_Supervised_Learning.pdf": {
        "folder": "data/notes",
        "title": "Machine Learning Unit 2 Notes: Supervised Learning",
        "doc_type": "Notes",
        "subject": "Machine Learning",
        "semester": "Semester 6",
        "paragraphs": [
            "DEPARTMENT OF ARTIFICIAL INTELLIGENCE & MACHINE LEARNING",
            "Subject: Machine Learning | Unit 2: Supervised Learning Algorithms",
            "1. Linear Regression and Cost Functions:",
            "Linear regression models the relationship between dependent continuous variables and independent explanatory features using the ordinary least squares or gradient descent optimization.",
            "2. Classification and Logistic Regression:",
            "Logistic regression applies the sigmoid activation function to output predicted class probabilities between 0 and 1.",
            "3. Decision Trees and Random Forests:",
            "Decision trees partition the feature space based on Information Gain or Gini Impurity. Ensemble learning combines multiple weak learners to decrease variance and prevent overfitting."
        ]
    },
    "DBMS_Unit_3_Normalization.pdf": {
        "folder": "data/notes",
        "title": "DBMS Unit 3 Notes: Relational Schema & Normalization",
        "doc_type": "Notes",
        "subject": "Database Management",
        "semester": "Semester 5",
        "paragraphs": [
            "DEPARTMENT OF COMPUTER ENGINEERING - COURSE: DATABASE MANAGEMENT SYSTEMS",
            "Unit 3: Relational Database Design and Normalization Theory",
            "1. Functional Dependencies:",
            "A functional dependency X -> Y specifies that if two tuples agree on attributes X, they must also agree on attributes Y.",
            "2. Normal Forms (1NF, 2NF, 3NF, BCNF):",
            "First Normal Form eliminates repeating groups and ensures atomic attributes. Second Normal Form eliminates partial functional dependencies. Third Normal Form eliminates transitive dependencies.",
            "3. ACID Properties in Transaction Processing:",
            "Atomicity, Consistency, Isolation, and Durability ensure reliable transaction execution and database integrity."
        ]
    },

    # Question Papers
    "DVM_Previous_Year_Question_Paper.pdf": {
        "folder": "data/question_papers",
        "title": "DVM Previous Year Question Paper",
        "doc_type": "Question Paper",
        "subject": "Data Visualization",
        "semester": "Semester 7",
        "paragraphs": [
            "SEMESTER END EXAMINATION - DATA VISUALIZATION AND MODELING (2025)",
            "Time: 3 Hours | Total Marks: 80",
            "SECTION A (Attempt Any Two):",
            "Q1. (a) State and explain the Central Limit Theorem. Discuss its importance in sampling distribution analysis and statistical data visualization. [10 Marks]",
            "Q1. (b) What are Q-Q plots? Explain how they are used to verify the normality assumption produced by the Central Limit Theorem. [10 Marks]",
            "SECTION B:",
            "Q2. (a) Differentiate between Bar Charts, Histograms, and Box Plots for exploratory data analysis. [10 Marks]",
            "Q2. (b) Explain visual encodings and perceptual principles formulated by Edward Tufte. [10 Marks]"
        ]
    },
    "IR_EndSem_Question_Paper_2025.pdf": {
        "folder": "data/question_papers",
        "title": "Information Retrieval End-Sem Question Paper 2025",
        "doc_type": "Question Paper",
        "subject": "Information Retrieval",
        "semester": "Semester 7",
        "paragraphs": [
            "END SEMESTER EXAMINATION - INFORMATION RETRIEVAL",
            "Total Marks: 70 | Duration: 2.5 Hours",
            "Q1. Explain the architecture of an Information Retrieval system. Define tokenization, stop-word removal, and inverted index creation. [14 Marks]",
            "Q2. Given a collection of 3 documents, calculate the TF-IDF weight matrix and rank documents for the query 'machine learning' using Cosine Similarity. [14 Marks]",
            "Q3. Discuss evaluation metrics in IR: Precision, Recall, and F1-score with clear formulas and confusion matrix examples. [14 Marks]"
        ]
    },
    "ML_MidSem_Question_Paper.pdf": {
        "folder": "data/question_papers",
        "title": "Machine Learning Mid-Sem Question Paper",
        "doc_type": "Question Paper",
        "subject": "Machine Learning",
        "semester": "Semester 6",
        "paragraphs": [
            "MID SEMESTER EXAMINATION - MACHINE LEARNING",
            "Total Marks: 50 | Duration: 2 Hours",
            "Q1. Derive the cost function for linear regression and explain gradient descent convergence. [10 Marks]",
            "Q2. Explain the bias-variance tradeoff and describe how L1 Lasso and L2 Ridge regularization prevent model overfitting. [10 Marks]",
            "Q3. Differentiate supervised learning, unsupervised learning, and reinforcement learning. [10 Marks]"
        ]
    },

    # Assignments
    "IR_Assignment_1_Vector_Space.pdf": {
        "folder": "data/assignments",
        "title": "IR Assignment 1: Inverted Index & Vector Space Model",
        "doc_type": "Assignment",
        "subject": "Information Retrieval",
        "semester": "Semester 7",
        "paragraphs": [
            "DEPARTMENT OF COMPUTER ENGINEERING - ASSIGNMENT 1",
            "Subject: Information Retrieval | Submission Deadline: Week 4",
            "Task Overview:",
            "1. Implement a text preprocessing pipeline consisting of tokenization, lowercase conversion, and stop-word elimination.",
            "2. Construct an Inverted Index mapping distinct vocabulary terms to document IDs.",
            "3. Compute TF-IDF matrices and implement Cosine Similarity retrieval ranking for multi-word queries.",
            "4. Evaluate the retrieved document rankings using Precision, Recall, and F1 score against a ground-truth relevance test set."
        ]
    },
    "DBMS_Assignment_2_SQL_Queries.pdf": {
        "folder": "data/assignments",
        "title": "DBMS Assignment 2: SQL & Schema Normalization",
        "doc_type": "Assignment",
        "subject": "Database Management",
        "semester": "Semester 5",
        "paragraphs": [
            "DEPARTMENT OF COMPUTER ENGINEERING - ASSIGNMENT 2",
            "Subject: Database Management Systems | Total Marks: 25",
            "Problem Statement:",
            "1. Design a normalized relational schema in Third Normal Form (3NF) for a university course registration database.",
            "2. Write complex SQL queries involving INNER JOIN, GROUP BY, and HAVING clauses.",
            "3. Demonstrate ACID transaction properties and write rollback scenarios."
        ]
    },

    # Syllabus
    "Data_Visualization_Syllabus.pdf": {
        "folder": "data/syllabus",
        "title": "Data Visualization & Modeling Course Syllabus",
        "doc_type": "Syllabus",
        "subject": "Data Visualization",
        "semester": "Semester 7",
        "paragraphs": [
            "CURRICULUM SYLLABUS - DATA VISUALIZATION AND MODELING",
            "Unit 1: Descriptive Statistics and Sampling Distributions. Central Limit Theorem, normal distributions, standard error, confidence intervals.",
            "Unit 2: Exploratory Data Analysis & Visual Encoding. Color theory, chart selection, Matplotlib and Seaborn visualization libraries.",
            "Unit 3: Regression and Predictive Data Modeling. Correlation, residual plots, evaluation metrics.",
            "Assessment: Mid-Semester Test (30%), Practical Assignments (20%), End-Semester Exam (50%)."
        ]
    },
    "Information_Retrieval_Syllabus.pdf": {
        "folder": "data/syllabus",
        "title": "Information Retrieval Course Syllabus",
        "doc_type": "Syllabus",
        "subject": "Information Retrieval",
        "semester": "Semester 7",
        "paragraphs": [
            "CURRICULUM SYLLABUS - INFORMATION RETRIEVAL",
            "Unit 1: Introduction to IR, Boolean Retrieval, Tokenization, Stemming, Stop-word lists.",
            "Unit 2: Inverted Index construction, dictionary compression, postings lists.",
            "Unit 3: Vector Space Model, Term Frequency (TF), Inverse Document Frequency (IDF), Cosine Similarity.",
            "Unit 4: Evaluation of Retrieval Systems: Precision, Recall, F-measure, Mean Average Precision (MAP).",
            "Unit 5: Web Search, PageRank, crawling, and modern search engine architectures."
        ]
    }
}


def create_pdf(filename, folder, title, paragraphs):
    os.makedirs(folder, exist_ok=True)
    pdf_path = os.path.join(folder, filename)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name="DocTitle",
        parent=styles["Heading1"],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=12
    )

    body_style = ParagraphStyle(
        name="DocBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=8
    )

    story = [
        Paragraph(title, title_style),
        Spacer(1, 10)
    ]

    for p in paragraphs:
        story.append(Paragraph(p, body_style))
        story.append(Spacer(1, 4))

    doc.build(story)
    print(f"Generated PDF: {pdf_path}")


def main():
    metadata_store = {}

    for filename, info in DOCUMENTS_METADATA.items():
        create_pdf(filename, info["folder"], info["title"], info["paragraphs"])
        metadata_store[filename] = {
            "title": info["title"],
            "doc_type": info["doc_type"],
            "subject": info["subject"],
            "semester": info["semester"]
        }

    # Also register doc1.txt, doc2.txt, doc3.txt for consistency
    metadata_store["doc1.txt"] = {
        "title": "AI & Machine Learning Overview",
        "doc_type": "Notes",
        "subject": "Artificial Intelligence",
        "semester": "Semester 6"
    }
    metadata_store["doc2.txt"] = {
        "title": "Deep Learning & Neural Networks",
        "doc_type": "Notes",
        "subject": "Artificial Intelligence",
        "semester": "Semester 6"
    }
    metadata_store["doc3.txt"] = {
        "title": "Python in Data Science",
        "doc_type": "Notes",
        "subject": "Machine Learning",
        "semester": "Semester 5"
    }

    # Save to data/metadata.json
    os.makedirs("data", exist_ok=True)
    with open("data/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, indent=2)

    print("Sample academic dataset created successfully with data/metadata.json!")


if __name__ == "__main__":
    main()
