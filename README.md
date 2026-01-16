# Advanced AI-Powered Document Insights & Data Extraction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Gemini](https://img.shields.io/badge/AI-Google%20Gemini-orange?style=for-the-badge&logo=google&logoColor=white)
![Gradio](https://img.shields.io/badge/UI-Gradio-purple?style=for-the-badge&logo=gradio&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

## Project Overview
This repository contains the final submission for the **Outamation Advanced AI-Powered Document Insights and Data Extraction Externship**. 

The project implements a robust **Retrieval-Augmented Generation (RAG)** pipeline designed to ingest, classify, and extract insights from complex PDF documents (Resumes, Invoices, Contracts, etc.). Unlike standard RAG implementations, this pipeline utilizes **Hybrid Search (Vector + KNN)** and **Docling** for high-fidelity document parsing, ensuring accurate retrieval even for diverse document layouts.

## Key Features

* **Advanced Parsing with Docling:** Utilizes `Docling` to intelligently parse PDFs, preserving structure and classifying pages by type (e.g., distinguishing an Invoice page from a Contract page).
* **Hybrid Retrieval System:** Combines semantic **Vector Search** (`all-MiniLM-L6-v2` embeddings) with keyword-based **KNN Search** (TF-IDF) to maximize retrieval accuracy and diversity.
* **LLM Integration:** Powered by **Google Gemini 2.5 Flash** for high-speed, context-aware answer generation with strict source citation.
* **Smart Chunking:** Implements context-aware chunking strategies with metadata retention (page numbers, document types).
* **Interactive UI:** A full-featured **Gradio** interface for uploading documents, visualizing database stats, and performing Q&A in real-time.

## Technical Challenges & Solutions

During the development of this pipeline, several key technical challenges were addressed:

### 1. Robustness in PDF Parsing
* **Challenge:** Modern PDFs vary wildly in structure (text-based, scanned images, complex tables), and single parsing libraries often fail on edge cases.
* **Solution:** I implemented a **Dual-Layer Extraction Strategy**. The system primarily attempts high-fidelity parsing using `Docling`. However, I engineered a seamless fallback mechanism that switches to `PyPDF2` if Docling encounters errors or returns empty text, ensuring 100% pipeline reliability across different file types.

### 2. Balancing Precision vs. Context
* **Challenge:** Standard Vector Search is excellent for semantic understanding but can miss exact keyword matches (like specific invoice numbers or rare technical skills).
* **Solution:** I built a **Hybrid Search Engine** that retrieves results using both Dense Vectors (Semantic) and Sparse TF-IDF (Keyword/KNN). The results are re-ranked and combined to ensure the LLM receives context that is both conceptually relevant and factually precise.

## Tech Stack

* **Core Logic:** Python
* **LLM:** Google Gemini API (`gemini-2.5-flash`)
* **Vector Store:** ChromaDB
* **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
* **Document Processing:** Docling, PyPDF2 (Fallback)
* **UI Framework:** Gradio
* **Libraries:** Scikit-Learn, Pandas, NumPy

## Methodology

The pipeline follows a strict architectural flow:

1.  **Ingestion:** PDFs are uploaded and passed through the extraction layer.
2.  **Classification:** A rule-based classifier identifies the document type (Resume, Invoice, Payslip, etc.) for *every* page based on keyword density.
3.  **Chunking:** Text is segmented into intelligent chunks (800 chars) with overlap (150 chars) to maintain context across boundaries.
4.  **Embedding:** Chunks are vectorized using `sentence-transformers` and stored in **ChromaDB**.
5.  **Retrieval:** When a question is asked, the system performs a **Hybrid Search** (Weighted combination of Dense Vector similarity and Sparse TF-IDF scores).
6.  **Generation:** The top-ranked chunks are fed into **Gemini 2.5 Flash** to generate a cited, fact-based answer.

## Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/Outamation-AI-Document-Insights.git](https://github.com/yourusername/Outamation-AI-Document-Insights.git)
    cd Outamation-AI-Document-Insights
    ```

2.  **Install Dependencies**
    ```bash
    pip install -q docling gradio chromadb sentence-transformers google-generativeai scikit-learn pandas
    ```

3.  **Set up API Keys**
    * Get a Google Gemini API Key from Google AI Studio.
    * Set the key in the script or environment variables: `GEMINI_API_KEY = "your_key_here"`

4.  **Run the Application**
    ```bash
    python app.py
    ```
    *Note: If running in a Jupyter Notebook/Colab, simply execute the cells sequentially.*

## Acknowledgments
* **Outamation** for the externship opportunity.
* **Google DeepMind** for the Gemini API.
* **Hugging Face** for the Sentence Transformers library.
