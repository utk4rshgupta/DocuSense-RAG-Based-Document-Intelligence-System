# DocuSense 📄🔍
### RAG-Based Document Intelligence System

A Retrieval-Augmented Generation (RAG) pipeline built in Google Colab that ingests multi-format document collections, indexes them semantically using FAISS, and answers natural language queries using Groq's LLaMA 3.3-70B. Also performs LDA-based topic modeling and visualizes topic-keyword relationships as an interactive graph.

---

## Features

- **Multi-format ingestion** — Reads PDF, TXT, and RTF files from Google Drive
- **Intelligent chunking** — Splits documents using LangChain's `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 100)
- **Semantic embeddings** — Encodes chunks with `all-MiniLM-L6-v2` via SentenceTransformers
- **FAISS vector search** — Flat L2 index for fast top-k nearest-neighbor retrieval
- **LLM-powered Q&A** — Answers grounded strictly in retrieved context using Groq's LLaMA 3.3-70B
- **Topic modeling** — LDA with coherence-based optimal K selection (tested K = 5–20)
- **Topic graph visualization** — NetworkX + Matplotlib spring-layout graph of topics and keywords

---

## Tech Stack

| Layer | Tool |
|---|---|
| Runtime | Google Colab |
| Document parsing | PyMuPDF (`fitz`), `striprtf` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector index | FAISS (`faiss-cpu`) |
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Topic modeling | Gensim LDA + CoherenceModel |
| Visualization | NetworkX, Matplotlib |
| Data handling | Pandas, NumPy |

---

## Pipeline Overview

```
Google Drive (PDF / TXT / RTF)
        │
        ▼
  Document Loader
  (PyMuPDF + striprtf)
        │
        ▼
  Text Chunker
  (LangChain RecursiveCharacterTextSplitter)
        │
        ▼
  Text Cleaner
  (stopword removal, regex normalization)
        │
        ▼
  Embedding Model
  (all-MiniLM-L6-v2)
        │
        ▼
  FAISS Index
  (IndexFlatL2)
        │
   ┌────┴────┐
   ▼         ▼
RAG Q&A    Topic Modeling
(Groq LLM) (LDA + Graph)
```

---

## Setup & Usage

### 1. Clone / Open in Colab

Upload `DocuSense.ipynb` to Google Colab or open it directly from Drive.

### 2. Prepare your documents

Create a folder in Google Drive at:
```
My Drive/article_analysis/
```
Place your `.pdf`, `.txt`, or `.rtf` files inside it.

### 3. Set your Groq API key

In the notebook, replace the placeholder:
```python
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
```
Get a free key at [console.groq.com](https://console.groq.com).

### 4. Run cells top to bottom

| Step | What it does |
|---|---|
| Cell 0 | Mounts Drive, loads all documents |
| Cell 1 | Chunks documents into 1000-char segments |
| Cell 6 | Cleans text (stopwords, regex) |
| Cell 7–8 | Generates embeddings, builds FAISS index |
| Cell 13 | `ask_rag(question)` — query your documents |
| Cell 20–22 | LDA topic modeling, finds optimal K |
| Cell 24–26 | LLM extracts topic names + keywords as JSON |
| Cell 27–28 | Renders topic-keyword graph |

---

## Example Queries

```python
ask_rag("What companies are mentioned across all articles?")
ask_rag("Give the companies in descending order of mention frequency.")
ask_rag("What is the overall theme of these articles?")
```

---

## Project Structure

```
DocuSense/
├── DocuSense.ipynb       # Main Colab notebook
└── README.md             # This file
```

---

## Notes

- The Groq API is accessed via the OpenAI-compatible client (`base_url="https://api.groq.com/openai/v1"`).
- Retrieval uses top-k = 30 chunks by default; adjustable in `ask_rag()`.
- LDA coherence is evaluated using the `u_mass` metric across K = 5 to 20.
- Keep your API key out of version control — use environment variables or Colab Secrets.

---

## Author

**Utkarsh**  
B.Tech Computer Science — Ajay Kumar Garg Engineering College (2027)  
[GitHub](#) · [LinkedIn](#)
