# 🔍 Adaptive Multi-Stage RAG System

A production-quality, modular **Retrieval-Augmented Generation** (RAG) pipeline for intelligent document processing and query understanding.

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        USER QUERY                            │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────┐
│         Query Classifier                │
│  factual / summarization / analytical   │
└─────────────────┬───────────────────────┘
                  │ determines top_k
                  ▼
┌─────────────────────────────────────────┐
│         Hybrid Retriever                │
│   Semantic (FAISS) + Keyword (TF-IDF)   │
│   Combined score = α·sem + (1-α)·kw     │
└─────────────────┬───────────────────────┘
                  │ top 10 candidates
                  ▼
┌─────────────────────────────────────────┐
│            Re-Ranker                    │
│  Coverage score + length score + MMR    │
└─────────────────┬───────────────────────┘
                  │ final 3-8 chunks
                  ▼
┌─────────────────────────────────────────┐
│       LLM Generator (OpenRouter)        │
│  Adaptive system prompt per query type  │
└─────────────────┬───────────────────────┘
                  │
                  ▼
         STRUCTURED ANSWER + SOURCES
```

---

## 📁 Project Structure

```
rag_system/
├── app.py                  # Streamlit UI
├── run_cli.py              # Command-line interface
├── requirements.txt
├── .env.example
│
├── rag/
│   ├── __init__.py
│   ├── pipeline.py         # Orchestrator — ties all modules together
│   ├── embedder.py         # SentenceTransformer + FAISS semantic search
│   ├── retriever.py        # Hybrid retrieval (semantic + TF-IDF)
│   ├── classifier.py       # Rule-based query type classifier
│   ├── ranker.py           # Re-ranking with coverage + MMR diversity
│   └── generator.py        # OpenRouter LLM generation + caching
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py    # PDF/TXT loading + text cleaning
│   └── chunking.py         # Sentence-aware chunking with overlap
│
├── data/                   # Drop your documents here
└── logs/                   # Auto-generated log files
```

---

## ⚙️ Setup

### 1. Clone / download the project

```bash
cd rag_system
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First run downloads the `all-MiniLM-L6-v2` model (~90 MB). Subsequent runs use the cached model.

### 4. Set your OpenRouter API key

```bash
# Linux / Mac
export OPENROUTER_API_KEY="sk-or-your-key-here"

# Windows (Command Prompt)
set OPENROUTER_API_KEY=sk-or-your-key-here

# Windows (PowerShell)
$env:OPENROUTER_API_KEY="sk-or-your-key-here"
```

Get a free key at [openrouter.ai](https://openrouter.ai).

---

## 🚀 Running the System

### Streamlit UI (recommended)

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Command-line Interface

```bash
# From a file
python run_cli.py --file data/paper.pdf --query "What are the key findings?"

# From pasted text
python run_cli.py --text "Your long document text here..." --query "Summarize this"

# Force a query type
python run_cli.py --file data/report.txt --query "Compare the methods" --type analytical

# JSON output
python run_cli.py --file data/report.txt --query "Who is the author?" --json

# Verbose (debug) mode
python run_cli.py --file data/report.txt --query "..." --verbose
```

---

## 🧩 Module Reference

### `utils/preprocessing.py`
Loads PDFs (via PyMuPDF) and TXT files. Cleans whitespace, normalises line endings, removes control characters.

### `utils/chunking.py`
Splits text into overlapping chunks preserving sentence boundaries. Default: 500 chars per chunk, 100 char overlap.

### `rag/embedder.py`
Encodes chunks with `all-MiniLM-L6-v2`, L2-normalises vectors, and stores in a FAISS `IndexFlatIP` for exact cosine similarity search.

### `rag/retriever.py`
Combines semantic FAISS scores and TF-IDF keyword scores (both normalised to [0,1]) with a configurable α weight. Default α = 0.6.

### `rag/classifier.py`
Rule-based regex classifier. Maps queries to:
- **factual** → top 3 chunks
- **summarization** → top 8 chunks
- **analytical** → top 5 diverse chunks (MMR)

### `rag/ranker.py`
Re-scores candidates using query token coverage + chunk length quality. Uses Maximal Marginal Relevance (MMR) for `analytical` queries to ensure context diversity.

### `rag/generator.py`
Calls OpenRouter's `/v1/chat/completions` with an adaptive system prompt per query type. Includes MD5-based in-memory caching for repeated queries.

### `rag/pipeline.py`
Orchestrates the full pipeline. `ingest_file()` / `ingest_text()` → `build()` → `query()`.

---

## 🔧 Configuration

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 500 | Target chunk character length |
| `chunk_overlap` | 100 | Character overlap between chunks |
| `semantic_weight` | 0.6 | α for hybrid scoring |
| `candidate_pool` | 10 | Candidates before re-ranking |
| `llm_model` | `mistralai/mistral-7b-instruct` | OpenRouter model |
| `temperature` | 0.2 | LLM creativity |
| `use_cache` | True | Cache repeated queries |

---

## 💡 Example Queries

| Query | Expected Type | Chunks |
|---|---|---|
| "What is the capital of France?" | factual | 3 |
| "Who wrote this paper?" | factual | 3 |
| "Summarize the document" | summarization | 8 |
| "What are the main topics covered?" | summarization | 8 |
| "Why did the authors choose this method?" | analytical | 5 (diverse) |
| "Compare the two approaches discussed" | analytical | 5 (diverse) |

---

## 🎓 College Project Talking Points

1. **Hybrid Retrieval** — explains why pure semantic or pure keyword search each have blind spots
2. **Query Classification** — shows how intent-aware systems outperform one-size-fits-all retrieval
3. **Re-ranking** — demonstrates the gap between initial retrieval and context quality
4. **MMR Diversity** — addresses the "echo chamber" problem in retrieved context
5. **Adaptive Prompting** — shows how system prompts should vary with task type

---

## 📄 License

MIT — free for academic and personal use.
