# SQL Lineage Tool – Prompt Optimisation & Lineage Extraction

A powerful tool that uses **LangChain + Hugging Face** to extract source‑to‑target lineage from SQL statements and automatically **optimise the extraction prompt** via a reflexion agent.  
The project includes a **Streamlit web interface** for interactive exploration and batch processing.

---

## ✨ Features

- **Single‑query lineage extraction** – get target table and source tables as JSON + graph.
- **Batch processing** – upload multiple `.sql` / `.txt` files (each may contain several statements, split by `;`).
- **Table‑centric view** – enter a table name to see all queries where it is target or source, plus a dependency graph.
- **Prompt optimisation agent** – iteratively improves the extraction prompt using a **reflexion loop** (F1‑score guided) and the same LLM as the extractor.
- **LangChain Hugging Face integration** – uses `ChatHuggingFace` + `HuggingFaceEndpoint` for both extraction and reflection.
- **All outputs saved** – optimisation history (prompts, validation results) can be written to JSON.

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd sql-lineage-tool
```

### 2. Install dependencies

We recommend using uv (fast Python package installer):

```bash
uv venv
source .venv/bin/activate      # macOS/Linux
# or .venv\Scripts\activate    # Windows
uv pip install -e .
```

### 3. Set your Hugging Face token

Create a .env file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
```

Or export it as an environment variable:

```bash
export HF_TOKEN=your_huggingface_token_here
```

## 🚀 Running the Streamlit App

The main entry point for interactive use is Web/app.py.

```bash
streamlit run Web/app.py
```

The app will open in your browser at http://localhost:8501.

### App layout

Left sidebar: configure the Hugging Face model, provider, token, and extraction parameters.
Two main tabs:

Single Query Lineage – paste one SQL, get JSON + graph.
Table Lineage (Batch) – upload files, see an overview, click a target to explore.

## 📚 How the Classes Are Connected

The project is structured into several classes, each with a clear responsibility. Below is a simplified diagram:

```text
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit Frontend                         │
│                         (Web/app.py)                            │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SQLLineageExtractor                         │
│  (langchain_huggingface.ChatHuggingFace + HuggingFaceEndpoint)  │
│  - _create_prompt_template()  → human_prompt_template           │
│  - _create_chain()            → prompt │ model │ parser         │
│  - extract(sql)               → {"target": ..., "sources": ...} │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SQLLineageValidator                         │
│  (from validation_classes)                                      │
│  - run_comprehensive_validation(extractor, sql, expected)       │
│    → returns {"status": "SUCCESS"/"FAILED", "metrics": {...}}   │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                  HuggingFaceSQLLineageAgent                     │
│  (prompt optimisation using reflexion)                          │
│  - owns an extractor (for lineage extraction)                   │
│  - owns a separate ChatHuggingFace (for reflection)             │
│  - create_workflow() → LangGraph with validate + reflect nodes  │
│  - optimize_prompt_sync() → best prompt & F1 history            │
└─────────────────────────────────────────────────────────────────┘
```
