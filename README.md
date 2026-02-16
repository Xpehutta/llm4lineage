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


### 2. Install dependencies

We recommend using uv (fast Python package installer):

```bash
uv venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt
If you prefer pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
