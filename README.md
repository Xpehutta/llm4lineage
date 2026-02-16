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

## Detailed connections


#### 1. `SQLLineageExtractor`
- Wraps `HuggingFaceEndpoint` + `ChatHuggingFace`.  
- Its `human_prompt_template` is the **prompt being optimised**.  
- Provides the `extract()` method used by both the app and the agent.

#### 2. `SQLLineageValidator`
- Compares the extractor’s output against a **ground‑truth result**.  
- Returns **F1 score, precision, recall** – metrics that the agent uses to decide if the prompt is good enough.

#### 3. `HuggingFaceSQLLineageAgent`
- **Reuses the same `SQLLineageExtractor`** (or creates one) to perform extraction during optimisation.  
- Maintains **its own** `ChatHuggingFace` instance (with identical parameters) to generate improved prompts.  
- Implements a **reflexion loop** as a LangGraph workflow:
  - `validate_node`: runs extraction + validation, records F1.
  - `reflect_node`: feeds errors and current prompt to the chat model → produces a refined prompt.
- Stops when F1 = 1.0 or max iterations reached.

#### 4. Streamlit app (`Web/app.py`)
- Instantiates `SQLLineageExtractor` (cached).  
- For the batch tab, stores every extracted statement together with its **full SQL** and lineage.  
- When a user clicks a target table, the lookup input is populated and the downstream/upstream graph is drawn.  
- The app **does not** directly use the agent – the agent is meant for **offline prompt optimisation**.

This modular design keeps the interactive web interface separate from the prompt‑optimisation logic, making both parts easier to maintain and extend.

## 🧪 Example Workflows

### Workflow 1: Optimise an extraction prompt (using the agent)

```python
from Classes.model_classes import SQLLineageExtractor
from Classes.prompt_refiner import HuggingFaceSQLLineageAgent
import os

# Create extractor
extractor = SQLLineageExtractor(
    model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
    provider="scaleway",
    hf_token=os.environ["HF_TOKEN"]
)

# Create agent, passing the extractor (so they share the same model)
agent = HuggingFaceSQLLineageAgent(
    model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
    provider="scaleway",
    hf_token=os.environ["HF_TOKEN"],
    extractor=extractor
)

# Ground truth (what the correct lineage should be)
expected = {
    "target": "analytics.sales_summary",
    "sources": ["products.raw_data", "sales.transactions"]
}

# Run optimisation
result = agent.optimize_prompt_sync(
    sql="INSERT INTO analytics.sales_summary ...",
    expected_result=expected,
    output_file="optimisation_log.json",
    verbose=True
)

print("Best prompt:\n", result["optimized_prompt"])
print("F1 score:", result["f1_score"])
```

### Workflow 2: Interactive Exploration in the Streamlit App

This workflow guides you through using the **Table Lineage (Batch)** tab of the web interface to explore lineage across multiple SQL scripts.

#### Prerequisites
- You have started the app with `streamlit run Web/app.py`
- Your Hugging Face token is entered in the sidebar (or set in `.env`)

---

#### Step‑by‑Step

1. **Open the “Table Lineage (Batch)” tab**  
   Click the second tab at the top of the page.

2. **Upload one or more SQL files**  
   - Drag & drop `.sql` or `.txt` files into the file uploader, or click “Browse files”.  
   - Files may contain **multiple SQL statements** separated by semicolons (`;`).  
   - Example file content:
     ```sql
     INSERT INTO target1 SELECT * FROM source1;
     INSERT INTO target2 SELECT * FROM source2;
     ```

3. **Processing**  
   The app will:
   - Split each file into individual statements.
   - Run lineage extraction on each statement using the `SQLLineageExtractor`.
   - Display a progress bar and show any errors per statement.
   - Store results in the session.

4. **View the extracted lineage overview**  
   After processing, you’ll see a table with:
   - **File** name
   - **Statement** number
   - **Target** table (clickable button)
   - **Sources** count

   ![Overview table](https://via.placeholder.com/800x200?text=Extracted+Lineage+Overview)

5. **Click on a target table**  
   - Clicking any target button automatically fills the “Look up a table” input field.  
   - The page scrolls to the lookup section and displays:
     - How many times the table appears as **Target** and as **Source**.
     - Expandable sections listing every occurrence.

6. **Explore occurrences**  
   - **As Target**: For each occurrence you see the source tables and the **full SQL** in a code block (with a copy button).  
   - **As Source**: For each occurrence you see the target table and the full SQL.

7. **Visualise the lineage graph**  
   Below the occurrence lists, an interactive **Graphviz graph** shows:
   - **Upstream sources** (tables that feed into the selected table) on the left.
   - **Downstream targets** (tables that use the selected table as a source) on the right.
   - The central node is your selected table.

   ![Lineage graph](https://via.placeholder.com/800x400?text=Lineage+Graph)

8. **Copy any SQL**  
   - Every displayed SQL code block has a **copy icon** in the top‑right corner – click it to copy the entire statement to your clipboard.

9. **Clear the session**  
   - Use the “Clear all stored lineage results” button at the bottom to reset and start fresh.

---

#### Tips
- The **Single Query Lineage** tab works the same way, but only for one SQL at a time – you’ll get immediate JSON and a graph.
- If you have many statements, the graph limits nodes to 15 for readability, with a `…` indicator if more exist.
- The app caches the extractor, so repeated lookups are fast.


