# Specification: SQL-to-Column-Level-Lineage Graph Using LLM

**Version 1.0**  
**Date:** 2026-05-29  
**Status:** Draft  

---

## 1. Introduction

### 1.1 Purpose
This document specifies a system that converts complex SQL queries into a **column-level lineage graph**. The graph must explicitly show:
- How each **output column** is derived from source columns (including expressions, aggregates, window functions).
- Which **filters** (WHERE, HAVING, JOIN … ON, etc.) affect the output.
- How **tables are joined** and which columns participate in those joins.

The system leverages a Large Language Model (LLM) for the semantic heavy‑lifting while relying on deterministic components for parsing, validation, and graph construction.

### 1.2 Scope
- **Input:** A single SQL `SELECT` statement (possibly with CTEs, subqueries, set operations) and, optionally, the schema of referenced tables/views.
- **Output:** A directed property graph representing column-level lineage, delivered in a machine-readable format (JSON graph structure) and optionally in visual formats (DOT, Mermaid).
- **Covered SQL constructs:**
  - SELECT lists with aliases, expressions, aggregates, window functions, CASE.
  - JOINs (INNER, LEFT/RIGHT/FULL OUTER, CROSS) with explicit ON conditions.
  - WHERE, GROUP BY, HAVING, QUALIFY (if dialect supports).
  - Subqueries in FROM, SELECT, and WHERE.
  - Common Table Expressions (CTEs, WITH clause).
  - Set operations (UNION, INTERSECT, EXCEPT) – treated as independent branches.
- **Out of scope (v1):** DML, DDL, recursive CTE graph expansion, stored procedures, dynamic SQL.

### 1.3 Definitions
- **Column node:** A fully qualified reference `table_alias.column_name`.
- **Output column node:** A named result column (`alias` from the outermost SELECT).
- **Filter node:** A logical condition that gates rows. It is connected to columns used in that condition.
- **Join edge:** Connects two columns that are equated in a join condition (or, for non-equi joins, connects the join condition itself to all referenced columns).

---

## 2. System Overview

### 2.1 High-Level Architecture
The system is composed of four main modules:

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌─────────────┐
│ SQL Parser  │────▶│ LLM Lineage      │────▶│ Graph Builder    │────▶│ Validator   │
│ (sqlglot)   │     │ Extractor (GPT)  │     │ (Python)         │     │ & Export    │
└─────────────┘     └──────────────────┘     └──────────────────┘     └─────────────┘
       │                     ▲                        │
       └─── AST (optional) ──┘                        ▼
                                            [graph JSON / Mermaid / DOT]
```

* **SQL Parser (optional but recommended):** Converts raw SQL text into a canonical Abstract Syntax Tree (AST). If not used, the LLM receives the raw SQL directly.
* **LLM Lineage Extractor:** The core intelligence. Given the SQL (and optionally the AST) plus table schemas, the LLM produces a structured JSON document that describes all output columns, filters, and joins.
* **Graph Builder:** Transforms the LLM’s JSON into a directed graph using a library like NetworkX. Applies deterministic rules (e.g., connecting filter nodes to output columns).
* **Validator:** Checks the graph for completeness (no dangling references) and optionally cross‑references with database execution plans.

### 2.2 Data Flow
1. Client submits SQL and (optionally) a table schema definition.
2. SQL is parsed into an AST (if parser enabled) to normalise dialect quirks.
3. The LLM receives a prompt containing the SQL, the AST (or a structured summary), and the schema. It returns a JSON object adhering to the specification below.
4. The Graph Builder reads the JSON, creates nodes and edges, and serialises the graph.
5. The Validator runs sanity checks and returns the final graph object (or an error report).

---

## 3. Detailed Design

### 3.1 SQL Preprocessing and Parsing
- **Library:** `sqlglot` (supports 20+ dialects, produces a uniform AST).
- **Input:** Raw SQL string.
- **Output:** A dictionary/JSON representation of the AST.
  - The AST must preserve table aliases, column references (resolved or qualified), and expression trees.
  - For consumption by the LLM, we extract a **simplified structure**:
    ```json
    {
      "select": { "columns": [...], "aliases": [...] },
      "from": [{ "table": "...", "alias": "..." }],
      "joins": [{ "type": "LEFT", "right_table": "...", "alias": "...", "on": "..." }],
      "where": "...",
      "group_by": [...],
      "having": "...",
      "ctes": [...]
    }
    ```
    This keeps the prompt concise while conveying all necessary structural information.

**Fallback:** If no parser is used, the LLM prompt includes only the original SQL. Accuracy may degrade for exotic dialects or `SELECT *`.

### 3.2 LLM-Based Lineage Extraction

#### 3.2.1 Prompt Design
The prompt is engineered to produce a strict JSON output. It includes:
- **System message:** Roles and instructions.
- **Table schema:** A compact description of all referenced tables (name, columns, types).
- **Simplified query structure** (if parser used) **or** raw SQL.
- **Chain-of-thought instruction:** “First, list all table aliases and their source tables. Then analyse the SELECT list column by column. Then extract filters and joins.”
- **Few-shot examples:** 2–3 pairs of (SQL, expected JSON) covering CTEs, aggregates, CASE, and multiple joins.
- **Output format specification** (reiterated in the system message).

**LLM call configuration:**
- Model: GPT-4 (or equivalent) with `response_format: { type: "json_object" }` (OpenAI) or strict JSON mode.
- Temperature: 0 (deterministic).
- Max tokens: 4,096 (adjust based on query complexity).

#### 3.2.2 Output Data Model (JSON Schema)
The LLM must return a JSON object with the following structure:

```json
{
  "ctes": [
    {
      "alias": "c",
      "output_columns": [ ... ],   // same structure as top-level output_columns
      "filters": [ ... ],
      "joins": [ ... ]
    }
  ],
  "output_columns": [
    {
      "alias": "total_price",
      "expression": "o.quantity * p.price",
      "dependencies": [
        { "table_alias": "o", "column": "quantity" },
        { "table_alias": "p", "column": "price" }
      ],
      "aggregate": false,
      "window_function": false
    }
  ],
  "filters": [
    {
      "clause": "WHERE",
      "condition": "o.order_date > '2025-01-01' AND p.active = true",
      "columns_used": [
        { "table_alias": "o", "column": "order_date" },
        { "table_alias": "p", "column": "active" }
      ]
    },
    {
      "clause": "HAVING",
      "condition": "SUM(amount) > 100",
      "columns_used": [
        { "table_alias": null, "column": "amount" }
      ]
    }
  ],
  "joins": [
    {
      "type": "LEFT",
      "left_alias": "o",
      "right_alias": "p",
      "condition": "o.product_id = p.id",
      "join_columns": [
        { "table_alias": "o", "column": "product_id" },
        { "table_alias": "p", "column": "id" }
      ]
    }
  ]
}
```

**Constraints:**
- `output_columns.dependencies` must list every column that directly contributes to the value of the expression (expanding `*` explicitly, including both branches of CASE, all arguments of functions, etc.).
- For aggregates, dependencies include the raw column(s) **and** every column in GROUP BY (the LLM is instructed to add them explicitly; if not, the Graph Builder can infer them).
- `filters.columns_used` lists every column referenced in the filter condition, including those inside functions like `COALESCE(t1.col, t2.col)`.
- `joins.join_columns` always contains exactly two entries (left key, right key). For composite keys, the LLM must flatten them into multiple join objects or we define a composite‑key extension. v1 assumes single‑column equality conditions.

**Formal JSON Schema (subset):**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["output_columns", "filters", "joins"],
  "properties": {
    "ctes": { "type": "array", "items": { "$ref": "#" } },
    "output_columns": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["alias", "expression", "dependencies"],
        "properties": {
          "alias": { "type": "string" },
          "expression": { "type": "string" },
          "dependencies": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["table_alias", "column"],
              "properties": {
                "table_alias": { "type": "string" },
                "column": { "type": "string" }
              }
            }
          }
        }
      }
    },
    "filters": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["clause", "condition", "columns_used"],
        "properties": {
          "clause": { "enum": ["WHERE", "HAVING", "ON"] },
          "condition": { "type": "string" },
          "columns_used": {
            "type": "array",
            "items": { "$ref": "#/properties/output_columns/items/properties/dependencies/items" }
          }
        }
      }
    },
    "joins": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["type", "left_alias", "right_alias", "condition", "join_columns"],
        "properties": {
          "type": { "type": "string" },
          "left_alias": { "type": "string" },
          "right_alias": { "type": "string" },
          "condition": { "type": "string" },
          "join_columns": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": { "$ref": "#/properties/output_columns/items/properties/dependencies/items" }
          }
        }
      }
    }
  }
}
```

#### 3.2.3 LLM Integration
- Implement a thin abstraction layer so the LLM backend can be swapped (OpenAI, Azure, self‑hosted).
- The LLM client will:
  1. Assemble the prompt from input SQL, schema, and (optionally) parsed AST.
  2. Call the model with structured output enabled.
  3. Parse the JSON response and validate it against the schema.
  4. If validation fails, retry once with an error description appended to the prompt.

### 3.3 Graph Construction

#### 3.3.1 Graph Data Model
The graph uses **nodes** and **edges** with type labels.

| Node type        | ID convention                  | Attributes       |
|------------------|--------------------------------|------------------|
| `source_column`  | `table_alias.column`           | table, column    |
| `output_column`  | `query_alias.output_column`    | alias, expression|
| `filter`         | `filter_<hash>`                | clause, condition|
| `join`           | `join_<hash>`                  | join_type        |

| Edge type        | Source → Target                  | Meaning |
|------------------|----------------------------------|---------|
| `DERIVED_FROM`   | `source_column` → `output_column`| Direct lineage |
| `FILTERED_BY`    | `filter` → `output_column`       | Filter applies to result (connected to every output column) |
| `USES_COLUMN`    | `filter` → `source_column`       | Filter references this column |
| `JOINS_ON`       | `source_column` ↔ `source_column`| Two columns equated in a join |
| `GROUPED_BY`     | `output_column` (aggregate) → `source_column` (group key) | Implicit dependency (optional but recommended) |

**Implementation:** Use NetworkX directed `MultiDiGraph`. Node attributes store the type and other metadata. Edges are labelled with the `edge_type`.

#### 3.3.2 Mapping Rules
1. **Output columns:**  
   For each `output_col` in the LLM JSON:
   - Create an `output_column` node named `output.<alias>`.
   - For each dependency `dep`, create or locate a `source_column` node `dep.table_alias.dep.column` and add a `DERIVED_FROM` edge from `dep.table_alias.dep.column` to `output.<alias>`.

2. **Filters:**  
   For each filter:
   - Create a `filter` node with a unique ID (hash of clause+condition).
   - For each column in `columns_used`, create/ensure a `source_column` node and add a `USES_COLUMN` edge from the filter node to that source column.
   - For **every** `output_column` node in the current query (or the outermost query), add a `FILTERED_BY` edge from the filter node to the output column node. This reflects that the filter gates the existence of the output rows.

   *Exception:* Filters from `ON` clauses of joins: these are captured in join edges, but we may optionally create filter nodes as well. v1 treats them as join predicates and does not create separate filter nodes; instead the join node connects the two key columns.

3. **Joins:**  
   For each join:
   - Create two `source_column` nodes for the left and right keys.
   - Add a bidirectional `JOINS_ON` edge between them (or two directed edges). Label it with the join type.

4. **CTEs:**  
   Recursively process each CTE as if it were a standalone query, then treat its `output_columns` as `source_column` nodes for the main query. Their IDs are prefixed with the CTE alias: `cte_alias.column`.

5. **Aggregates and GROUP BY:**  
   The LLM is encouraged to include GROUP BY columns in the dependency list of aggregate outputs. If it does not, the Graph Builder will detect that the query has a GROUP BY and an aggregate, and add `GROUPED_BY` edges from the aggregate output node to each GROUP BY column node (if not already present via `DERIVED_FROM`).

6. **Window functions:**  
   Treat `PARTITION BY` and `ORDER BY` columns as dependencies of the output column. The LLM must include them. No extra processing needed.

### 3.4 Validation and Error Handling
- **JSON Schema validation:** The LLM’s output is checked against the JSON Schema. If invalid:
  - Log the error and the raw JSON.
  - Retry the LLM call once with the schema violation message appended to the prompt.
  - If still invalid, return an error to the caller.
- **Column existence check:** For every column reference in the graph, verify that it exists in the provided schema (if schema was given). Flag unknown references.
- **Referential integrity:** Ensure all edges point to existing nodes. Any dangling references (e.g., a dependency on `o.unknown_col`) are reported as warnings.
- **Optional cross‑check:** Run `EXPLAIN` on the original SQL against a database (if available). Compare the list of columns mentioned in the plan with the leaf nodes of the graph. Significant differences trigger an alert.

### 3.5 API Design (when deployed as a service)
- **Endpoint:** `POST /lineage`
- **Request body:**
  ```json
  {
    "sql": "SELECT ...",
    "dialect": "postgres",
    "schema": {
      "tables": [
        {
          "name": "orders",
          "alias": "o",
          "columns": [{"name": "id", "type": "int"}, ...]
        }
      ]
    },
    "include_visualization": false
  }
  ```
- **Response (success):**
  ```json
  {
    "graph": { ... },           // networkx node-link data
    "visualization": {
      "mermaid": "flowchart TD ...",
      "dot": "digraph { ... }"
    },
    "warnings": [...]
  }
  ```
- **Response (error):**
  ```json
  {
    "error": "LLM output validation failed",
    "details": "..."
  }
  ```

---

## 4. Implementation Guidelines

### 4.1 Technology Stack
- **Backend:** Python 3.11+
- **Parser:** `sqlglot` (optional but recommended)
- **LLM client:** `openai` Python library (or `langchain` if additional prompting patterns needed)
- **Graph library:** `networkx` (for construction and analysis), export to `pygraphviz` or `mermaid` as needed.
- **API framework:** FastAPI (if a web service is required)
- **Validation:** `jsonschema` library.

### 4.2 Code Organization
```
src/
  parser.py           # SQL parsing and AST simplification
  llm_extractor.py    # Prompt assembly, LLM call, response parsing
  graph_builder.py    # JSON → NetworkX graph conversion
  validator.py        # Schema check, column existence, cross‑check
  api.py              # (optional) FastAPI endpoints
  schemas.py          # JSON Schema definitions
  templates/          # Prompt templates (few-shot examples)
```

### 4.3 Error Recovery
- LLM call failures: exponential backoff, max 3 retries.
- Incomplete LLM output (truncated JSON): attempt to fix by closing brackets, or retry with a lower token count.
- Unsupported SQL features: catch during parsing, return a clear “not supported” message.

---

## 5. Testing Strategy

### 5.1 Unit Tests
- **Parser:** Verify AST simplification for various SQL constructs.
- **LLM prompt:** Mock LLM responses; check that prompt assembly inserts schema and AST correctly.
- **Graph builder:** Given a known JSON, verify the resulting graph structure (nodes, edges).
- **Validator:** Test against JSON with missing references, unknown columns.

### 5.2 Integration Tests
- Run a suite of 50+ SQL queries (simple selects, nested subqueries, CTEs, complex joins, aggregates, windows) against the full pipeline with a real LLM.
- For each, manually inspect the resulting graph and compare with expected lineage.
- **Golden set:** Store expected graphs (or their fingerprints) to detect regressions.

### 5.3 Edge Cases
- `SELECT *` expansion when schema is provided.
- Self‑joins (same table, different aliases).
- Union queries: treat each branch separately and merge output columns.
- CASE WHEN inside aggregate.
- Subquery in WHERE (e.g., `EXISTS`): the LLM must identify the correlated columns.

---

## 6. Deployment Considerations
- LLM costs: one moderate‑complexity query consumes ~500–1000 tokens. Plan for rate limits and billing.
- Latency: pipeline takes 2–5 seconds (parser <0.1s, LLM 1–3s, graph building <0.2s). Acceptable for interactive use.
- Caching: identical SQL + schema inputs can be cached. Use a hash of the input as cache key.
- Security: SQL statements might contain sensitive table names or logic. Do not log raw SQL in plaintext unless necessary; use redaction.

---

## 7. Appendix: Example Input/Output

**Input SQL:**
```sql
WITH recent_orders AS (
  SELECT customer_id, SUM(amount) AS total
  FROM orders
  WHERE order_date > '2025-01-01'
  GROUP BY customer_id
)
SELECT c.name, r.total
FROM customers c
JOIN recent_orders r ON c.id = r.customer_id
WHERE c.active = true
```

**LLM output (abbreviated):**
```json
{
  "ctes": [
    {
      "alias": "recent_orders",
      "output_columns": [
        { "alias": "customer_id", "dependencies": [{"table_alias": "orders", "column": "customer_id"}] },
        { "alias": "total", "expression": "SUM(amount)", "dependencies": [
            {"table_alias": "orders", "column": "amount"},
            {"table_alias": "orders", "column": "customer_id"}  // GROUP BY
          ],
          "aggregate": true
        }
      ],
      "filters": [
        { "clause": "WHERE", "columns_used": [{"table_alias": "orders", "column": "order_date"}] }
      ],
      "joins": []
    }
  ],
  "output_columns": [
    {
      "alias": "name",
      "dependencies": [{"table_alias": "c", "column": "name"}]
    },
    {
      "alias": "total",
      "dependencies": [{"table_alias": "r", "column": "total"}]
    }
  ],
  "filters": [
    { "clause": "WHERE", "columns_used": [{"table_alias": "c", "column": "active"}] }
  ],
  "joins": [
    {
      "type": "INNER",
      "left_alias": "c",
      "right_alias": "r",
      "condition": "c.id = r.customer_id",
      "join_columns": [
        {"table_alias": "c", "column": "id"},
        {"table_alias": "r", "column": "customer_id"}
      ]
    }
  ]
}
```

**Resulting graph (conceptual):**
- `output.name` ← `DERIVED_FROM` ← `c.name`
- `output.total` ← `DERIVED_FROM` ← `r.total` (which itself derives from `orders.amount` and `orders.customer_id` inside the CTE)
- `filter_<hash>` (`c.active = true`) → `USES_COLUMN` → `c.active`; and `FILTERED_BY` → `output.name`, `output.total`
- `c.id` ↔ `JOINS_ON` ↔ `r.customer_id`
- Additional internal filter inside CTE connected to `orders.order_date` and to the CTE’s output columns.

This specification provides a complete blueprint. Implementation can begin with the JSON schema and a prototype prompt, iterating on accuracy with the test suite.