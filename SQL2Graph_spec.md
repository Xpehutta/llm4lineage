# Specification: SQL-to-Transformation-Lineage Graph

**Version 2.1**
**Status:** Draft

---

## Implementation Status in This Repo

The repo implements a **simplified column-level profile** of this specification in
`Classes/sql2graph_classes.py`. The v2.1 target model (RowSet nodes, transformation
nodes, full IR) is the roadmap; the current pipeline covers the following subset:

| Spec concept | Current implementation |
|---|---|
| Source column nodes (§4.1) | `node_type: source_column` (`alias.column`) |
| Output / target column nodes (§4.1) | `node_type: output_column` (`output.<alias>`) |
| Filter nodes (§4.1) | `node_type: filter` with `clause` + `condition` |
| `VALUE_FLOW` (§4.2) | `edge_type: DERIVED_FROM` |
| `FILTER_CONDITION` + row gating (§4.2) | `USES_COLUMN` + `FILTERED_BY` |
| `JOIN_KEY` (§4.2) | `JOINS_ON` (bidirectional between key columns) |
| `GROUPING_KEY` (§4.2) | `GROUPED_BY` (aggregate output → group column) |
| CTE column passthrough (§7.9 / §8.3) | recursive CTE scopes + `link_cte_aliases()` |
| Transitive value lineage (§7.10) | `materialize_transitive_derived_from()` |
| INSERT / CTAS detection (§2) | `statement_type` + `target_table` in parser output |
| Graph metadata (§5) | `metadata` on pipeline graph payload |
| LLM IR (§6) | simplified extraction JSON (`output_columns`, `filters`, `joins`, `ctes`) |

**Not yet implemented:** RowSet nodes, `ROW_FLOW_IN` / `ROW_FLOW_OUT`, transformation /
aggregate / window / union operator nodes, expression IR trees, MERGE/UPDATE lineage,
and deterministic IR-to-graph builder for the full v2.1 edge set.

Primary entry point: `SQL2GraphPipeline.run()`. Demo notebook: `SQL2Graph.ipynb`.

---

## 1. Purpose

This specification defines a system that converts SQL queries and ETL statements into a **directed lineage graph**. The graph captures:

1. Value lineage – what contributes to a column’s value  
2. Row lineage – what determines row existence and cardinality  
3. Transformation lineage – expressions, casts, functions, aggregates  
4. Structural lineage – the flow of row sets through operators (filters, joins, unions, windows, CTEs)  
5. Join semantics – key relationships and filter conditions  
6. Aggregation semantics – grouping keys vs. value inputs  
7. Window-function semantics – partitioning and ordering  
8. Set-operation semantics – UNION, UNION ALL, INTERSECT, EXCEPT  
9. Insert-target lineage – how source data maps into target tables  

The resulting graph must accurately represent **data movement** from source tables to target tables, suitable for data-warehouse ETL, Data Vault transformations, dimensional loads, and complex CTE pipelines.

---

## 2. Supported SQL

### Read Queries

```sql
SELECT ...
```

### ETL Statements

```sql
INSERT INTO ... SELECT ...
```

```sql
CREATE TABLE AS SELECT ...
```

```sql
MERGE INTO ...  (v2 only supports INSERT portion)
```

*UPDATE / DELETE lineage may be added in a future version.*

---

## 3. Core Principles

The graph must **never** collapse the following concepts into a single edge:

| Concept                | Meaning                              |
|------------------------|--------------------------------------|
| Value Lineage          | How a value is computed              |
| Row Lineage            | Which rows flow into an operation    |
| Transformation Logic   | The specific functions / expressions |
| Structural Flow        | Sequence of SQL operators            |

Each must be represented with distinct node types and edge types.

---

# 4. Graph Model

## 4.1 Node Types

### Source Table
Represents a base table or view.
```json
{
  "id": "orders",
  "type": "table"
}
```

### Source Column
A column of a source table.
```json
{
  "id": "orders.amount",
  "type": "column"
}
```

### Target Table
An insert target.
```json
{
  "id": "d_agr_cred_dmcl_attr",
  "type": "target_table"
}
```

### Target Column
A column of a target table.
```json
{
  "id": "d_agr_cred_dmcl_attr.attr_name",
  "type": "target_column"
}
```

### CTE
A common table expression.
```json
{
  "id": "cte_recent_orders",
  "type": "cte"
}
```

### CTE Column
An output column of a CTE.
```json
{
  "id": "cte_recent_orders.total",
  "type": "cte_column"
}
```

### RowSet
An intermediate result set that flows between operators. Every operator that produces or consumes rows has an associated RowSet node.  
- A CTE body produces a RowSet.  
- A FROM clause produces a RowSet.  
- A filter (WHERE) consumes one RowSet and produces another.  
- A JOIN consumes two RowSets and produces one.  
- A UNION consumes multiple RowSets and produces one.  
- An aggregate/window consumes one RowSet and produces one.  
- A SELECT list finally consumes the last RowSet and maps its columns to output columns (or target columns).

```json
{
  "id": "rs_filtered_orders",
  "type": "rowset"
}
```

### Expression / Transformation Node
Represents a scalar expression. Examples: arithmetic, CASE, CAST, COALESCE, CONCAT, UUID_HASH, ABS, UPPER, arithmetic operations.

```json
{
  "id": "transform_001",
  "type": "transformation",
  "function": "ABS",
  "expression_text": "ABS(UUID_HASH(attr_val_uuid))"
}
```

Complex nested expressions are broken down: each function/operator is a separate transformation node, with VALUE_FLOW edges chaining them.

### Constant Node
A literal or constant expression.
```json
{
  "id": "const_agr_qlty_cat_type_id",
  "type": "constant",
  "value": "agr_qlty_cat_type_id"
}
```

### Aggregate Node
Represents an aggregate function over a group. Contains the aggregate function name (SUM, COUNT, etc.).

```json
{
  "id": "agg_sum_amount",
  "type": "aggregate",
  "function": "SUM"
}
```

### Window Node
Represents a window function call. The window specification (PARTITION BY, ORDER BY) is captured via edges, not in the node itself.

```json
{
  "id": "window_row_number",
  "type": "window",
  "function": "ROW_NUMBER"
}
```

### Filter Node
Represents a WHERE or HAVING clause. It consumes a RowSet and produces a new RowSet.

```json
{
  "id": "filter_001",
  "type": "filter",
  "clause_type": "WHERE",
  "condition_text": "status = 'ACTIVE'"
}
```

### Join Node
Represents a JOIN operation. Consumes two RowSets and produces one.

```json
{
  "id": "join_001",
  "type": "join",
  "join_type": "INNER"
}
```

### Union Node
Represents a set operation (UNION, UNION ALL, INTERSECT, EXCEPT).

```json
{
  "id": "union_001",
  "type": "union",
  "operation": "UNION ALL"
}
```

### Branch Node
A Branch is a child RowSet that flows into a union (or a set operation). Each arm of the union is a separate Branch, which itself is a RowSet.

Technically, a Branch is a RowSet – we can use the `rowset` type with a property `{ "branch_of": "union_001" }`. To keep the model simple, we treat each branch as a regular RowSet.

### Row Generator Node
Represents an operation that creates new rows without an input rowset, such as `UNNEST(...)`, `GENERATE_SERIES()`, `VALUES(...)`, or set-returning functions. This node produces a RowSet.

```json
{
  "id": "rowgen_001",
  "type": "row_generator",
  "function": "UNNEST",
  "expression": "array_column"
}
```

---

## 4.2 Edge Types

### VALUE_FLOW
Indicates that a source contributes to the value of a target. This is the primary edge for expression trees, aggregate results, window results, and direct column mappings.

*Allowed sources: column, cte_column, constant, transformation, aggregate, window*  
*Allowed targets: transformation, aggregate, window, target_column, cte_column*

*Properties:*
- `expression_position` (optional) – for functions with multiple arguments, the 0-based index.

---

### PIPELINE
Connects operators in execution order. It shows how a RowSet flows from one operator to the next. This is **structural lineage**.

*Allowed sources: rowset (or a source table implicitly producing a rowset)*  
*Allowed targets: filter, join, aggregate, window, union, row_generator (output rowset)*

Direction: from the consuming RowSet to the operator, and from the operator to the resulting RowSet. However, to keep edges simple, we use two edges:

- **INPUT** edge: from a RowSet to an operator (e.g., `rs_orders → filter`)
- **OUTPUT** edge: from an operator to a RowSet (e.g., `filter → rs_filtered_orders`)

We combine these under a unified edge type `PIPELINE` with a property `direction: "input" | "output"`.

Alternatively, we can define two distinct edge types: `OP_INPUT` and `OP_OUTPUT`. For clarity, we’ll use:

- **ROW_FLOW_IN** – from a RowSet to an operator that consumes it.
- **ROW_FLOW_OUT** – from an operator to the RowSet it produces.

---

### FILTER_CONDITION
Connects a column or expression to a Filter node, indicating that the filter condition references that column. This is a **row‑level dependency**, not a value flow.

*Source: column, transformation, constant*  
*Target: filter*

*Properties:*
- `condition_role` – e.g., "left_operand", "right_operand", "in_list"

---

### JOIN_KEY
Connects a column from one side of a join to the Join node, denoting an equi‑join condition.

```text
orders.customer_id → JOIN_KEY → join_001
customers.id       → JOIN_KEY → join_001
```

*Source: column, cte_column*  
*Target: join*

*Properties:*
- `side` – "left" or "right"

---

### JOIN_FILTER
Connects a column or expression to a Join node for a non‑equi join condition (e.g., `ON a.x > b.y`). This indicates that the column is used in the join predicate but not as an equality key.

*Source: column, transformation, constant*  
*Target: join*

*Properties:*
- `expression_role` – describes how the column participates

---

### GROUPING_KEY
Connects a column (or expression) to an Aggregate node, indicating that it is part of the GROUP BY clause.

*Source: column, cte_column*  
*Target: aggregate*

This edge **does not** represent a value contribution; it defines grain.

---

### PARTITION_KEY
Connects a column to a Window node for PARTITION BY.

*Source: column, cte_column*  
*Target: window*

---

### ORDERING_KEY
Connects a column to a Window node for ORDER BY.

*Source: column, cte_column*  
*Target: window*

---

### UNION_MEMBER
Connects a Branch RowSet to a Union node, indicating that this branch contributes rows.

*Source: rowset (the branch)*  
*Target: union*

*Properties:*
- `branch_index` – position of the branch (0-based)

---

### INSERTS_INTO
Connects a Target Column to its Target Table.

*Source: target_column*  
*Target: target_table*

---

### CTE_REFERENCES
Connects a CTE node to the query block that uses it. However, since CTE usage is just a table reference, we model this by creating a RowSet for the CTE result and then using `PIPELINE` edges to feed into joins, filters, etc.

A CTE’s body (the SELECT inside the CTE) will be fully modeled; the CTE node acts as a container. We add a `CTE_PRODUCES` edge from the CTE to its output RowSet, and then any reference to that CTE is a RowSet that can be used as input to other operators.

---

### TRANSITIVE_VALUE_FLOW (optional)
Not a separate edge type; the graph builder should generate **VALUE_FLOW** edges that skip intermediate nodes where appropriate (e.g., direct source-to-target for simple columns). The spec requires that transitive lineage through CTEs is always materialized.

---

# 5. Output Schema

The final lineage graph is a JSON object containing `nodes`, `edges`, and `metadata`.

```json
{
  "nodes": [],
  "edges": [],
  "metadata": {
    "source_sql_hash": "...",
    "generated_at": "..."
  }
}
```

### Node Schema
```json
{
  "id": "string",
  "type": "string",
  "name": "string",
  "properties": {}
}
```

### Edge Schema
```json
{
  "source": "node_id",
  "target": "node_id",
  "edge_type": "string",
  "properties": {}
}
```

`edge_type` must be one of the defined types: `VALUE_FLOW`, `ROW_FLOW_IN`, `ROW_FLOW_OUT`, `FILTER_CONDITION`, `JOIN_KEY`, `JOIN_FILTER`, `GROUPING_KEY`, `PARTITION_KEY`, `ORDERING_KEY`, `UNION_MEMBER`, `INSERTS_INTO`, `CTE_PRODUCES`.

---

# 6. LLM Extraction Model – Intermediate Representation (IR)

To prevent hallucinations, an LLM (or any parser) outputs an Intermediate Representation that fully describes the SQL semantics. A deterministic **Graph Builder** then converts the IR into the final lineage graph.

## 6.1 Query IR Structure

```json
{
  "ctes": [
    {
      "name": "cte_name",
      "columns": ["col1", "col2"],
      "query": <ir_subquery>
    }
  ],
  "tables": [
    {
      "alias": "o",
      "name": "orders",
      "columns": ["id", "amount", "customer_id", "status"]
    }
  ],
  "joins": [
    {
      "type": "INNER",
      "left": { "table": "o", "column": "customer_id" },
      "right": { "table": "c", "column": "id" },
      "conditions": [
        { "type": "equi", "left": {...}, "right": {...} },
        { "type": "comparison", "op": ">", "left": {...}, "right": {...} }
      ]
    }
  ],
  "filters": [
    {
      "clause": "WHERE",
      "expression": <ir_expression>
    }
  ],
  "group_by": [
    { "expression": <ir_expression> }
  ],
  "having": [
    { "expression": <ir_expression> }
  ],
  "aggregates": [
    {
      "function": "SUM",
      "arguments": [ <ir_expression> ],
      "output_alias": "total_amount"
    }
  ],
  "windows": [
    {
      "function": "ROW_NUMBER",
      "output_alias": "rn",
      "partition_by": [ <ir_expression> ],
      "order_by": [ { "expression": <ir_expression>, "direction": "ASC" } ]
    }
  ],
  "select_list": [
    {
      "alias": "col_alias",
      "expression": <ir_expression>
    }
  ],
  "union_operations": [
    {
      "operation": "UNION ALL",
      "branches": [ <ir_query>, <ir_query>, ... ]
    }
  ],
  "target": {
    "table": "target_table",
    "columns": ["col1", "col2"],
    "mapping": [ ... ] // optional, usually derived from select list
  }
}
```

## 6.2 Expression IR

Expressions are represented as a tree of operations:

```json
{
  "type": "column_ref",
  "table": "o",
  "column": "amount"
}
```

```json
{
  "type": "function",
  "name": "ABS",
  "arguments": [ <ir_expression> ]
}
```

```json
{
  "type": "binary_op",
  "op": "+",
  "left": <ir_expression>,
  "right": <ir_expression>
}
```

```json
{
  "type": "constant",
  "value": "agr_qlty_cat_type_id",
  "data_type": "string"
}
```

```json
{
  "type": "case",
  "when_clauses": [ { "condition": <ir_expression>, "result": <ir_expression> } ],
  "else_result": <ir_expression>
}
```

## 6.3 Subquery IR

Any subquery (e.g., CTE body, branch of a union) is itself a Query IR object, enabling recursive construction.

---

# 7. Graph Construction Rules (from IR)

The Graph Builder reads the IR and creates nodes/edges deterministically. Here are the key rules:

### 7.1 Source Tables
For each table in `tables`, create a Table node and Column nodes for every referenced column.

### 7.2 RowSet Generation
- The **initial RowSet** for a table scan is created from the table alias (e.g., `rs_o`). An `ROW_FLOW_OUT` edge is added from the table to the RowSet (optional; instead, treat the RowSet as directly available). To simplify, we consider the initial RowSet as implicitly existing and omit the table→RowSet edge. The first operator (filter, join) will reference this RowSet via `ROW_FLOW_IN`.
- When multiple tables are joined, the join’s left and right inputs are RowSets.

### 7.3 Filter
- Create a Filter node.
- Add `ROW_FLOW_IN` from the input RowSet to the Filter.
- Add `ROW_FLOW_OUT` from the Filter to a new output RowSet (e.g., `rs_filtered`).
- For each column reference in the filter expression, add `FILTER_CONDITION` edges.

### 7.4 Join
- Create a Join node.
- Add `ROW_FLOW_IN` from the left RowSet and right RowSet to the Join (two edges).
- Add `ROW_FLOW_OUT` from the Join to a new output RowSet.
- For equi conditions, add `JOIN_KEY` edges from the participating columns.
- For non-equi conditions, add `JOIN_FILTER` edges from the expression columns.

### 7.5 Aggregation
- Create an Aggregate node for each aggregate function.
- The input RowSet is connected via `ROW_FLOW_IN`.
- The output RowSet via `ROW_FLOW_OUT`.
- For each group-by column, add a `GROUPING_KEY` edge from the column to the Aggregate node.
- For each aggregate argument expression, decompose into transformation nodes and link them via `VALUE_FLOW` to the Aggregate node. Then the Aggregate node’s output is connected via `VALUE_FLOW` to the target column (or select list alias).

### 7.6 Window Function
- Create a Window node.
- `ROW_FLOW_IN` / `ROW_FLOW_OUT` similar to aggregation.
- `PARTITION_KEY` from partition columns.
- `ORDERING_KEY` from ordering columns.
- If the window function uses an argument (e.g., `SUM(amount)`), link that expression via `VALUE_FLOW` to the Window node. The Window node then connects via `VALUE_FLOW` to the output column.

### 7.7 SELECT List
- For each expression in the select list, decompose into transformation nodes.
- Link the final transformation node to the output column (Target Column or CTE Column) via `VALUE_FLOW`.
- If the expression is a simple column reference, link the source column directly.

### 7.8 Union
- For each branch, the branch query is processed independently, producing a final RowSet that feeds the union.
- Create a Union node.
- Add `UNION_MEMBER` edges from each branch RowSet to the Union node.
- The Union node produces an output RowSet via `ROW_FLOW_OUT`.

### 7.9 CTE
- Process the CTE body as a full query. The select list outputs become CTE Column nodes.
- The CTE body’s final RowSet is linked to the CTE node via `CTE_PRODUCES` (or we simply use the CTE node as the RowSet). We model the CTE as a container; the CTE’s output RowSet is directly attached to the CTE node.
- Any query that references the CTE uses that RowSet as an input.

### 7.10 Transitive Lineage
After constructing the graph, the builder adds direct `VALUE_FLOW` edges from ultimate source columns to final targets if they only pass through intermediate nodes that are not transformations (e.g., through a CTE column). The rule: if a CTE column derives its value from a source column with no intermediate transformation (i.e., just a column reference), then the final target column also gets a `VALUE_FLOW` edge directly from that same source column, alongside the path through the CTE. Both edges co-exist.

### 7.11 Insert Target
- Target Table and Target Columns are created.
- For each insert column, link the output expression’s final node to the Target Column via `VALUE_FLOW`.
- Add `INSERTS_INTO` from each Target Column to the Target Table.

---

# 8. Examples

## 8.1 Simple Aggregate

```sql
SELECT customer_id, SUM(amount) AS total
FROM orders
WHERE status = 'ACTIVE'
GROUP BY customer_id
```

**Graph nodes (simplified list)**:
- `orders` (table)
- `orders.customer_id`, `orders.amount`, `orders.status` (columns)
- `rs_orders` (rowset for table scan)
- `filter` (WHERE)
- `rs_filtered` (after WHERE)
- `agg_sum` (aggregate node, function SUM)
- `rs_aggregated` (after aggregation)
- `tgt_total` (target column)

**Edges**:
- `ROW_FLOW_IN`: rs_orders → filter
- `ROW_FLOW_OUT`: filter → rs_filtered
- `FILTER_CONDITION`: orders.status → filter
- `ROW_FLOW_IN`: rs_filtered → agg_sum
- `ROW_FLOW_OUT`: agg_sum → rs_aggregated
- `GROUPING_KEY`: orders.customer_id → agg_sum
- `VALUE_FLOW`: orders.amount → agg_sum   (value input)
- `VALUE_FLOW`: agg_sum → tgt_total        (aggregate result)

Note: The group-by column `customer_id` is not linked via VALUE_FLOW to the output; the output column directly inherits value from the column reference via a simple VALUE_FLOW from `orders.customer_id` (because the select list contains the raw column, not from the aggregate). That flow goes: `orders.customer_id` → (through the aggregation’s output rowset implicitly) but since `GROUP BY` guarantees the value is the same, the graph can show a direct VALUE_FLOW edge from `orders.customer_id` to the target column.

## 8.2 Window Function

```sql
SELECT
  order_id,
  ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) AS rn
FROM orders
```

**Key edges**:
- Window node `win_rn` (function ROW_NUMBER)
- `PARTITION_KEY`: orders.customer_id → win_rn
- `ORDERING_KEY`: orders.order_date → win_rn
- `ROW_FLOW_IN`: rs_orders → win_rn
- `ROW_FLOW_OUT`: win_rn → rs_with_rn
- `VALUE_FLOW`: win_rn → target column `rn`

## 8.3 CTE with Transitive Lineage

```sql
WITH cte AS (
  SELECT amount FROM orders
)
SELECT amount FROM cte
```

**Graph**:
- CTE node `cte`
- CTE column `cte.amount`
- `CTE_PRODUCES`: cte → rs_cte (the rowset)
- Inside CTE: direct VALUE_FLOW from `orders.amount` to `cte.amount`.
- Outer query: select list maps `cte.amount` to target column `out_amount`. A direct VALUE_FLOW `cte.amount → out_amount` is added.
- Transitive lineage: graph builder also adds a direct VALUE_FLOW `orders.amount → out_amount`. This edge is explicitly materialized.

---

# 9. Row-Generating Transformations

For functions like `UNNEST`, `GENERATE_SERIES`, or lateral joins that multiply rows:
- Create a **RowGenerator** node.
- Its output RowSet is connected via `ROW_FLOW_OUT`.
- Input arguments (e.g., the array column for UNNEST) are linked via `VALUE_FLOW` from the source column to the RowGenerator node.
- The generated rows flow into subsequent operators via `ROW_FLOW_IN` from the generator’s output RowSet.

---

# 10. Version History

| Version | Date       | Changes                                                       |
|---------|------------|---------------------------------------------------------------|
| 2.0     | 2025-xx-xx | Original draft with transformation graph concept.             |
| 2.1     | 2026-06-13 | Added RowSet nodes, pipeline edges, detailed IR, row generator, window output edges, transitive lineage rules, union_type property, constant node usage clarification. |

---

**End of Specification**