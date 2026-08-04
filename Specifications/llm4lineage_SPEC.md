# llm4lineage — Spec-Driven Implementation Guide

**Version:** 1.0  
**Repo:** [https://github.com/Xpehutta/llm4lineage](https://github.com/Xpehutta/llm4lineage)  
**Audience:** AI coding agent (Cursor IDE) — spec-driven implementation  
**Date:** 2026-08-04

---

## How to Use This Spec (для агента)

1. Работай **по фазам, строго по порядку** (0 → 6).
2. В каждой фазе выполни все задачи из раздела **Tasks**, затем прогони **Acceptance Criteria**.
3. Не переходи к следующей фазе, пока все критерии текущей не выполнены.
4. После каждой фазы запускай: `pytest tests/ -q` и `ruff check .`
5. Если критерий нельзя проверить автоматически — добавь тест, который его проверяет.
6. Не меняй публичные контракты из раздела **Data Contracts** без явного указания.
7. Код-стайл: Python 3.9+, типизация, Pydantic v2, sqlglot — единственный SQL-парсер.

---



## Context

`llm4lineage` — LLM-assisted toolkit для извлечения data lineage из SQL (GreenPlum/банковское DWH). Четыре трека:

- SQL pipeline (sqlglot → AST JSON → column lineage → LLM analysis)
- Table-level lineage (target/sources)
- Column-level graph (SQL2Graph: DERIVED_FROM, FILTERED_BY, USES_COLUMN, JOINS_ON, GROUPED_BY)
- SQL logical chunk parser (CTE/UNION/INSERT декомпозиция)

Текущие ограничения, которые закрывает этот спек:

1. Дефолтный диалект `spark`, а реальные данные — GreenPlum (`::text`, `NULL::uuid`, UNION ALL) → неточный парсинг.
2. Отсутствует schema registry → `SELECT *` и неоднозначные колонки не резолвятся.
3. UNION/aggregate/window не представлены как ноды графа (только флаги).
4. Нет экспорта в открытый стандарт (OpenLineage) → нет интеграций.
5. Нет confidence/provenance на LLM-рёбрах.
6. Нет impact/downstream API («что сломается»).
7. Нет golden-набора и CI → регрессии точности не видны.
8. Тяжёлый core-инсталл (jupyter/streamlit/langgraph в обязательных зависимостях).

---



## Industry Best Practices (что подтверждают лидеры индустрии)

Исследование индустрии (август 2026):


| Источник                                      | Ключевые практики                                                                                                                                                                                                                                                                                                                                                                                                                       |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DataHub SQL Parser** (на sqlglot)           | Декларирует **97–99% точность** column-level lineage; `confidence_score` (0–1) в результате парсинга; `SELECT `* **расширяется только при наличии схем**; явный список не поддерживаемого: UDF (lineage на входные колонки UDF), табличные UDF, `json_extract`, `UNNEST` (best-effort), structs (best-effort), multi-statement SQL; `SqlParsingAggregator` резолвит lineage **через временные таблицы и переименования/подмены таблиц** |
| **SQLMesh** (column-level lineage на sqlglot) | «See impact of changes **before** you run them in your warehouse with column-level lineage» — impact-анализ до выполнения; lineage как часть plan/apply цикла                                                                                                                                                                                                                                                                           |
| **OpenLineage**                               | Разделение **design-time (static)** и **runtime** lineage; модель Dataset + Job; интеграция с каталогами (DataHub/Marquez) через стандартные события                                                                                                                                                                                                                                                                                    |
| **SQLLineage**                                | Объединение lineage **нескольких SQL-операторов** в один граф с выявлением промежуточных таблиц; парсер-агностичность (pluggable parsers)                                                                                                                                                                                                                                                                                               |
| **dbt**                                       | Lineage как DAG от явных `ref()`/`source()`; impact-анализ и тесты на каждом изменении                                                                                                                                                                                                                                                                                                                                                  |


**Выводы, которые закладываем в спек:**

1. **Схема — главный фактор точности.** И DataHub, и SQLMesh упираются в schema registry для `SELECT `* и колоночного резолва. → Phase 1 критична.
2. **Confidence — индустриальный стандарт.** DataHub отдаёт `confidence_score` для каждого результата парсинга. → Phase 3 (не только для LLM-рёбер, но и для детерминированных — как сигнал «есть сомнения в резолве»).
3. **Явно документировать ограничения.** DataHub публикует список «supported/not supported» — потребители знают границы. → README-раздел и `metadata.limitations` в контракте C3.
4. **Кросс-стейтмент резолв (temp tables, rename/swap)** — ключевая фича продакшен-парсеров. → агрегатор в Phase 3/5.
5. **Impact-анализ до прогона** — паттерн SQLMesh. → Phase 5 подтверждён индустрией.
6. **Бенчмарк точности** — DataHub заявляет 97–99%: для llm4lineage цель golden-тестов ≥ 0.9 edge-F1 на детерминированной части (Phase 3.6).

---



## Academic Research (arXiv, август 2026)

Поиск по arXiv API (11 запросов, ~45 записей; полный перечень — см. `arxiv_llm4lineage_research.md`). Ключевые работы, влияющие на спек:


| Работа                                                                                | arXiv      | Релевантность → фаза                                                                                                                     |
| ------------------------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **LINEAGEX: A Column Lineage Extraction System for SQL** (2025)                       | 2505.23133 | Прямой аналог SQL2Graph; детерминированный колоночный lineage для корпоративных хранилищ → бенчмарк-ориентир и источник идей для Phase 2 |
| **Dialect-Agnostic SQL Parsing via LLM-Based Segmentation** (2026)                    | 2603.16155 | LLM-сегментация для диалектов, не поддерживаемых детерминированным парсером → обоснование LLM-fallback в Phase 1                         |
| **DBAutoDoc: Undocumented Database Schemas via LLM** (2026)                           | 2603.23050 | Автовосстановление схем (FK, криптонимы колонок) → усиление SchemaRegistry в Phase 1                                                     |
| **Calibrating LLMs for Text-to-SQL by Sub-clause Frequencies** (2025)                 | 2505.23804 | Калибровка уверенности LLM → методы confidence в Phase 3                                                                                 |
| **Error Detection for Text-to-SQL Semantic Parsing** (2023)                           | 2305.13683 | Over-confidence и обнаружение ошибок → обоснование confidence-разметки в Phase 3                                                         |
| **ErrorLLM: Modeling SQL Errors for Text-to-SQL Refinement** (2026)                   | 2603.03742 | Классификация SQL-ошибок → улучшение refine/verify цикла (SQLRefiner)                                                                    |
| **SQL Equivalence Checking with LLMs** (2024)                                         | 2412.05561 | Проверка эквивалентности SQL → усиление SQL2GraphValidator                                                                               |
| **DW-Bench: LLM Reasoning on DWH Graph Topology** (2026)                              | 2604.18964 | Бенчмарк LLM на графах хранилища (FK + lineage) → способ оценки графов в Phase 6                                                         |
| **You Say 'What', I Hear 'Where' and 'Why': Fine-Grained Provenance from SQL** (2018) | 1805.11517 | Теоретическая база тонкозернистого lineage («где»/«почему») → Phase 5 impact                                                             |
| **SQLong: NL2SQL for Longer Contexts** (2025)                                         | 2502.16747 | Работа с большими схемами → управление размером контекста (пачки DDL)                                                                    |


**Выводы по академической литературе:**

1. Готовых **LLM-only** систем извлечения lineage из SQL в литературе нет — ниша свободна, llm4lineage оригинален. Ближайшие: LINEAGEX (детерминированный) и DW-Bench (оценка).
2. Направление «LLM как fallback для диалектов/неоднозначностей поверх детерминированного парсера» — подтверждено (Dialect-Agnostic, 2603.16155).
3. Confidence/калибровка — активная исследовательская тема; методы применимы к Phase 3.

---



## Data Contracts (не менять без явного указания)



### C1. Table-level lineage

```json
{
  "type": "object",
  "required": ["target", "sources"],
  "properties": {
    "target": { "type": "string" },
    "sources": { "type": "array", "items": { "type": "string" } }
  }
}
```



### C2. SQL2Graph extraction (simplified)

```json
{
  "type": "object",
  "required": ["ctes", "output_columns", "filters", "joins", "group_by_columns"],
  "properties": {
    "ctes": { "type": "array" },
    "output_columns": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["alias", "expression", "dependencies", "aggregate", "window_function"],
        "properties": {
          "alias": { "type": "string" },
          "expression": { "type": "string" },
          "dependencies": { "type": "array" },
          "aggregate": { "type": "boolean" },
          "window_function": { "type": "boolean" }
        }
      }
    },
    "filters": { "type": "array" },
    "joins": { "type": "array" },
    "group_by_columns": { "type": "array" }
  }
}
```



### C3. Graph JSON (node-link)

```json
{
  "nodes": [
    { "id": "output.total", "node_type": "output_column" },
    { "id": "orders.amount", "node_type": "source_column" }
  ],
  "links": [
    { "source": "orders.amount", "target": "output.total", "edge_type": "DERIVED_FROM" }
  ],
  "metadata": {
    "source_sql_hash": "...",
    "generated_at": "...",
    "spec_version": "2.1",
    "implementation_profile": "column_level_v2",
    "limitations": ["udf_inputs_only", "unnest_best_effort"]
  }
}
```

**Новое (добавляется в этой серии фаз):** каждое ребро дополняется полями
`confidence: float` (0–1) и `provenance: "deterministic" | "llm" | "llm_verified"`.
Поле `metadata.spec_version` не меняется, `implementation_profile` → `column_level_v2`.

### C4. OpenLineage export (добавляется в Phase 4)

Дизайн-тайм событие (static lineage) на каждый SQL-скрипт:

```json
{
  "eventType": "START",
  "eventTime": "ISO-8601",
  "job": { "namespace": "llm4lineage", "name": "<sql_hash>" },
  "inputs": [
    { "namespace": "greenplum", "name": "schema.table",
      "facets": { "columnLineage": { "fields": { "col": { "inputFields": [ { "namespace": "greenplum", "name": "schema.src_table", "field": "src_col" } ] } } } } }
  ],
  "outputs": [
    { "namespace": "greenplum", "name": "schema.target_table" }
  ]
}
```

---



## Phase 0 — Foundations (фундамент)

**Goal:** стабильная база: пины зависимостей, CI, лёгкий инсталл.

### Tasks

- [ ] **T0.1** В `pyproject.toml`: зафиксировать `sqlglot>=26.26.0,<27` (мажорный пин). Проверить, что `uv.lock` обновлён (`uv lock`).
- [ ] **T0.2** Разделить зависимости на optional extras:
  - core: pydantic, pydantic-settings, sqlglot, networkx, python-dotenv, tenacity, json5, dataclasses-json, typing-extensions
  - `[web]`: streamlit, plotly, graphviz, matplotlib, pyperclip, watchdog
  - `[notebooks]`: jupyter, jupyterlab, notebook, ipykernel, ipywidgets, anywidget, pandas, numpy
  - `[llm]`: langchain, langchain-core, langchain-community, langchain-huggingface, langchain-gigachat, gigachat, langgraph, huggingface-hub, tqdm
  - `[dev]`: pytest, ruff
  Обязательные зависимости — только core.
- [ ] **T0.3** Добавить `.github/workflows/ci.yml`: `pytest tests/ -q`, `ruff check .`, Python 3.9–3.12 matrix.
- [ ] **T0.4** Добавить `ruff` конфиг (line-length 100, target-version py39) в `pyproject.toml`.



### Acceptance Criteria

- [ ] `pip install -e ".[core]"` работает без jupyter/streamlit.
- [ ] CI зелёный на свежем checkout.
- [ ] `pytest tests/ -q` — все существующие тесты проходят без изменений.

---



## Phase 1 — Dialect + Schema Registry

**Goal:** точный парсинг GreenPlum-кода и резолв колонок через схему.

### Tasks

- [ ] **T1.1** `Classes/Config`: поле `default_dialect: str = "postgres"` (переопределяется per-call). Обновить все вызовы `SQLParser`/`SQL2GraphParser`/`SQLLogicalChunkParser` — дефолт берётся из Config, не хардкодится `"spark"`. Добавить `"teradata"` как поддерживаемое значение для легаси-анализа.
- [ ] **T1.2** Новый модуль `Classes/schema_registry.py`:
  - `DDLParser` — парсит `CREATE TABLE`/`CREATE VIEW` через sqlglot → `{schema: {table: {col: type}}}`
  - `SchemaRegistry` — загрузка из текста DDL, из CSV (имя таблицы, колонки), merge, `to_sqlglot_schema() -> sqlglot.schema.Schema`
- [ ] **T1.3** В `SQL2GraphParser.simplify()`: перед извлечением прогонять `sqlglot.optimizer.qualify.qualify_columns(ast, schema=registry.to_sqlglot_schema())` — резолв `SELECT `*, алиасов, неоднозначных колонок. Флаг `use_schema: bool = True`; при отсутствии схемы — graceful degradation (текущее поведение).
- [ ] **T1.4** `ViewExpander`: разворачивание обращений к views в базовые таблицы (подстановка тела view из registry) — до qualify. Модуль `Classes/view_expander.py`.
- [ ] **T1.5** Обновить `Web/app.py`: выбор диалекта — `["postgres", "spark", "teradata", "hive"]`, upload DDL → SchemaRegistry.
- [ ] **T1.6** LLM-fallback для диалектов (arXiv 2603.16155 «Dialect-Agnostic SQL Parsing via LLM-Based Segmentation»): если sqlglot не смог распарсить statement (parse error / unknown syntax), передать SQL в `SQL2GraphLLMExtractor` с пометкой `parse_fallback=true` — LLM сегментирует и извлекает lineage; результат помечается `provenance="llm"`, `confidence<1.0`.
- [ ] **T1.7** Автовосстановление схемы (arXiv 2603.23050 «DBAutoDoc»): если для таблицы нет DDL — опциональный LLM-проход по `SELECT`/INSERT-выражениям для восстановления набора колонок (криптонимы, отсутствующие PK/FK) с последующим мержем в SchemaRegistry.
- [ ] **T1.8** Управление размером контекста (arXiv 2502.16747 «SQLong»): при больших пачках DDL/длинных SQL — чанкинг схемы и инкрементальный резолв; тест на `data/DDLs.txt` (полный файл).



### Acceptance Criteria

- [ ] На `data/DDLs.txt` + `data/SQL.txt`: все `schema.table` в sources/target резолвятся полностью квалифицированными (без алиасов).
- [ ] Запрос с `SELECT *` из таблицы со схемой даёт колоночные зависимости (не пустой lineage).
- [ ] Тест: `tests/test_schema_registry.py` — парсинг DDL, merge, qualify-резолв, view expansion.
- [ ] Без схемы — поведение идентично текущему (регрессий нет).

---



## Phase 2 — SQL2Graph: UNION / aggregate / window ноды

**Goal:** граф отражает реальную структуру GreenPlum-запросов (UNION ALL, CASE, агрегаты, окна).

### Tasks

- [ ] **T2.1** В `SQL2GraphBuilder` добавить типы нод:
  - `union` (с атрибутом `union_type: ALL | DISTINCT`)
  - `aggregate` (имя функции: SUM/COUNT/AVG/…)
  - `window` (partition_by, order_by)
  - `transformation` (CASE, CAST, COALESCE, арифметика; поле `function`, `expression_text`)
  - `rowset` (материализованный для каждого CTE)
- [ ] **T2.2** Новые типы рёбер: `ROW_FLOW_IN`, `ROW_FLOW_OUT`, `VALUE_FLOW` (трансформации), `AGGREGATES_ON`, `WINDOW_OVER`. Сохранить обратную совместимость существующих 5 рёбер.
- [ ] **T2.3** В `SQL2GraphParser.simplify()`: детект UNION-веток (уже есть в chunk parser — переиспользовать), агрегатов (`_looks_aggregate` → нода), оконных (`_looks_window` → нода), выражений → цепочка transformation-нод.
- [ ] **T2.4** CTE → `rowset`-ноды + `ROW_FLOW` рёбра через существующий `link_cte_aliases()`.
- [ ] **T2.5** Обновить `SQL2GraphValidator`: валидация новых типов нод/рёбер (allowed types), edge-F1 расширен на новые рёбра.
- [ ] **T2.6** Обновить документацию: `Specifications/SQL2Graph_spec.md` — раздел Implementation Status (отметить выполненное).
- [ ] **T2.7** LLM-проверка эквивалентности (arXiv 2412.05561 «SQL Equivalence Checking with LLMs»): в `SQL2GraphValidator` — опциональный шаг «эквивалентен ли трансформированный SQL исходному» (после qualify/view-expansion), как защита от потери семантики при нормализации.



### Acceptance Criteria

- [ ] На реальном запросе с 2+ UNION ALL ветками граф содержит ноды `union` и корректные `ROW_FLOW` рёбра.
- [ ] `SELECT SUM(x) ... GROUP BY y` даёт: `aggregate`-ноду, `AGGREGATES_ON x`, `GROUPED_BY y`.
- [ ] `ROW_NUMBER() OVER (PARTITION BY a ORDER BY b)` даёт `window`-ноду с partition/order атрибутами.
- [ ] JSON-экспорт графа валиден по C3 (новые node_type/edge_type добавлены в схему).
- [ ] Все старые тесты зелёные (обратная совместимость).

---



## Phase 3 — Confidence + Provenance + Golden Set

**Goal:** измеримая точность и доверие к LLM-рёбрам.

### Tasks

- [ ] **T3.1** Pydantic-модели: `EdgeConfidence` — поля `confidence: float` (0–1), `provenance: Literal["deterministic","llm","llm_verified"]`. Встроить в `SQL2GraphBuilder.build()`: deterministic-рёбра → `confidence=1.0, provenance="deterministic"`; LLM-рёбра из `SQL2GraphLLMExtractor` → `confidence=<из модели или 0.8 по умолчанию>, provenance="llm"`; после `verify()` → `provenance="llm_verified"`.
- [ ] **T3.1a** Confidence не только для LLM-рёбер (паттерн DataHub `confidence_score`): детерминированный резолв, где парсер «сомневается» (нерезолвленная колонка, best-effort UNNEST/struct, UDF-входы), помечать `confidence < 1.0` + код причины в `metadata.limitations`.
- [ ] **T3.2** Структурированный вывод LLM: в `SQL2GraphLLMExtractor.extract()` максимально использовать `with_structured_output` (GigaChat/OpenAI); для HF-local — строгий JSON-парсер с повторной попыткой. Добавить запрос confidence в промпт.
- [ ] **T3.3** Персистентный кэш LLM: `Classes/llm_cache.py` — SQLite/JSON на диске, ключ = хэш(SQL + prompt_version + model). Использовать в `SQL2GraphLLMExtractor` и `SQLRefiner` (заменить/обернуть `_generate_cache_key`).
- [ ] **T3.3a** `SqlStatementAggregator` (паттерн DataHub `SqlParsingAggregator`): при пакетной обработке пачки INSERT-скриптов резолвить lineage **через временные таблицы и переименования/подмены таблиц** (target одного оператора = source другого; alias-цепочки). Выход — объединённый граф + карта «логическая таблица → физическая».
- [ ] **T3.4** Golden set: `tests/golden/` — на основе `data/DDLs_10` зафиксировать ожидаемый граф (node-link JSON) как golden files. Скрипт `tests/golden/update_golden.py` для перегенерации (явный флаг).
- [ ] **T3.5** Edge-level F1: в `validation_classes.py` добавить `calculate_edge_f1(expected_graph, actual_graph)` (по (source,target,edge_type) тройкам). Прогнать на golden set — зафиксировать baseline в CI-отчёт.
- [ ] **T3.6** CI: job «golden» — `pytest tests/golden/` и публикация F1-метрик в summary. Целевой порог: **edge-F1 ≥ 0.9** на детерминированной части (ориентир DataHub — 97–99% точность).
- [ ] **T3.7** Калибровка уверенности (arXiv 2505.23804 «Calibrating LLMs for Text-to-SQL» + 2305.13683 «Error Detection for Text-to-SQL»): если LLM возвращает confidence, сопоставлять его с фактической точностью на golden set (reliability diagram); при систематическом over-confidence — корректировать пороги/промпт.



### Acceptance Criteria

- [ ] Каждое ребро в JSON-графе содержит `confidence` и `provenance`.
- [ ] Golden-тесты детерминированы: два прогона без изменений кода дают идентичный граф (для mock-провайдера).
- [ ] Edge-F1 на golden set ≥ 0.9 для deterministic-части (без LLM).
- [ ] Кэш: повторный прогон с теми же SQL не вызывает LLM-вызовы (mock-счётчик).

---



## Phase 4 — OpenLineage Exporter

**Goal:** интеграция с экосистемой (DataHub/Marquez) через открытый стандарт.

### Tasks

- [ ] **T4.1** Новый модуль `Classes/openlineage_exporter.py`:
  - `to_openlineage_run_event(graph, sql, sql_hash) -> dict` — по C4
  - `to_openlineage_job_event(graph, sql, sql_hash) -> dict` (design-time static lineage)
  - field-level lineage: выходные колонки → `inputFields` из DERIVED_FROM-рёбер
- [ ] **T4.2** Маппинг node_type/edge_type → OpenLineage-концепции (Dataset/Field/Job). RowSet/transformation — в `custom_facets` (не теряем информацию).
- [ ] **T4.3** CLI: `python -m Classes.openlineage_exporter --sql file.sql --format run|job [--emit http://marquez:5000]` (emit — опционально, только если передан URL).
- [ ] **T4.4** Тесты: `tests/test_openlineage_exporter.py` — валидность JSON по C4, корректность field-level маппинга.



### Acceptance Criteria

- [ ] Экспорт в формате run-event и job-event валиден (JSON schema из C4).
- [ ] Field-level lineage: для каждой output-колонки есть inputFields из источников.
- [ ] `--emit` без URL не отправляет ничего в сеть (безопасность по умолчанию).
- [ ] Тесты зелёные.

---



## Phase 5 — Impact / Downstream API

**Goal:** ответ на «что сломается, если изменю колонку/таблицу» — ключевой кейс миграций.

### Tasks

- [ ] **T5.1** В `Classes/graph_drawer.py` (или новый `Classes/impact_analyzer.py`): `analyze_impact(graph, target_node) -> {"upstream": [...], "downstream": [...]}` — полный транзитивный обход в обе стороны с типами рёбер.
- [ ] **T5.2** Table-level impact: из колоночного графа агрегировать до таблиц («какие витрины/отчёты зависят от таблицы X»).
- [ ] **T5.3** CLI: `python -m Classes.impact --sql file.sql --target schema.table --direction down|up|both`.
- [ ] **T5.4** Streamlit: вкладка «Impact» — выбор узла, визуализация затронутых путей (есть `_get_upstream_nodes` — расширить до downstream).
- [ ] **T5.5** Использовать `SqlStatementAggregator` (T3.3a) для impact-анализа по пачке скриптов: «какие витрины/отчёты затронет изменение таблицы X во всём наборе SQL» — паттерн SQLMesh «impact before run».
- [ ] **T5.6** Тонкозернистый impact «где/почему» (arXiv 1805.11517 «Fine-Grained Provenance from SQL»): для колонки возвращать не только узлы, но и **причину** попадания данных в результат (фильтр/джойн/агрегация) — расширение `analyze_impact` полем `reason` по типам рёбер (FILTERED_BY/JOINS_ON/AGGREGATES_ON).



### Acceptance Criteria

- [ ] Для графа из `data/sql2graph_result.json`: impact-запрос по колонке возвращает все транзитивные downstream-узлы.
- [ ] Тест: `tests/test_impact_analyzer.py` — chain из 3+ узлов, проверка транзитивности.
- [ ] CLI работает без Streamlit.

---



## Phase 6 — MERGE/UPDATE + Полировка

**Goal:** покрытие реальных DWH-паттернов и чистота кода.

### Tasks

- [ ] **T6.1** MERGE: детерминированный target/sources для `MERGE INTO ... USING ... ON ...` (минимум table-level; column-level — если sqlglot даёт AST). UPDATE: target + sources из FROM.
- [ ] **T6.2** `data/SQL_valid.txt` и `data/DDLs_10.txt` — добавить в golden set (расширить покрытие).
- [ ] **T6.3** README: раздел «Output contracts» (C1–C4 с примерами) + «CI & Evaluation» + «Roadmap» (ссылка на этот спек).
- [ ] **T6.3a** README: раздел **«Supported / Not supported»** (по образцу DataHub SQL Parser) — перечислить покрываемые конструкции (SELECT/INSERT/CTAS/CTE/UNION ALL/`SELECT `* со схемой) и явные ограничения (UDF — lineage на входные колонки, табличные UDF, `json_extract`, `UNNEST` best-effort, structs best-effort, multi-statement SQL).
- [ ] **T6.4** `ruff check .` без warning'ов, docstring'и на публичных классах.
- [ ] **T6.5** Бенчмарк-сравнение с LINEAGEX (arXiv 2505.23133): на golden set прогнать детерминированную часть llm4lineage и сопоставить покрытие/точность с опубликованными результатами LINEAGEX (метрики и конструктивное покрытие) — зафиксировать сравнение в README/CI-отчёт.
- [ ] **T6.6** Оценка графов LLM (arXiv 2604.18964 «DW-Bench»): использовать методологию DW-Bench (FK + lineage рёбра, топологический рейзинг) для валидации, что LLM-шаги (verify/enhance) не ломают граф; опционально — мини-бенчмарк на их данных.



### Acceptance Criteria

- [ ] MERGE/UPDATE скрипты дают корректный target/sources.
- [ ] Golden set покрывает ≥ 3 реальных скрипта из data/.
- [ ] README обновлён, ruff чистый.

---



## Definition of Done (для всего спека)

- [ ] Все фазы 0–6 выполнены, все Acceptance Criteria закрыты.
- [ ] `pytest tests/ -q` — зелёный (включая golden).
- [ ] `ruff check .` — без ошибок.
- [ ] CI на GitHub Actions проходит.
- [ ] Публичные контракты C1–C4 валидны; примеры в README актуальны.
- [ ] Реальная проверка: прогнать `data/SQL.txt` через полный пайплайн → JSON-граф + OpenLineage-экспорт + impact-отчёт, приложить пример вывода в PR.

---



## Sources (best practice research, август 2026)

- **DataHub SQL Parser** (на sqlglot, 97–99% точность, confidence_score, SqlParsingAggregator): [https://github.com/datahub-project/datahub/blob/master/docs/lineage/sql_parsing.md](https://github.com/datahub-project/datahub/blob/master/docs/lineage/sql_parsing.md)
- **DataHub Lineage guide** (column vs table lineage, impact analysis): [https://datahubproject.io/docs/features/feature-guides/lineage/](https://datahubproject.io/docs/features/feature-guides/lineage/)
- **SQLMesh** (column-level lineage на sqlglot, impact-before-run): [https://sqlmesh.readthedocs.io/](https://sqlmesh.readthedocs.io/) + [https://github.com/TobikoData/sqlmesh](https://github.com/TobikoData/sqlmesh)
- **OpenLineage** (design-time vs runtime, Dataset+Job object model): [https://openlineage.io/docs/spec/object-model](https://openlineage.io/docs/spec/object-model) + [https://github.com/OpenLineage/OpenLineage](https://github.com/OpenLineage/OpenLineage)
- **SQLLineage** (много-стейтмент агрегация, pluggable parsers): [https://github.com/reata/sqllineage](https://github.com/reata/sqllineage)
- **sqlglot** (парсер/транспилятор, 30+ диалектов, optimizer): [https://github.com/tobymao/sqlglot](https://github.com/tobymao/sqlglot) + [https://sqlglot.com/](https://sqlglot.com/)
- **dbt** (DAG-lineage, impact): [https://docs.getdbt.com/](https://docs.getdbt.com/)
- Статья «Как LLM могут помочь определить Data Lineage» (Habr, Сбербанк): [https://habr.com/ru/companies/sberbank/articles/1058618/](https://habr.com/ru/companies/sberbank/articles/1058618/)
- giga4sql (127 скриптов для тестирования): [https://github.com/Xpehutta/giga4sql](https://github.com/Xpehutta/giga4sql)



### Academic (arXiv)

- LINEAGEX: A Column Lineage Extraction System for SQL (2025): [https://arxiv.org/abs/2505.23133](https://arxiv.org/abs/2505.23133)
- Dialect-Agnostic SQL Parsing via LLM-Based Segmentation (2026): [https://arxiv.org/abs/2603.16155](https://arxiv.org/abs/2603.16155)
- DBAutoDoc: Undocumented Database Schemas via LLM (2026): [https://arxiv.org/abs/2603.23050](https://arxiv.org/abs/2603.23050)
- Calibrating LLMs for Text-to-SQL by Sub-clause Frequencies (2025): [https://arxiv.org/abs/2505.23804](https://arxiv.org/abs/2505.23804)
- Error Detection for Text-to-SQL Semantic Parsing (2023): [https://arxiv.org/abs/2305.13683](https://arxiv.org/abs/2305.13683)
- ErrorLLM: Modeling SQL Errors for Text-to-SQL Refinement (2026): [https://arxiv.org/abs/2603.03742](https://arxiv.org/abs/2603.03742)
- SQL Equivalence Checking with LLMs (2024): [https://arxiv.org/abs/2412.05561](https://arxiv.org/abs/2412.05561)
- DW-Bench: LLM Reasoning on DWH Graph Topology (2026): [https://arxiv.org/abs/2604.18964](https://arxiv.org/abs/2604.18964)
- Fine-Grained Provenance from SQL (2018): [https://arxiv.org/abs/1805.11517](https://arxiv.org/abs/1805.11517)
- SQLong: NL2SQL for Longer Contexts (2025): [https://arxiv.org/abs/2502.16747](https://arxiv.org/abs/2502.16747)
- Полный перечень проверенных arXiv-запросов: см. `arxiv_llm4lineage_research.md`

---



## Constraints / Notes

- **Не** добавлять новые обязательные зависимости в core (только optional extras).
- sqlglot — единственный SQL-парсер; sqlparse/sqlfluff не вводить.
- LLM-провайдеры: сохранить абстракцию `LLMFactory` — новый код не должен знать про конкретного провайдера.
- Всё, что агент не может проверить сам (реальная GreenPlum-среда) — помечать `[MANUAL]` в PR-описании.
- При сомнениях между «быстрее» и «по спеку» — выбирать спек.

