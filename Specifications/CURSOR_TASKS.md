# llm4lineage — CURSOR_TASKS.md (план доработки для AI IDE)

**Версия:** 1.0
**Дата:** 2026-08-06
**Репозиторий:** https://github.com/Xpehutta/llm4lineage
**Аудитория:** AI coding agent (Cursor IDE)
**Входные артефакты:** `Specifications/llm4lineage_SPEC.md`, `Specifications/ADDITIONALS.md`, `Specifications/SQL2Graph_spec.md`, этот файл

---

## Как использовать этот план (для агента)

1. Работай **строго по фазам, по порядку** (A → B → C → D → E; F и G — по готовности инфраструктуры).
2. В каждой фазе выполни все **Tasks**, затем прогони **Acceptance Criteria**.
3. Не переходи к следующей фазе, пока все критерии текущей не выполнены.
4. После каждой фазы: `pytest tests/ -q && ruff check . && git commit -m "phase <X>: <summary>"`.
5. Если критерий нельзя проверить автоматически — добавь тест, который его проверяет.
6. **Не ломай публичные контракты** из разделов Data Contracts SPEC-файлов. Обратная совместимость — обязательна (реэкспорт имён).
7. Код-стайл: Python 3.9+, типизация, Pydantic v2, sqlglot — единственный SQL-парсер (кроме нового слоя pg_query для PL/pgSQL).
8. **Принцип честности:** не выдумывай lineage. Неразрешимое помечается `unresolved`/`dynamic`, а не генерируется «на глаз». У отказа должно быть имя.

---

## Контекст

`llm4lineage` — LLM-assisted toolkit для извлечения data lineage из SQL (GreenPlum/банковское DWH, целевой стек: **GreenPlum + PL/pgSQL процедуры**). Четыре трека: SQL pipeline, table-level lineage, column-level graph (SQL2Graph), SQL logical chunk parser.

**Ключевые ограничения, которые закрывает этот план:**
1. Тела PL/pgSQL-функций не парсятся (sqlglot не понимает процедурный синтаксис) — главный разрыв с целевым стеком.
2. Дефолтный диалект `spark` вместо `postgres` (GreenPlum-совместимого).
3. Нет извлечения DDL из каталога GreenPlum (`pg_proc`, `pg_views`, `pg_attribute`).
4. langchain — жёсткая зависимость core-модулей при объявленном optional extra.
5. Разбор LLM-ответов через regex (хрупко, нет confidence).
6. Нет LICENSE, в корне репо — случайные артефакты.
7. Монолиты: `sql2graph_classes.py` (3628 строк), `Web/app.py` (1216 строк).
8. CI без coverage-гейта, mypy и golden-drift проверки.

---

## Фаза A — PL/pgSQL поддержка (блокер целевого сценария)

### A1. ~~Зависимость pg_query~~ — ОТМЕНЕНО (решение 2026-08-07)

Ревизия показала, что исходное задание опиралось на неверные допущения:

1. **`pg_query` — устаревшее имя `pglast`** (тот же автор, последний релиз `0.29`, пакет переименован). Актуальный пакет — `pglast` 8.x.
2. **`pglast`/`pg_query` лицензированы под GPLv3+**, что несовместимо с MIT из задачи C1.
3. **`parse_plpgsql()` не возвращает AST** — только сырые `dict`/`list` без узлов и без границ стейтментов
   (см. документацию pglast и issue lelit/pglast#88). `line_end` пришлось бы выводить вручную.

**Решение:** внешний нативный парсер не подключаем. Сплиттер, описанный в A2 как fallback,
становится **основным путём**: разбор по `;` вне долларовых кавычек + прогон каждого
извлечённого стейтмента через существующий `SQL2GraphParser` (sqlglot).
Лицензия проекта остаётся MIT, нативной сборки нет.

### A2. `Classes/plpgsql_splitter.py` — разбивка тел функций — ✅ ВЫПОЛНЕНО
- **Tasks:**
  - Модуль `Classes/plpgsql_splitter.py` с dataclass:
    ```python
    @dataclass
    class PlpgsqlStmt:
        kind: str            # select | insert | update | delete | execute | perform | create_temp | call
        sql: str
        line_start: int
        line_end: int
        is_dynamic: bool = False
        dynamic_reason: str = ""   # напр. "EXECUTE format(...) с переменными"
    ```
  - Функция `split_function_body(body: str) -> List[PlpgsqlStmt]`:
    - сплиттер по `;` вне долларовых кавычек (`$$...$$`, `$tag$...$tag$`), строковых литералов
      (`'...'` с учётом `''`), идентификаторов в кавычках и комментариев (`--`, `/* */`);
    - корректная обработка вложенных блоков `BEGIN…END`, `IF/ELSIF/ELSE`, `LOOP`, `CASE`;
    - классификация стейтментов по типу; `EXECUTE` со статической строкой → `is_dynamic=False` (парсится), с `format(...)`/переменными → `is_dynamic=True`.
  - Функция `extract_function_def(create_function_sql: str) -> Tuple[str, str]` — вытаскивает имя функции и тело (`AS $$ ... $$`), с учётом `LANGUAGE plpgsql`.
- **Acceptance (тесты `tests/test_plpgsql_splitter.py`):**
  - Функция с `BEGIN...END`, циклом `FOR`, `IF/ELSIF`, `EXECUTE format(...)`, `CREATE TEMP TABLE`, `RETURN QUERY` — все стейтменты извлечены, `kind` корректен, позиции верны.
  - Статический `EXECUTE 'SELECT * FROM t'` → `is_dynamic=False`; `EXECUTE format('...%s...', v)` → `is_dynamic=True` c `dynamic_reason`.

### A3. `Classes/plpgsql_lineage.py` — lineage для функций — ✅ ВЫПОЛНЕНО

> Реализовано сверх задания: колоночный lineage для `UPDATE` (штатный `SQL2GraphParser`
> его не разбирает), нормализация алиасов к физическим таблицам — за счёт неё
> lineage связывается **сквозь** temp-таблицы между стейтментами, и best-effort
> восстановление источников из `EXECUTE format(...)` (рёбра помечены
> `confidence=0.3`, `provenance='unresolved'`).
- **Tasks:**
  - `class PlpgsqlLineageExtractor`:
    - вход: `create_function_sql`, `SchemaRegistry`, `dialect='postgres'`;
    - контекст функции: **temp-таблицы** (`CREATE TEMP TABLE t AS SELECT...` → узел `t` в рамках функции; последующие обращения к `t` связываются с ним), переменные (`v := expr` — локальные источники);
    - каждый извлечённый стейтмент прогоняется через `SQL2GraphParser` (существующий);
    - **консервативное объединение веток** IF/циклов: все ветки попадают в граф;
    - динамический EXECUTE → ребро `transform_type='dynamic'`, `confidence=0.3`, `provenance='unresolved'`, `sql_fragment` = строка EXECUTE;
    - выход — node-link JSON, совместимый с `SQL2GraphPipeline` (поле `pipeline_stage: "plpgsql"` в metadata).
  - Обнаружение рекурсии функций (функция вызывает себя / цикл вызовов) — защита от зацикливания.
- **Acceptance (тесты + golden):**
  - Golden-тест на 3 реальные PL/pgSQL-функции (с temp-таблицей, с IF-ветками, с EXECUTE): граф проходит `SQL2GraphValidator`, unresolved помечены явно.
  - Рекурсивная функция не зацикливается (защита по глубине).

### A4. Интеграция в пайплайн и Web — ✅ ВЫПОЛНЕНО
- **Tasks:**
  - `SQL2GraphPipeline.run()`: параметр `parse_plpgsql: bool = False` (по умолчанию выключен — не ломает текущее поведение). При `True`: детект `CREATE FUNCTION ... LANGUAGE plpgsql` → маршрутизация в `PlpgsqlLineageExtractor`.
  - `Web/app.py`: чекбокс «Parse PL/pgSQL function bodies», загрузка .sql с процедурами, вывод unresolved-списка.
- **Acceptance:**
  - e2e-тест: «загрузил функцию → получил граф + unresolved-отчёт».
  - Поведение без `parse_plpgsql=True` не изменилось (существующие тесты зелёные).

---

## Фаза B — Извлечение из каталога GreenPlum

### B1. `Classes/gp_catalog.py`
- **Tasks:**
  - `class GPCatalogExtractor` (psycopg2, `read_only=True`, креды из env/`.env`):
    - функции: `SELECT n.nspname, p.proname, pg_get_functiondef(p.oid) FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace WHERE p.prolang IN (SELECT oid FROM pg_language WHERE lanname IN ('plpgsql','sql')) AND n.nspname NOT IN ('pg_catalog','information_schema','gp_toolkit');`
    - вьюхи: `pg_views`; матвьюхи: `pg_matviews`;
    - колонки: `information_schema.columns` (исключая системные схемы);
    - внешние таблицы GP (`pg_exttable`) — как source-узлы.
  - Методы: `dump_ddl_text() -> str` (формат `DDLParser.parse_registry`), `dump_csv(out_dir)`, `iter_functions()`.
  - CLI: `python -m Classes.gp_catalog --dsn "..." --out data/gp_dump/` (безопасно: `--read-only` по умолчанию).
- **Acceptance:**
  - Выгрузка работает с read-only ролью; формат совместим с `SchemaRegistry`.
  - Тест на фикстуре (мок-каталог): корректный DDL-текст.

### B2. Инкрементальность
- **Tasks:**
  - `data/gp_dump_state.json`: `{object_key: definition_hash}`; обрабатывать только изменённые объекты.
- **Acceptance:**
  - Повторный запуск без изменений — идемпотентен (не пересобирает граф).

---

## Фаза C — Гигиена кода

### C1. LICENSE — ✅ ВЫПОЛНЕНО
- **Tasks:** добавить `LICENSE` (MIT или согласованный со Сбером), поле `license = "MIT"` в `pyproject.toml`.
- **Acceptance:** `pip install .` не ругается на license; файл в корне.
- **Сделано:** `LICENSE` (MIT), `license = "MIT"` + `license-files` по PEP 639, `setuptools>=77`.

### C2. Чистка репозитория — ✅ ВЫПОЛНЕНО
- **Tasks:**
  - Удалить: `data_lineage_dag`, `my_lineage`, `data_lineage.dot`.
  - `.gitignore` += `*.dot`, `data/*.html`, `data/*.png`, `data/*_snapshot.json`.
  - Осознанные примеры перенести в `examples/data/`.
- **Acceptance:** `git status` чист после `git add .`.
- **Сделано:** три артефакта удалены; HTML/снапшоты сняты с отслеживания (файлы на диске сохранены).
  **Исключение:** `data/sqlglot_ddls10_first_snapshot.json` — тестовая фикстура
  (`tests/test_graph_drawer.py:7`), оставлена в git через negate-правило в `.gitignore`.

### C3. Декомпозиция `sql2graph_classes.py` → `Classes/sql2graph/`
- **Tasks:**
  - Создать пакет `Classes/sql2graph/`: `parser.py`, `builder.py`, `validator.py`, `llm_extractor.py`, `visualizer.py`.
  - `Classes/sql2graph/__init__.py` — реэкспорт всех публичных имён (`SQL2GraphParser`, `SQL2GraphPipeline`, `SQL2GraphBuilder`, `SQL2GraphValidator`, `SQL2GraphLLMExtractor`, `SQL2GraphVisualizer`, …) — обратная совместимость.
  - Внутренние перекрёстные импорты — по модулям.
- **Acceptance:**
  - `from Classes.sql2graph import SQL2GraphParser` и старый `from Classes.sql2graph_classes import SQL2GraphParser` — оба работают.
  - `pytest tests/ -q` зелёный.

### C4. Декомпозиция `Web/app.py`
- **Tasks:**
  - `Web/components/`: `sidebar.py`, `uploader.py`, `graph_view.py`, `results_panel.py`.
  - `Web/services/`: `pipeline_service.py`, `cache_service.py`.
  - `app.py` — только сборка и роутинг (Streamlit).
- **Acceptance:** приложение запускается (`streamlit run Web/app.py`), функциональность не потеряна (ручная проверка + AppTest smoke-тест).

### C5. Изоляция langchain — ✅ ВЫПОЛНЕНО (core без langchain)
- **Tasks:**
  - Новый `Classes/pipeline/core/llm_interface.py`:
    ```python
    class LLMInterface(Protocol):
        def invoke(self, prompt: str) -> str: ...
        def invoke_messages(self, messages) -> str: ...
    ```
  - `orchestrator.py`, `chain.py`, `model_classes.py`: убрать верхнеуровневые `from langchain_core...`; типизировать через `LLMInterface`.
  - langchain-адаптеры остаются только в `llm_factory.py` (уже есть `HuggingFaceLLMAdapter` — сделать общим паттерном).
  - `Config(llm_provider="mock")` — рабочий путь без langchain.
- **Acceptance:**
  - `pip install -e .` (без extra `[llm]`) → `PipelineOrchestrator(Config(llm_provider="mock")).run("SELECT 1")` работает.
  - `grep -rn "langchain" Classes/pipeline/core/` — только в `llm_factory.py` и `llm_interface.py` (импорт Protocol не считается).

---

## Фаза D — Structured output вместо regex

### D1–D2. JSON mode у провайдеров
- **Tasks:**
  - `Classes/pipeline/prompts/system.txt`, `human.txt`: добавить JSON-схему ответа (поля: `target`, `sources`, `reasoning`, `confidence`).
  - `llm_factory.py`: включать structured output/`response_format` у провайдеров, где поддерживается (OpenAI, Anthropic, GigaChat); HF — инструкция в промпте + `temperature=0.1`.
- **Acceptance:** golden-прогон: 100% ответов — валидный JSON по схеме.

### D3. Parser
- **Tasks:**
  - `SQLLineageOutputParser`: Pydantic-схема как основной путь; regex — только fallback: `confidence=0.3`, `provenance="regex"`.
  - Любой провал разбора → `parse_error` в результате (не молчать).
- **Acceptance:** тест «LLM вернул мусор → parse_error с confidence<0.5, без исключения вверх».

### D4. Provenance на всех рёбрах
- **Tasks:**
  - Все LLM-рёбра проходят `apply_edge_provenance(provenance, confidence)`; `verified=False` до прохода Reviewer (Фаза G).
- **Acceptance:** в graph JSON у каждого ребра есть `confidence` и `provenance`.

---

## Фаза E — CI/CD

### E1–E5
- **Tasks:**
  - `pytest-cov`: `--cov=Classes --cov-fail-under=80` (core), artifact с отчётом.
  - mypy: `mypy Classes/pipeline --strict` (в CI; начать с core).
  - CI на `uv` (uv.lock уже есть): `uv sync --extra llm --extra web --extra dev` + кэш `~/.cache/uv`.
  - Golden-drift: `tests/golden/test_golden.py` запускает `update_golden.py` в dry-run → падение при расхождении.
  - ruff: `select = ["E","F","I","W","UP","B","S"]` (S — bandit: секреты, SQL-инъекции, exec).
- **Acceptance:** CI зелёный на Python 3.10–3.13; coverage ≥80%; golden не дрейфует.

> **Решение 2026-08-07:** поддержка Python 3.9 снята (EOL октябрь 2025).
> `requires-python = ">=3.10"`, ruff `target-version = "py310"`, матрица CI — 3.10–3.13.

---

## Фаза F — Продакшн-интеграция (по готовности инфраструктуры)

### F1. REST API
- **Tasks:** `Web/api/main.py` (FastAPI): `GET /impact/{object}/{column}`, `GET /lineage/{object}`, `GET /coverage`, `GET /pii` (рекурсивные CTE по edges).
- **Acceptance:** тест: impact-запрос возвращает downstream-цепочку.

### F2. OpenLineage lifecycle
- **Tasks:** `openlineage_exporter.py`: полный run lifecycle (START → COMPLETE/FAIL), конфигурируемый `namespace`, маппинг `output.{alias}` → реальные таблицы из table-lineage.
- **Acceptance:** событие COMPLETE содержит `outputs` с реальными именами датасетов.

### F3. Airflow DAG
- **Tasks:** `dags/lineage_daily.py`: extract (GPCatalogExtractor) → parse → build → publish; сенсор на изменения репозитория.
- **Acceptance:** DAG импортируется без ошибок (`python -c "import dags.lineage_daily"`).

### F4. Визуализация
- **Tasks:** подключить `graph_drawer.py` к API (рендер DOT/Mermaid по запросу).
- **Acceptance:** `GET /lineage/{object}?format=dot` отдаёт валидный DOT.

---

## Фаза G — LLM-агенты (Resolver / Reviewer / Doc) — по готовности внутренней LLM

### G1. Resolver Agent
- **Tasks:** `Classes/agents/resolver_agent.py`: вход — функция + unresolved-отчёт; выход — кандидатные рёбра `{src, dst, transform_type, confidence, reasoning}`. Бюджет ≤30K токенов на объект; кэш по хэшу (`LLMCache`).
- **Acceptance:** на golden-set точность (по Reviewer) ≥90%.

### G2. Reviewer Agent
- **Tasks:** `Classes/agents/reviewer_agent.py`: сверка кандидатного ребра с исходным кодом (read-only) → PASS/FAIL + `sql_fragment`; **PASS только при подтверждении кодом**.
- **Acceptance:** рёбра с `verified=False` не публикуются; `verified=True` только после PASS.

### G3. Doc Agent
- **Tasks:** `Classes/agents/doc_agent.py`: документация → PII-метки, владельцы, описания (structured output).
- **Acceptance:** метки попадают в `columns.is_pii`.

### G4. Orchestrator агентов
- **Tasks:** `Classes/agents/orchestrator.py`: очередь unresolved → распределение → эскалация человеку после N попыток; coverage-отчёт.
- **Acceptance:** unresolved-очередь не растёт бесконечно (эскалация работает).

---

## Определение готовности (Definition of Done)

- [ ] Все фазы A–E выполнены, `pytest tests/ -q` зелёный
- [ ] `ruff check .` без замечаний
- [ ] CI зелёный на 3.9–3.12 (coverage ≥80%)
- [ ] Публичные контракты сохранены (реэкспорты на месте)
- [ ] Ни одного «молчаливого» провала: ошибки всегда в результате (`success=false`, `error`, `parse_error`)
- [ ] Golden-тесты не дрейфуют
- [ ] Каждая фаза — отдельный коммит с префиксом `phase <X>:`

---

## Запрещено (чтобы не сломать проект)

- ❌ Менять Data Contracts из SPEC без обновления SPEC
- ❌ Генерировать lineage-рёбра без уверенности (только unresolved/confidence<1)
- ❌ Ломать импорты `Classes.sql2graph_classes` и публичные имена (обратная совместимость)
- ❌ Добавлять langchain-импорты в core-модули (только через `LLMInterface`)
- ❌ Коммитить артефакты (dot/png/html/snapshots) и секреты (.env, токены)

---

*Сгенерировано по результатам ревью llm4lineage (коммит 7378bb5) и архитектуры lineage_greenplum_architecture.md. 2026-08-06.*
